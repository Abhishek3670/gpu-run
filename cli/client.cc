#include "cli/client.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>

#include <google/protobuf/empty.pb.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace gpu::cli {
namespace {

constexpr std::size_t kChunkSize = 64 * 1024;

std::vector<std::filesystem::path> EnumerateFiles(const std::filesystem::path& path) {
  std::vector<std::filesystem::path> files;
  if (std::filesystem::is_regular_file(path)) {
    files.push_back(path);
    return files;
  }

  for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
    if (entry.is_regular_file()) {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

bool IsExecutable(const std::filesystem::path& path) {
  const auto permissions = std::filesystem::status(path).permissions();
  return (permissions & std::filesystem::perms::owner_exec) != std::filesystem::perms::none;
}

}  // namespace

GpuRunClient::GpuRunClient(ClientOptions options)
    : options_(std::move(options)) {
  channel_ = grpc::CreateChannel(options_.server_address, grpc::InsecureChannelCredentials());
  stub_ = GpuService::NewStub(channel_);
}

absl::StatusOr<std::string> GpuRunClient::RunJob(const RunOptions& options) {
  auto bundle_id = UploadBundle(options.script_path);
  if (!bundle_id.ok()) {
    return bundle_id.status();
  }

  const std::string entrypoint = DefaultEntrypoint(options.script_path, options.entrypoint);
  if (entrypoint.empty()) {
    return absl::InvalidArgumentError("entrypoint is required when --script points to a directory");
  }
  return SubmitRemote(options, *bundle_id, entrypoint);
}

absl::StatusOr<JobStatusView> GpuRunClient::GetStatus(const std::string& job_id) {
  grpc::ClientContext context;
  ApplyMetadata(&context);

  GetStatusRequest request;
  request.set_job_id(job_id);
  GetStatusResponse response;
  const grpc::Status status = stub_->GetStatus(&context, request, &response);
  if (!status.ok()) {
    return FromGrpcStatus(status);
  }

  JobStatusView view;
  view.job_id = job_id;
  view.state = ToStateString(response.state());
  view.queue_position = response.queue_position();
  view.exit_code = response.exit_code();
  view.status_message = response.status_message();
  for (const auto gpu_id : response.assigned_gpu_ids()) {
    view.assigned_gpu_ids.push_back(static_cast<int>(gpu_id));
  }
  return view;
}

absl::StatusOr<std::vector<GpuInfoView>> GpuRunClient::ListGpus() {
  grpc::ClientContext context;
  ApplyMetadata(&context);

  google::protobuf::Empty request;
  ListGpusResponse response;
  const grpc::Status status = stub_->ListGPUs(&context, request, &response);
  if (!status.ok()) {
    return FromGrpcStatus(status);
  }

  std::vector<GpuInfoView> gpus;
  gpus.reserve(response.gpus_size());
  for (const auto& gpu_info : response.gpus()) {
    GpuInfoView view;
    view.gpu_id = static_cast<int>(gpu_info.gpu_id());
    view.model_name = gpu_info.model_name();
    view.total_memory_bytes = gpu_info.total_memory_bytes();
    view.used_memory_bytes = gpu_info.used_memory_bytes();
    view.utilization_percent = gpu_info.utilization_percent();
    view.available = gpu_info.available();
    view.locked_job_id = gpu_info.locked_job_id();
    gpus.push_back(std::move(view));
  }
  return gpus;
}

absl::Status GpuRunClient::StreamLogs(const std::string& job_id, std::ostream& stream) {
  grpc::ClientContext context;
  ApplyMetadata(&context);

  StreamLogsRequest request;
  request.set_job_id(job_id);
  auto reader = stub_->StreamLogs(&context, request);

  StreamLogsResponse chunk;
  while (reader->Read(&chunk)) {
    stream.write(chunk.data().data(), static_cast<std::streamsize>(chunk.data().size()));
    stream.flush();
  }

  return FromGrpcStatus(reader->Finish());
}

absl::Status GpuRunClient::CancelJob(const std::string& job_id) {
  grpc::ClientContext context;
  ApplyMetadata(&context);

  CancelJobRequest request;
  request.set_job_id(job_id);
  CancelJobResponse response;
  return FromGrpcStatus(stub_->CancelJob(&context, request, &response));
}

absl::StatusOr<std::string> GpuRunClient::UploadBundle(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    return absl::NotFoundError("script path does not exist");
  }

  grpc::ClientContext context;
  ApplyMetadata(&context);

  UploadBundleResponse response;
  auto writer = stub_->UploadBundle(&context, &response);

  const std::filesystem::path root = std::filesystem::is_regular_file(path) ? path.parent_path() : path;
  for (const auto& file_path : EnumerateFiles(path)) {
    const std::filesystem::path relative = std::filesystem::is_regular_file(path)
        ? file_path.filename()
        : std::filesystem::relative(file_path, root);

    std::ifstream input(file_path, std::ios::binary);
    if (!input.is_open()) {
      return absl::InternalError("failed to read input file");
    }

    std::array<char, kChunkSize> buffer{};
    bool sent_any_data = false;
    while (input.good()) {
      input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
      const std::streamsize bytes_read = input.gcount();
      if (bytes_read <= 0) {
        break;
      }

      UploadBundleChunk chunk;
      chunk.set_relative_path(relative.generic_string());
      chunk.set_executable(IsExecutable(file_path));
      chunk.set_data(buffer.data(), static_cast<int>(bytes_read));
      chunk.set_eof(false);
      if (!writer->Write(chunk)) {
        return absl::InternalError("bundle upload stream closed unexpectedly");
      }
      sent_any_data = true;
    }

    UploadBundleChunk eof_chunk;
    eof_chunk.set_relative_path(relative.generic_string());
    eof_chunk.set_executable(IsExecutable(file_path));
    eof_chunk.set_eof(true);
    if (!sent_any_data) {
      eof_chunk.clear_data();
    }
    if (!writer->Write(eof_chunk)) {
      return absl::InternalError("bundle upload stream closed before EOF");
    }
  }

  writer->WritesDone();
  const grpc::Status status = writer->Finish();
  if (!status.ok()) {
    return FromGrpcStatus(status);
  }
  return response.bundle_id();
}

absl::StatusOr<std::string> GpuRunClient::SubmitRemote(
    const RunOptions& options,
    const std::string& bundle_id,
    const std::string& entrypoint) {
  grpc::ClientContext context;
  ApplyMetadata(&context);

  SubmitJobRequest request;
  request.set_bundle_id(bundle_id);
  request.set_entrypoint(entrypoint);
  request.set_task_type(options.task_type);
  request.set_docker_image(options.docker_image);
  request.set_priority(options.priority);
  request.set_gpu_count(static_cast<std::uint32_t>(options.gpu_count));
  for (const auto& arg : options.args) {
    request.add_args(arg);
  }
  for (const int gpu_id : options.preferred_gpu_ids) {
    request.add_preferred_gpu_ids(static_cast<std::uint32_t>(gpu_id));
  }

  SubmitJobResponse response;
  const grpc::Status status = stub_->SubmitJob(&context, request, &response);
  if (!status.ok()) {
    return FromGrpcStatus(status);
  }
  return response.job_id();
}

void GpuRunClient::ApplyMetadata(grpc::ClientContext* context) const {
  if (options_.bearer_token.has_value() && !options_.bearer_token->empty()) {
    context->AddMetadata("authorization", "Bearer " + *options_.bearer_token);
  }
}

absl::Status GpuRunClient::FromGrpcStatus(const grpc::Status& status) {
  if (status.ok()) {
    return absl::OkStatus();
  }
  switch (status.error_code()) {
    case grpc::StatusCode::INVALID_ARGUMENT:
      return absl::InvalidArgumentError(status.error_message());
    case grpc::StatusCode::NOT_FOUND:
      return absl::NotFoundError(status.error_message());
    case grpc::StatusCode::UNAUTHENTICATED:
      return absl::UnauthenticatedError(status.error_message());
    case grpc::StatusCode::PERMISSION_DENIED:
      return absl::PermissionDeniedError(status.error_message());
    case grpc::StatusCode::RESOURCE_EXHAUSTED:
      return absl::ResourceExhaustedError(status.error_message());
    case grpc::StatusCode::FAILED_PRECONDITION:
      return absl::FailedPreconditionError(status.error_message());
    case grpc::StatusCode::UNAVAILABLE:
      return absl::UnavailableError(status.error_message());
    default:
      return absl::InternalError(status.error_message());
  }
}

std::string GpuRunClient::DefaultEntrypoint(
    const std::filesystem::path& script_path,
    const std::string& entrypoint) {
  if (!entrypoint.empty()) {
    return entrypoint;
  }
  if (std::filesystem::is_regular_file(script_path)) {
    return "./" + script_path.filename().generic_string();
  }
  return {};
}

std::string GpuRunClient::ToStateString(JobState state) {
  switch (state) {
    case JobState::QUEUED:
      return "QUEUED";
    case JobState::DISPATCHING:
      return "DISPATCHING";
    case JobState::RUNNING:
      return "RUNNING";
    case JobState::SUCCEEDED:
      return "SUCCEEDED";
    case JobState::FAILED:
      return "FAILED";
    case JobState::CANCELED:
      return "CANCELED";
    case JobState::JOB_STATE_UNSPECIFIED:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

}  // namespace gpu::cli
