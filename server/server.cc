#include "server/server.h"

#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>

#include <google/protobuf/empty.pb.h>
#include <grpc/support/time.h>
#include <grpcpp/alarm.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "worker/log_streamer.h"

namespace gpu::server {
namespace {

constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

std::string HexDigest(std::uint64_t value) {
  std::ostringstream stream;
  stream << std::hex << value;
  return stream.str();
}

void UpdateDigest(std::uint64_t* digest, std::string_view value) {
  for (const unsigned char byte : value) {
    *digest ^= byte;
    *digest *= kFnvPrime;
  }
}

bool IsRelativePathSafe(const std::filesystem::path& path) {
  if (path.empty() || path.is_absolute()) {
    return false;
  }
  for (const auto& part : path) {
    if (part == "..") {
      return false;
    }
  }
  return true;
}

gpr_timespec MillisecondsFromNow(std::chrono::milliseconds delay) {
  return gpr_time_add(gpr_now(GPR_CLOCK_MONOTONIC), gpr_time_from_millis(delay.count(), GPR_TIMESPAN));
}

}  // namespace

class GpuServer::CallBase {
 public:
  virtual ~CallBase() = default;
  virtual void Proceed(bool ok) = 0;
};

class GpuServer::SubmitJobCall final : public GpuServer::CallBase {
 public:
  explicit SubmitJobCall(GpuServer* server)
      : server_(server), responder_(&context_) {
    Proceed(true);
  }

  void Proceed(bool ok) override {
    if (state_ == State::kCreate) {
      state_ = State::kProcess;
      server_->service_.RequestSubmitJob(&context_, &request_, &responder_, server_->cq_.get(), server_->cq_.get(), this);
      return;
    }

    if (state_ == State::kProcess) {
      if (!ok) {
        delete this;
        return;
      }

      new SubmitJobCall(server_);
      const grpc::Status status = server_->HandleSubmit(context_, request_, &response_);
      state_ = State::kFinish;
      responder_.Finish(response_, status, this);
      return;
    }

    delete this;
  }

 private:
  enum class State {
    kCreate,
    kProcess,
    kFinish,
  };

  GpuServer* server_;
  State state_ = State::kCreate;
  grpc::ServerContext context_;
  SubmitJobRequest request_;
  SubmitJobResponse response_;
  grpc::ServerAsyncResponseWriter<SubmitJobResponse> responder_;
};

class GpuServer::GetStatusCall final : public GpuServer::CallBase {
 public:
  explicit GetStatusCall(GpuServer* server)
      : server_(server), responder_(&context_) {
    Proceed(true);
  }

  void Proceed(bool ok) override {
    if (state_ == State::kCreate) {
      state_ = State::kProcess;
      server_->service_.RequestGetStatus(&context_, &request_, &responder_, server_->cq_.get(), server_->cq_.get(), this);
      return;
    }

    if (state_ == State::kProcess) {
      if (!ok) {
        delete this;
        return;
      }

      new GetStatusCall(server_);
      const grpc::Status status = server_->HandleGetStatus(context_, request_, &response_);
      state_ = State::kFinish;
      responder_.Finish(response_, status, this);
      return;
    }

    delete this;
  }

 private:
  enum class State {
    kCreate,
    kProcess,
    kFinish,
  };

  GpuServer* server_;
  State state_ = State::kCreate;
  grpc::ServerContext context_;
  GetStatusRequest request_;
  GetStatusResponse response_;
  grpc::ServerAsyncResponseWriter<GetStatusResponse> responder_;
};

class GpuServer::ListGpusCall final : public GpuServer::CallBase {
 public:
  explicit ListGpusCall(GpuServer* server)
      : server_(server), responder_(&context_) {
    Proceed(true);
  }

  void Proceed(bool ok) override {
    if (state_ == State::kCreate) {
      state_ = State::kProcess;
      server_->service_.RequestListGPUs(&context_, &request_, &responder_, server_->cq_.get(), server_->cq_.get(), this);
      return;
    }

    if (state_ == State::kProcess) {
      if (!ok) {
        delete this;
        return;
      }

      new ListGpusCall(server_);
      const grpc::Status status = server_->HandleListGpus(context_, &response_);
      state_ = State::kFinish;
      responder_.Finish(response_, status, this);
      return;
    }

    delete this;
  }

 private:
  enum class State {
    kCreate,
    kProcess,
    kFinish,
  };

  GpuServer* server_;
  State state_ = State::kCreate;
  grpc::ServerContext context_;
  google::protobuf::Empty request_;
  ListGpusResponse response_;
  grpc::ServerAsyncResponseWriter<ListGpusResponse> responder_;
};

class GpuServer::CancelJobCall final : public GpuServer::CallBase {
 public:
  explicit CancelJobCall(GpuServer* server)
      : server_(server), responder_(&context_) {
    Proceed(true);
  }

  void Proceed(bool ok) override {
    if (state_ == State::kCreate) {
      state_ = State::kProcess;
      server_->service_.RequestCancelJob(&context_, &request_, &responder_, server_->cq_.get(), server_->cq_.get(), this);
      return;
    }

    if (state_ == State::kProcess) {
      if (!ok) {
        delete this;
        return;
      }

      new CancelJobCall(server_);
      const grpc::Status status = server_->HandleCancel(context_, request_, &response_);
      state_ = State::kFinish;
      responder_.Finish(response_, status, this);
      return;
    }

    delete this;
  }

 private:
  enum class State {
    kCreate,
    kProcess,
    kFinish,
  };

  GpuServer* server_;
  State state_ = State::kCreate;
  grpc::ServerContext context_;
  CancelJobRequest request_;
  CancelJobResponse response_;
  grpc::ServerAsyncResponseWriter<CancelJobResponse> responder_;
};

class GpuServer::UploadBundleCall final : public GpuServer::CallBase {
 public:
  explicit UploadBundleCall(GpuServer* server)
      : server_(server), reader_(&context_) {
    Proceed(true);
  }

  void Proceed(bool ok) override {
    if (state_ == State::kCreate) {
      state_ = State::kRead;
      server_->service_.RequestUploadBundle(&context_, &reader_, server_->cq_.get(), server_->cq_.get(), this);
      return;
    }

    if (state_ == State::kRead) {
      if (!started_) {
        if (!ok) {
          delete this;
          return;
        }

        new UploadBundleCall(server_);
        started_ = true;
        const grpc::Status auth_status = server_->ValidateAuth(context_);
        if (!auth_status.ok()) {
          state_ = State::kFinish;
          reader_.Finish(response_, auth_status, this);
          return;
        }

        auto bundle = server_->BeginBundleUpload();
        if (!bundle.ok()) {
          state_ = State::kFinish;
          reader_.Finish(response_, GpuServer::ToGrpcStatus(bundle.status()), this);
          return;
        }
        bundle_ = std::move(*bundle);
        reader_.Read(&chunk_, this);
        return;
      }

      if (!ok) {
        CloseCurrentFile();
        if (!status_.ok()) {
          state_ = State::kFinish;
          reader_.Finish(response_, GpuServer::ToGrpcStatus(status_), this);
          return;
        }
        if (!received_chunks_) {
          state_ = State::kFinish;
          reader_.Finish(response_, grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "bundle stream was empty"), this);
          return;
        }

        bundle_.digest = HexDigest(digest_);
        server_->RegisterBundle(bundle_);
        response_.set_bundle_id(bundle_.bundle_id);
        response_.set_digest(bundle_.digest);
        response_.set_staged_path(bundle_.path.string());
        state_ = State::kFinish;
        reader_.Finish(response_, grpc::Status::OK, this);
        return;
      }

      received_chunks_ = true;
      status_ = ProcessChunk(chunk_);
      if (!status_.ok()) {
        CloseCurrentFile();
        state_ = State::kFinish;
        reader_.Finish(response_, GpuServer::ToGrpcStatus(status_), this);
        return;
      }
      reader_.Read(&chunk_, this);
      return;
    }

    delete this;
  }

 private:
  enum class State {
    kCreate,
    kRead,
    kFinish,
  };

  absl::Status ProcessChunk(const UploadBundleChunk& chunk) {
    const std::filesystem::path relative_path = chunk.relative_path();
    if (!IsRelativePathSafe(relative_path)) {
      return absl::InvalidArgumentError("relative_path must stay within the bundle root");
    }

    const std::filesystem::path absolute_path = bundle_.path / relative_path;
    if (current_relative_path_ != relative_path.generic_string()) {
      CloseCurrentFile();
      std::error_code error;
      std::filesystem::create_directories(absolute_path.parent_path(), error);
      if (error) {
        return absl::InternalError("failed to create bundle directories");
      }
      current_file_.open(absolute_path, std::ios::binary | std::ios::trunc);
      if (!current_file_.is_open()) {
        return absl::InternalError("failed to open bundle file for writing");
      }
      current_relative_path_ = relative_path.generic_string();
      current_file_executable_ = chunk.executable();
      UpdateDigest(&digest_, current_relative_path_);
    }

    if (!chunk.data().empty()) {
      current_file_.write(chunk.data().data(), static_cast<std::streamsize>(chunk.data().size()));
      if (!current_file_) {
        return absl::InternalError("failed while writing bundle data");
      }
      UpdateDigest(&digest_, chunk.data());
    }

    if (chunk.eof()) {
      CloseCurrentFile();
      if (current_file_executable_) {
        std::error_code error;
        std::filesystem::permissions(
            absolute_path,
            std::filesystem::perms::owner_exec |
                std::filesystem::perms::group_exec |
                std::filesystem::perms::others_exec,
            std::filesystem::perm_options::add,
            error);
      }
      current_relative_path_.clear();
      current_file_executable_ = false;
    }

    return absl::OkStatus();
  }

  void CloseCurrentFile() {
    if (current_file_.is_open()) {
      current_file_.close();
    }
  }

  GpuServer* server_;
  State state_ = State::kCreate;
  bool started_ = false;
  bool received_chunks_ = false;
  std::uint64_t digest_ = kFnvOffsetBasis;
  BundleInfo bundle_;
  grpc::ServerContext context_;
  UploadBundleChunk chunk_;
  UploadBundleResponse response_;
  grpc::ServerAsyncReader<UploadBundleResponse, UploadBundleChunk> reader_;
  std::ofstream current_file_;
  std::string current_relative_path_;
  bool current_file_executable_ = false;
  absl::Status status_ = absl::OkStatus();
};

class GpuServer::StreamLogsCall final : public GpuServer::CallBase {
 public:
  explicit StreamLogsCall(GpuServer* server)
      : server_(server), writer_(&context_) {
    Proceed(true);
  }

  void Proceed(bool ok) override {
    if (state_ == State::kCreate) {
      state_ = State::kStreaming;
      server_->service_.RequestStreamLogs(&context_, &request_, &writer_, server_->cq_.get(), server_->cq_.get(), this);
      return;
    }

    if (!started_) {
      if (!ok) {
        delete this;
        return;
      }

      new StreamLogsCall(server_);
      started_ = true;
      const grpc::Status auth_status = server_->ValidateAuth(context_);
      if (!auth_status.ok()) {
        state_ = State::kFinish;
        writer_.Finish(auth_status, this);
        return;
      }
      if (request_.job_id().empty()) {
        state_ = State::kFinish;
        writer_.Finish(grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "job_id is required"), this);
        return;
      }
      Pump();
      return;
    }

    if (!ok) {
      delete this;
      return;
    }

    if (state_ == State::kStreaming) {
      Pump();
      return;
    }

    delete this;
  }

 private:
  enum class State {
    kCreate,
    kStreaming,
    kFinish,
  };

  void Pump() {
    if (!pending_entries_.empty()) {
      WriteNext();
      return;
    }

    auto logs = server_->log_streamer_.ReadSince(request_.job_id(), last_sequence_);
    if (!logs.ok()) {
      state_ = State::kFinish;
      writer_.Finish(GpuServer::ToGrpcStatus(logs.status()), this);
      return;
    }

    for (const auto& entry : logs->entries) {
      pending_entries_.push_back(entry);
    }

    if (!pending_entries_.empty()) {
      WriteNext();
      return;
    }

    if (logs->complete) {
      state_ = State::kFinish;
      writer_.Finish(grpc::Status::OK, this);
      return;
    }

    alarm_.Set(server_->cq_.get(), MillisecondsFromNow(server_->config_.poll_interval), this);
  }

  void WriteNext() {
    const worker::LogEntry entry = pending_entries_.front();
    pending_entries_.pop_front();
    last_sequence_ = entry.sequence;

    response_buffer_.set_sequence(entry.sequence);
    response_buffer_.set_timestamp_unix_ms(entry.timestamp_unix_ms);
    response_buffer_.set_source(GpuServer::ToProtoLogSource(entry.source));
    response_buffer_.set_data(entry.payload);
    response_buffer_.set_complete(false);
    writer_.Write(response_buffer_, this);
  }

  GpuServer* server_;
  State state_ = State::kCreate;
  bool started_ = false;
  grpc::ServerContext context_;
  StreamLogsRequest request_;
  grpc::ServerAsyncWriter<StreamLogsResponse> writer_;
  grpc::Alarm alarm_;
  StreamLogsResponse response_buffer_;
  std::int64_t last_sequence_ = 0;
  std::deque<worker::LogEntry> pending_entries_;
};

class GpuServer::TickCall final : public GpuServer::CallBase {
 public:
  explicit TickCall(GpuServer* server) : server_(server) {
    Arm();
  }

  void Proceed(bool ok) override {
    if (!ok) {
      delete this;
      return;
    }

    server_->Tick();
    Arm();
  }

 private:
  void Arm() {
    alarm_.Set(server_->cq_.get(), MillisecondsFromNow(server_->config_.poll_interval), this);
  }

  GpuServer* server_;
  grpc::Alarm alarm_;
};

GpuServer::GpuServer(ServerConfig config)
    : config_(std::move(config)),
      auth_(config_.auth_token),
      gpu_manager_(config_.nvidia_smi_path),
      log_streamer_(config_.max_log_entries),
      worker_({config_.docker_path, config_.allowed_images}, &log_streamer_) {}

absl::Status GpuServer::Run() {
  std::error_code error;
  std::filesystem::create_directories(config_.bundle_root, error);
  if (error) {
    return absl::InternalError("failed to create bundle root");
  }

  absl::Status status = gpu_manager_.Initialize();
  if (!status.ok()) {
    return status;
  }
  status = worker_.Initialize();
  if (!status.ok()) {
    return status;
  }

  grpc::ServerBuilder builder;
  builder.AddListeningPort(config_.bind_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service_);
  cq_ = builder.AddCompletionQueue();
  server_ = builder.BuildAndStart();
  if (!server_) {
    return absl::InternalError("failed to start gRPC server");
  }

  std::cout << "gpu-server listening on " << config_.bind_address << '\n';
  SpawnHandlers();
  new TickCall(this);

  void* tag = nullptr;
  bool ok = false;
  while (cq_->Next(&tag, &ok)) {
    static_cast<CallBase*>(tag)->Proceed(ok);
  }

  return absl::OkStatus();
}

void GpuServer::Shutdown() {
  if (server_) {
    server_->Shutdown();
  }
  if (cq_) {
    cq_->Shutdown();
  }
}

void GpuServer::SpawnHandlers() {
  new SubmitJobCall(this);
  new GetStatusCall(this);
  new ListGpusCall(this);
  new CancelJobCall(this);
  new UploadBundleCall(this);
  new StreamLogsCall(this);
}

void GpuServer::Tick() {
  for (const auto& event : worker_.PollEvents()) {
    const absl::Status unlock_status = gpu_manager_.UnlockGpus(event.job_id);
    (void)unlock_status;
    const absl::Status finished_status = scheduler_.OnWorkerFinished(event.job_id, event.exit_code, event.status_message);
    (void)finished_status;
  }
  scheduler_.TryDispatch(gpu_manager_, worker_);
  CleanupTerminalBundles();
}

grpc::Status GpuServer::HandleSubmit(
    const grpc::ServerContext& context,
    const SubmitJobRequest& request,
    SubmitJobResponse* response) {
  const grpc::Status auth_status = ValidateAuth(context);
  if (!auth_status.ok()) {
    return auth_status;
  }
  if (request.bundle_id().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "bundle_id is required");
  }
  if (request.entrypoint().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "entrypoint is required");
  }
  if (request.docker_image().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "docker_image is required");
  }
  if (request.task_type() == TaskType::TASK_TYPE_UNSPECIFIED) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "task_type is required");
  }
  if (request.priority() == Priority::PRIORITY_UNSPECIFIED) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "priority is required");
  }
  if (request.gpu_count() == 0) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "gpu_count must be at least one");
  }
  if (request.preferred_gpu_ids_size() > 0 && request.preferred_gpu_ids_size() < request.gpu_count()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "preferred_gpu_ids must cover gpu_count when provided");
  }

  auto bundle_path = ResolveBundlePath(request.bundle_id());
  if (!bundle_path.ok()) {
    return ToGrpcStatus(bundle_path.status());
  }

  scheduler::JobRequest job_request;
  job_request.bundle_id = request.bundle_id();
  job_request.bundle_path = bundle_path->string();
  job_request.entrypoint = request.entrypoint();
  job_request.args.assign(request.args().begin(), request.args().end());
  job_request.task_type = FromProtoTaskType(request.task_type());
  job_request.priority = FromProtoPriority(request.priority());
  job_request.docker_image = request.docker_image();
  job_request.gpu_count = static_cast<int>(request.gpu_count());
  for (const std::uint32_t gpu_id : request.preferred_gpu_ids()) {
    job_request.preferred_gpu_ids.push_back(static_cast<int>(gpu_id));
  }

  auto job_id = scheduler_.Submit(std::move(job_request));
  if (!job_id.ok()) {
    return ToGrpcStatus(job_id.status());
  }

  RecordBundleUse(*job_id, request.bundle_id());
  response->set_job_id(*job_id);
  response->set_state(JobState::QUEUED);
  scheduler_.TryDispatch(gpu_manager_, worker_);
  CleanupTerminalBundles();
  return grpc::Status::OK;
}

grpc::Status GpuServer::HandleGetStatus(
    const grpc::ServerContext& context,
    const GetStatusRequest& request,
    GetStatusResponse* response) {
  const grpc::Status auth_status = ValidateAuth(context);
  if (!auth_status.ok()) {
    return auth_status;
  }

  auto status = scheduler_.GetStatus(request.job_id());
  if (!status.ok()) {
    return ToGrpcStatus(status.status());
  }

  response->set_state(ToProtoJobState(status->state));
  response->set_queue_position(static_cast<std::uint32_t>(status->queue_position));
  response->set_exit_code(status->exit_code.value_or(-1));
  response->set_status_message(status->status_message);
  for (const int gpu_id : status->assigned_gpu_ids) {
    response->add_assigned_gpu_ids(static_cast<std::uint32_t>(gpu_id));
  }
  return grpc::Status::OK;
}

grpc::Status GpuServer::HandleListGpus(
    const grpc::ServerContext& context,
    ListGpusResponse* response) {
  const grpc::Status auth_status = ValidateAuth(context);
  if (!auth_status.ok()) {
    return auth_status;
  }

  auto metrics = gpu_manager_.Snapshot();
  if (!metrics.ok()) {
    return ToGrpcStatus(metrics.status());
  }

  for (const auto& metric : *metrics) {
    auto* gpu_info = response->add_gpus();
    gpu_info->set_gpu_id(static_cast<std::uint32_t>(metric.gpu_id));
    gpu_info->set_model_name(metric.model_name);
    gpu_info->set_total_memory_bytes(metric.total_memory_bytes);
    gpu_info->set_used_memory_bytes(metric.used_memory_bytes);
    gpu_info->set_utilization_percent(metric.utilization_percent);
    gpu_info->set_available(metric.available);
    gpu_info->set_locked_job_id(metric.locked_job_id);
  }

  return grpc::Status::OK;
}

grpc::Status GpuServer::HandleCancel(
    const grpc::ServerContext& context,
    const CancelJobRequest& request,
    CancelJobResponse* response) {
  const grpc::Status auth_status = ValidateAuth(context);
  if (!auth_status.ok()) {
    return auth_status;
  }

  auto cancellation = scheduler_.Cancel(request.job_id());
  if (!cancellation.ok()) {
    return ToGrpcStatus(cancellation.status());
  }

  if (cancellation->requires_worker_cancel) {
    absl::Status status = worker_.CancelJob(request.job_id());
    if (!status.ok()) {
      return ToGrpcStatus(status);
    }
  }

  response->set_job_id(cancellation->view.job_id);
  response->set_state(ToProtoJobState(cancellation->view.state));
  response->set_status_message(cancellation->view.status_message);
  CleanupTerminalBundles();
  return grpc::Status::OK;
}

grpc::Status GpuServer::ValidateAuth(const grpc::ServerContext& context) const {
  return ToGrpcStatus(auth_.Validate(context));
}

absl::StatusOr<GpuServer::BundleInfo> GpuServer::BeginBundleUpload() {
  const std::uint64_t ordinal = next_bundle_id_.fetch_add(1);
  BundleInfo info;
  info.bundle_id = "bundle-" + std::to_string(ordinal);
  info.path = config_.bundle_root / info.bundle_id;

  std::error_code error;
  std::filesystem::create_directories(info.path, error);
  if (error) {
    return absl::InternalError("failed to create bundle staging directory");
  }
  return info;
}

absl::StatusOr<std::filesystem::path> GpuServer::ResolveBundlePath(const std::string& bundle_id) const {
  std::lock_guard<std::mutex> lock(bundle_mutex_);
  const auto it = bundles_.find(bundle_id);
  if (it == bundles_.end()) {
    return absl::NotFoundError("unknown bundle id");
  }
  return it->second.path;
}

void GpuServer::RegisterBundle(BundleInfo bundle) {
  std::lock_guard<std::mutex> lock(bundle_mutex_);
  bundles_[bundle.bundle_id] = std::move(bundle);
}

void GpuServer::RecordBundleUse(const std::string& job_id, const std::string& bundle_id) {
  std::lock_guard<std::mutex> lock(bundle_mutex_);
  const auto it = bundles_.find(bundle_id);
  if (it == bundles_.end()) {
    return;
  }

  ++it->second.active_jobs;
  job_bundles_[job_id] = bundle_id;
}

void GpuServer::CleanupTerminalBundles() {
  for (const auto& job : scheduler_.Snapshot()) {
    if (!IsTerminalState(job.state)) {
      continue;
    }
    ReleaseBundleForJob(job.job_id);
  }
}

void GpuServer::ReleaseBundleForJob(const std::string& job_id) {
  std::filesystem::path bundle_path;
  {
    std::lock_guard<std::mutex> lock(bundle_mutex_);
    const auto job_it = job_bundles_.find(job_id);
    if (job_it == job_bundles_.end()) {
      return;
    }

    const auto bundle_it = bundles_.find(job_it->second);
    if (bundle_it != bundles_.end()) {
      if (bundle_it->second.active_jobs > 0) {
        --bundle_it->second.active_jobs;
      }
      if (bundle_it->second.active_jobs == 0) {
        bundle_path = bundle_it->second.path;
        bundles_.erase(bundle_it);
      }
    }

    job_bundles_.erase(job_it);
  }

  if (bundle_path.empty()) {
    return;
  }

  std::error_code error;
  std::filesystem::remove_all(bundle_path, error);
  if (error) {
    std::cerr << "failed to remove bundle path " << bundle_path << ": " << error.message() << '\n';
  }
}

grpc::Status GpuServer::ToGrpcStatus(const absl::Status& status) {
  switch (status.code()) {
    case absl::StatusCode::kOk:
      return grpc::Status::OK;
    case absl::StatusCode::kInvalidArgument:
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, std::string(status.message()));
    case absl::StatusCode::kNotFound:
      return grpc::Status(grpc::StatusCode::NOT_FOUND, std::string(status.message()));
    case absl::StatusCode::kAlreadyExists:
      return grpc::Status(grpc::StatusCode::ALREADY_EXISTS, std::string(status.message()));
    case absl::StatusCode::kPermissionDenied:
      return grpc::Status(grpc::StatusCode::PERMISSION_DENIED, std::string(status.message()));
    case absl::StatusCode::kUnauthenticated:
      return grpc::Status(grpc::StatusCode::UNAUTHENTICATED, std::string(status.message()));
    case absl::StatusCode::kResourceExhausted:
      return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, std::string(status.message()));
    case absl::StatusCode::kFailedPrecondition:
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, std::string(status.message()));
    case absl::StatusCode::kUnavailable:
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, std::string(status.message()));
    default:
      return grpc::Status(grpc::StatusCode::INTERNAL, std::string(status.message()));
  }
}

scheduler::TaskType GpuServer::FromProtoTaskType(TaskType task_type) {
  return task_type == TaskType::TRAINING ? scheduler::TaskType::kTraining : scheduler::TaskType::kCompute;
}

scheduler::JobPriority GpuServer::FromProtoPriority(Priority priority) {
  switch (priority) {
    case Priority::HIGH:
      return scheduler::JobPriority::kHigh;
    case Priority::LOW:
      return scheduler::JobPriority::kLow;
    case Priority::MEDIUM:
    case Priority::PRIORITY_UNSPECIFIED:
      return scheduler::JobPriority::kMedium;
  }

  return scheduler::JobPriority::kMedium;
}

JobState GpuServer::ToProtoJobState(scheduler::JobState state) {
  switch (state) {
    case scheduler::JobState::kQueued:
      return JobState::QUEUED;
    case scheduler::JobState::kDispatching:
      return JobState::DISPATCHING;
    case scheduler::JobState::kRunning:
      return JobState::RUNNING;
    case scheduler::JobState::kSucceeded:
      return JobState::SUCCEEDED;
    case scheduler::JobState::kFailed:
      return JobState::FAILED;
    case scheduler::JobState::kCanceled:
      return JobState::CANCELED;
  }

  return JobState::JOB_STATE_UNSPECIFIED;
}

LogSource GpuServer::ToProtoLogSource(worker::LogSource source) {
  return source == worker::LogSource::kStdout ? LogSource::STDOUT : LogSource::STDERR;
}

bool GpuServer::IsTerminalState(scheduler::JobState state) {
  switch (state) {
    case scheduler::JobState::kSucceeded:
    case scheduler::JobState::kFailed:
    case scheduler::JobState::kCanceled:
      return true;
    case scheduler::JobState::kQueued:
    case scheduler::JobState::kDispatching:
    case scheduler::JobState::kRunning:
      return false;
  }

  return false;
}

}  // namespace gpu::server






