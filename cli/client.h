#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gpu_service.grpc.pb.h"

namespace gpu::cli {

struct ClientOptions {
  std::string server_address = "127.0.0.1:50051";
  std::optional<std::string> bearer_token;
};

struct RunOptions {
  std::filesystem::path script_path;
  std::string docker_image;
  std::string entrypoint;
  std::vector<std::string> args;
  TaskType task_type = TaskType::COMPUTE;
  Priority priority = Priority::MEDIUM;
  int gpu_count = 1;
  std::vector<int> preferred_gpu_ids;
};

struct JobStatusView {
  std::string job_id;
  std::string state;
  std::vector<int> assigned_gpu_ids;
  std::uint32_t queue_position = 0;
  int exit_code = -1;
  std::string status_message;
};

struct GpuInfoView {
  int gpu_id = -1;
  std::string model_name;
  std::uint64_t total_memory_bytes = 0;
  std::uint64_t used_memory_bytes = 0;
  std::uint32_t utilization_percent = 0;
  bool available = false;
  std::string locked_job_id;
};

class GpuRunClient {
 public:
  explicit GpuRunClient(ClientOptions options);

  [[nodiscard]] absl::StatusOr<std::string> RunJob(const RunOptions& options);
  [[nodiscard]] absl::StatusOr<JobStatusView> GetStatus(const std::string& job_id);
  [[nodiscard]] absl::StatusOr<std::vector<GpuInfoView>> ListGpus();
  [[nodiscard]] absl::Status StreamLogs(const std::string& job_id, std::ostream& stream);
  [[nodiscard]] absl::Status CancelJob(const std::string& job_id);

 private:
  [[nodiscard]] absl::StatusOr<std::string> UploadBundle(const std::filesystem::path& path);
  [[nodiscard]] absl::StatusOr<std::string> SubmitRemote(
      const RunOptions& options,
      const std::string& bundle_id,
      const std::string& entrypoint);
  void ApplyMetadata(grpc::ClientContext* context) const;

  static absl::Status FromGrpcStatus(const grpc::Status& status);
  static std::string DefaultEntrypoint(const std::filesystem::path& script_path, const std::string& entrypoint);
  static std::string ToStateString(JobState state);

  ClientOptions options_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<GpuService::Stub> stub_;
};

}  // namespace gpu::cli
