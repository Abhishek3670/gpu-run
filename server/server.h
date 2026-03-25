#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "absl/status/status.h"
#include "gpu_manager/gpu_manager.h"
#include "gpu_service.grpc.pb.h"
#include "scheduler/scheduler.h"
#include "server/auth.h"
#include "worker/worker.h"

namespace gpu::server {

struct ServerConfig {
  std::string bind_address = "127.0.0.1:50051";
  std::filesystem::path bundle_root = "bundles";
  std::optional<std::string> auth_token;
  std::vector<std::string> allowed_images;
  std::string docker_path = "docker";
  std::string nvidia_smi_path = "nvidia-smi";
  std::chrono::milliseconds poll_interval{50};
  std::size_t max_log_entries = 1024;
};

class GpuServer {
 public:
  explicit GpuServer(ServerConfig config);

  [[nodiscard]] absl::Status Run();
  void Shutdown();

 private:
  class CallBase;
  class SubmitJobCall;
  class GetStatusCall;
  class ListGpusCall;
  class CancelJobCall;
  class UploadBundleCall;
  class StreamLogsCall;
  class TickCall;

  struct BundleInfo {
    std::string bundle_id;
    std::filesystem::path path;
    std::string digest;
    std::size_t active_jobs = 0;
  };

  friend class SubmitJobCall;
  friend class GetStatusCall;
  friend class ListGpusCall;
  friend class CancelJobCall;
  friend class UploadBundleCall;
  friend class StreamLogsCall;
  friend class TickCall;

  void SpawnHandlers();
  void Tick();

  [[nodiscard]] grpc::Status HandleSubmit(
      const grpc::ServerContext& context,
      const SubmitJobRequest& request,
      SubmitJobResponse* response);
  [[nodiscard]] grpc::Status HandleGetStatus(
      const grpc::ServerContext& context,
      const GetStatusRequest& request,
      GetStatusResponse* response);
  [[nodiscard]] grpc::Status HandleListGpus(
      const grpc::ServerContext& context,
      ListGpusResponse* response);
  [[nodiscard]] grpc::Status HandleCancel(
      const grpc::ServerContext& context,
      const CancelJobRequest& request,
      CancelJobResponse* response);

  [[nodiscard]] grpc::Status ValidateAuth(const grpc::ServerContext& context) const;
  [[nodiscard]] absl::StatusOr<BundleInfo> BeginBundleUpload();
  [[nodiscard]] absl::StatusOr<std::filesystem::path> ResolveBundlePath(const std::string& bundle_id) const;
  void RegisterBundle(BundleInfo bundle);
  void RecordBundleUse(const std::string& job_id, const std::string& bundle_id);
  void CleanupTerminalBundles();
  void ReleaseBundleForJob(const std::string& job_id);

  [[nodiscard]] static grpc::Status ToGrpcStatus(const absl::Status& status);
  [[nodiscard]] static scheduler::TaskType FromProtoTaskType(TaskType task_type);
  [[nodiscard]] static scheduler::JobPriority FromProtoPriority(Priority priority);
  [[nodiscard]] static JobState ToProtoJobState(scheduler::JobState state);
  [[nodiscard]] static LogSource ToProtoLogSource(worker::LogSource source);
  [[nodiscard]] static bool IsTerminalState(scheduler::JobState state);

  ServerConfig config_;
  TokenAuth auth_;
  scheduler::Scheduler scheduler_;
  manager::GpuManager gpu_manager_;
  worker::LogStreamer log_streamer_;
  worker::Worker worker_;

  GpuService::AsyncService service_;
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  std::unique_ptr<grpc::Server> server_;

  mutable std::mutex bundle_mutex_;
  std::unordered_map<std::string, BundleInfo> bundles_;
  std::unordered_map<std::string, std::string> job_bundles_;
  std::atomic<std::uint64_t> next_bundle_id_{1};
};

}  // namespace gpu::server

