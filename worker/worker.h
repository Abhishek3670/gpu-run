#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/types.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "scheduler/job.h"
#include "worker/log_streamer.h"

namespace gpu::worker {

struct WorkerConfig {
  std::string docker_path = "docker";
  std::vector<std::string> allowed_images;
};

struct LaunchResult {
  pid_t pid = -1;
};

struct WorkerEvent {
  std::string job_id;
  std::vector<int> gpu_ids;
  int exit_code = 0;
  bool canceled = false;
  std::string status_message;
};

class Worker {
 public:
  Worker(WorkerConfig config, LogStreamer* log_streamer);

  [[nodiscard]] absl::Status Initialize();
  [[nodiscard]] absl::StatusOr<LaunchResult> StartJob(
      const scheduler::Job& job,
      const std::vector<int>& gpu_ids);
  [[nodiscard]] absl::Status CancelJob(const std::string& job_id);
  [[nodiscard]] std::vector<WorkerEvent> PollEvents();

 private:
  struct RunningJob {
    std::string job_id;
    pid_t pid = -1;
    std::vector<int> gpu_ids;
    int stdout_fd = -1;
    int stderr_fd = -1;
    bool cancel_requested = false;
  };

  [[nodiscard]] absl::StatusOr<std::string> ResolveDockerBinary() const;
  [[nodiscard]] absl::Status ValidateConfiguredImages() const;
  [[nodiscard]] absl::StatusOr<std::vector<std::string>> BuildDockerArgs(
      const scheduler::Job& job,
      const std::vector<int>& gpu_ids) const;
  [[nodiscard]] absl::Status RunDockerKill(const std::string& job_id) const;
  [[nodiscard]] static std::string ContainerName(const std::string& job_id);
  [[nodiscard]] static std::string JoinGpuIds(const std::vector<int>& gpu_ids);
  static void DrainFd(int fd, const std::string& job_id, LogSource source, LogStreamer* log_streamer);
  static void SetNonBlocking(int fd);

  WorkerConfig config_;
  std::string docker_binary_;
  LogStreamer* log_streamer_ = nullptr;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, RunningJob> running_jobs_;
};

}  // namespace gpu::worker
