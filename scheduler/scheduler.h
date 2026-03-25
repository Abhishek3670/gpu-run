#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "scheduler/job.h"

namespace gpu::manager {
class GpuManager;
}

namespace gpu::worker {
class Worker;
}

namespace gpu::scheduler {

class Scheduler {
 public:
  Scheduler() = default;

  [[nodiscard]] absl::StatusOr<std::string> Submit(JobRequest request);
  [[nodiscard]] absl::StatusOr<JobStatusView> GetStatus(const std::string& job_id) const;
  [[nodiscard]] absl::StatusOr<CancelAction> Cancel(const std::string& job_id);
  [[nodiscard]] std::vector<JobStatusView> Snapshot() const;

  void TryDispatch(manager::GpuManager& gpu_manager, worker::Worker& worker);
  [[nodiscard]] absl::Status OnWorkerStarted(
      const std::string& job_id,
      const std::vector<int>& gpu_ids,
      pid_t worker_pid);
  [[nodiscard]] absl::Status OnWorkerFinished(
      const std::string& job_id,
      int exit_code,
      const std::string& status_message);

 private:
  struct DispatchCandidate {
    Job snapshot;
    JobPriority priority = JobPriority::kMedium;
  };

  [[nodiscard]] static std::size_t QueueIndex(JobPriority priority);
  [[nodiscard]] std::optional<DispatchCandidate> PopNextCandidateLocked();
  [[nodiscard]] JobStatusView MakeViewLocked(const Job& job) const;
  void RequeueLocked(const std::string& job_id, JobPriority priority);
  void FailJobLocked(Job& job, const std::string& message);

  mutable std::mutex mutex_;
  std::unordered_map<std::string, Job> jobs_;
  std::array<std::deque<std::string>, 3> queues_;
  std::uint64_t next_job_id_ = 1;
};

}  // namespace gpu::scheduler
