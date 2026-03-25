#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gpu_manager/metrics.h"

namespace gpu::manager {

class GpuManager {
 public:
  explicit GpuManager(std::string nvidia_smi_path = "nvidia-smi");

  [[nodiscard]] absl::Status Initialize();
  [[nodiscard]] absl::StatusOr<std::vector<GpuMetrics>> Snapshot();
  [[nodiscard]] absl::StatusOr<std::vector<int>> TryLockGpus(
      const std::string& job_id,
      int count,
      const std::vector<int>& preferred_gpu_ids);
  [[nodiscard]] absl::Status UnlockGpus(const std::string& job_id);

 private:
  [[nodiscard]] absl::StatusOr<std::vector<GpuMetrics>> FetchInventory();
  [[nodiscard]] absl::StatusOr<std::vector<GpuMetrics>> FetchFromNvml() const;
  [[nodiscard]] absl::StatusOr<std::vector<GpuMetrics>> FetchFromNvidiaSmi() const;
  [[nodiscard]] absl::StatusOr<std::string> RunCommandCapture(const std::string& command) const;

  std::string nvidia_smi_path_;
  bool use_nvml_ = true;

  mutable std::mutex mutex_;
  std::vector<GpuMetrics> last_snapshot_;
  std::unordered_map<int, std::string> lock_owners_;
};

}  // namespace gpu::manager
