#include "gpu_manager/gpu_manager.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <utility>

#include <nvml.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"

namespace gpu::manager {
namespace {

constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;

std::string NvmlErrorToString(nvmlReturn_t result) {
  return nvmlErrorString(result);
}

}  // namespace

GpuManager::GpuManager(std::string nvidia_smi_path)
    : nvidia_smi_path_(std::move(nvidia_smi_path)) {}

absl::Status GpuManager::Initialize() {
  auto snapshot = FetchInventory();
  if (!snapshot.ok()) {
    return snapshot.status();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  last_snapshot_ = std::move(*snapshot);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<GpuMetrics>> GpuManager::Snapshot() {
  auto snapshot = FetchInventory();
  if (!snapshot.ok()) {
    return snapshot.status();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  last_snapshot_ = *snapshot;
  for (auto& metric : last_snapshot_) {
    const auto owner = lock_owners_.find(metric.gpu_id);
    if (owner != lock_owners_.end()) {
      metric.available = false;
      metric.locked_job_id = owner->second;
    } else {
      metric.available = true;
      metric.locked_job_id.clear();
    }
  }
  return last_snapshot_;
}

absl::StatusOr<std::vector<int>> GpuManager::TryLockGpus(
    const std::string& job_id,
    int count,
    const std::vector<int>& preferred_gpu_ids) {
  if (count < 1) {
    return absl::InvalidArgumentError("count must be at least one");
  }

  auto snapshot = Snapshot();
  if (!snapshot.ok()) {
    return snapshot.status();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<int> selected;
  selected.reserve(count);
  std::unordered_set<int> seen;

  const auto try_add = [&](int gpu_id) {
    if (selected.size() >= static_cast<std::size_t>(count) || !seen.insert(gpu_id).second) {
      return;
    }
    const auto owner = lock_owners_.find(gpu_id);
    if (owner != lock_owners_.end()) {
      return;
    }
    auto metric_it = std::find_if(
        last_snapshot_.begin(),
        last_snapshot_.end(),
        [gpu_id](const GpuMetrics& metric) { return metric.gpu_id == gpu_id; });
    if (metric_it == last_snapshot_.end()) {
      return;
    }
    selected.push_back(gpu_id);
  };

  for (int gpu_id : preferred_gpu_ids) {
    try_add(gpu_id);
  }
  for (const auto& metric : last_snapshot_) {
    try_add(metric.gpu_id);
  }

  if (selected.size() != static_cast<std::size_t>(count)) {
    return absl::ResourceExhaustedError("requested GPUs are not available");
  }

  for (int gpu_id : selected) {
    lock_owners_[gpu_id] = job_id;
  }
  return selected;
}

absl::Status GpuManager::UnlockGpus(const std::string& job_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = lock_owners_.begin(); it != lock_owners_.end();) {
    if (it->second == job_id) {
      it = lock_owners_.erase(it);
    } else {
      ++it;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<GpuMetrics>> GpuManager::FetchInventory() {
  if (use_nvml_) {
    auto snapshot = FetchFromNvml();
    if (snapshot.ok()) {
      return snapshot;
    }

    std::cerr << "NVML unavailable, falling back to nvidia-smi: " << snapshot.status() << '\n';
    use_nvml_ = false;
  }

  return FetchFromNvidiaSmi();
}

absl::StatusOr<std::vector<GpuMetrics>> GpuManager::FetchFromNvml() const {
  const nvmlReturn_t init_result = nvmlInit_v2();
  if (init_result != NVML_SUCCESS) {
    return absl::UnavailableError(NvmlErrorToString(init_result));
  }

  unsigned int device_count = 0;
  nvmlReturn_t count_result = nvmlDeviceGetCount_v2(&device_count);
  if (count_result != NVML_SUCCESS) {
    nvmlShutdown();
    return absl::UnavailableError(NvmlErrorToString(count_result));
  }

  std::vector<GpuMetrics> metrics;
  metrics.reserve(device_count);
  for (unsigned int index = 0; index < device_count; ++index) {
    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex_v2(index, &device) != NVML_SUCCESS) {
      continue;
    }

    std::array<char, NVML_DEVICE_NAME_BUFFER_SIZE> name{};
    nvmlMemory_t memory{};
    nvmlUtilization_t utilization{};

    if (nvmlDeviceGetName(device, name.data(), name.size()) != NVML_SUCCESS ||
        nvmlDeviceGetMemoryInfo(device, &memory) != NVML_SUCCESS ||
        nvmlDeviceGetUtilizationRates(device, &utilization) != NVML_SUCCESS) {
      continue;
    }

    GpuMetrics metric;
    metric.gpu_id = static_cast<int>(index);
    metric.model_name = name.data();
    metric.total_memory_bytes = memory.total;
    metric.used_memory_bytes = memory.used;
    metric.utilization_percent = utilization.gpu;
    metric.available = true;
    metrics.push_back(std::move(metric));
  }

  nvmlShutdown();
  return metrics;
}

absl::StatusOr<std::vector<GpuMetrics>> GpuManager::FetchFromNvidiaSmi() const {
  auto output = RunCommandCapture(
      nvidia_smi_path_ +
      " --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits");
  if (!output.ok()) {
    return output.status();
  }

  std::vector<GpuMetrics> metrics;
  for (absl::string_view raw_line : absl::StrSplit(*output, '\n', absl::SkipEmpty())) {
    std::string line(raw_line);
    absl::StripAsciiWhitespace(&line);
    if (line.empty()) {
      continue;
    }

    const std::vector<std::string> parts = absl::StrSplit(line, ',');
    if (parts.size() < 5) {
      continue;
    }

    GpuMetrics metric;
    metric.gpu_id = std::stoi(std::string(absl::StripAsciiWhitespace(parts[0])));
    metric.model_name = std::string(absl::StripAsciiWhitespace(parts[1]));
    metric.total_memory_bytes = static_cast<std::uint64_t>(
        std::stoull(std::string(absl::StripAsciiWhitespace(parts[2])))) * kMiB;
    metric.used_memory_bytes = static_cast<std::uint64_t>(
        std::stoull(std::string(absl::StripAsciiWhitespace(parts[3])))) * kMiB;
    metric.utilization_percent = static_cast<std::uint32_t>(
        std::stoul(std::string(absl::StripAsciiWhitespace(parts[4]))));
    metric.available = true;
    metrics.push_back(std::move(metric));
  }

  if (metrics.empty()) {
    return absl::UnavailableError("no GPUs discovered via nvidia-smi");
  }
  return metrics;
}

absl::StatusOr<std::string> GpuManager::RunCommandCapture(const std::string& command) const {
  std::array<char, 256> buffer{};
  std::string output;

  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
  if (!pipe) {
    return absl::UnavailableError("failed to execute command");
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    output.append(buffer.data());
  }

  const int exit_code = pclose(pipe.release());
  if (exit_code != 0) {
    return absl::UnavailableError("command failed: " + command);
  }

  return output;
}

}  // namespace gpu::manager


