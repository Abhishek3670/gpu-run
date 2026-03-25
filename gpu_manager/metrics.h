#pragma once

#include <cstdint>
#include <string>

namespace gpu::manager {

struct GpuMetrics {
  int gpu_id = -1;
  std::string model_name;
  std::uint64_t total_memory_bytes = 0;
  std::uint64_t used_memory_bytes = 0;
  std::uint32_t utilization_percent = 0;
  bool available = false;
  std::string locked_job_id;
};

}  // namespace gpu::manager
