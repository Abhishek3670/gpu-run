#pragma once

// The TUI API stays small on purpose: main.cc only needs one entry point,
// while client.cc can depend on JobConfig without pulling ncurses into headers.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gpu_service.grpc.pb.h"

namespace gpu::cli {
class GpuRunClient;
}  // namespace gpu::cli

namespace gpu_run::tui {

struct GpuInfo {
  int gpu_id = -1;
  std::string model_name;
  std::uint64_t total_memory_bytes = 0;
  std::uint64_t used_memory_bytes = 0;
  std::uint32_t utilization_percent = 0;
  bool available = false;
  std::string locked_job_id;
};

struct JobConfig {
  std::string server_addr = "localhost:50051";
  std::string token;
  std::filesystem::path script_path;
  std::string docker_image = "ubuntu:22.04";
  std::string entrypoint;
  gpu::TaskType task_type = gpu::TaskType::TRAINING;
  gpu::Priority priority = gpu::Priority::MEDIUM;
  int gpu_count = 1;
  std::vector<int> preferred_gpu_ids;
};

enum class PostSubmitAction {
  StreamLogs,
  PollStatus,
  Exit,
};

struct SessionResult {
  bool submitted = false;
  std::string job_id;
  PostSubmitAction post_action = PostSubmitAction::Exit;
  std::string task_type_label;
};

// Collect server address and optional bearer token for the session.
bool ScreenServerConfig(JobConfig& config);

// Collect preferred GPU selections from a cached ListGPUs snapshot.
bool ScreenGpuSelect(JobConfig& config, const std::vector<GpuInfo>& gpus);

// Collect local upload and job submission fields without making RPCs.
bool ScreenJobConfig(JobConfig& config);

// Stream live logs and poll job status inside the active ncurses session.
void ScreenLogViewer(const std::string& job_id, gpu::cli::GpuRunClient& client);

// Run the four-screen wizard and choose the post-submit action.
SessionResult RunWizard(gpu::cli::GpuRunClient& client);

// Parse interactive argv, own ncurses lifecycle, and dispatch the wizard.
int Run(int argc, char** argv);

}  // namespace gpu_run::tui