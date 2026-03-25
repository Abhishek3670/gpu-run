#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <sys/types.h>

namespace gpu::scheduler {

enum class TaskType {
  kTraining,
  kCompute,
};

enum class JobPriority {
  kLow = 0,
  kMedium = 1,
  kHigh = 2,
};

enum class JobState {
  kQueued,
  kDispatching,
  kRunning,
  kSucceeded,
  kFailed,
  kCanceled,
};

struct JobRequest {
  std::string bundle_id;
  std::string bundle_path;
  std::string entrypoint;
  std::vector<std::string> args;
  TaskType task_type = TaskType::kCompute;
  JobPriority priority = JobPriority::kMedium;
  std::string docker_image;
  int gpu_count = 1;
  std::vector<int> preferred_gpu_ids;
};

struct JobRuntime {
  JobState state = JobState::kQueued;
  std::vector<int> assigned_gpu_ids;
  std::optional<int> exit_code;
  std::string status_message;
  bool cancel_requested = false;
  pid_t worker_pid = -1;
};

struct Job {
  std::string job_id;
  JobRequest request;
  JobRuntime runtime;
  std::chrono::system_clock::time_point submitted_at;
};

struct JobStatusView {
  std::string job_id;
  JobState state = JobState::kQueued;
  std::vector<int> assigned_gpu_ids;
  std::size_t queue_position = 0;
  std::optional<int> exit_code;
  std::string status_message;
  bool cancel_requested = false;
};

struct CancelAction {
  JobStatusView view;
  bool requires_worker_cancel = false;
};

inline std::string_view ToString(JobState state) {
  switch (state) {
    case JobState::kQueued:
      return "QUEUED";
    case JobState::kDispatching:
      return "DISPATCHING";
    case JobState::kRunning:
      return "RUNNING";
    case JobState::kSucceeded:
      return "SUCCEEDED";
    case JobState::kFailed:
      return "FAILED";
    case JobState::kCanceled:
      return "CANCELED";
  }

  return "UNKNOWN";
}

inline std::string_view ToString(JobPriority priority) {
  switch (priority) {
    case JobPriority::kLow:
      return "LOW";
    case JobPriority::kMedium:
      return "MEDIUM";
    case JobPriority::kHigh:
      return "HIGH";
  }

  return "UNKNOWN";
}

inline std::string_view ToString(TaskType type) {
  switch (type) {
    case TaskType::kTraining:
      return "TRAINING";
    case TaskType::kCompute:
      return "COMPUTE";
  }

  return "UNKNOWN";
}

}  // namespace gpu::scheduler
