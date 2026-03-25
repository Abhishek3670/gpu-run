#include "scheduler/scheduler.h"

#include <algorithm>
#include <sstream>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gpu_manager/gpu_manager.h"
#include "worker/worker.h"

namespace gpu::scheduler {
namespace {

std::string MakeJobId(std::uint64_t ordinal) {
  std::ostringstream stream;
  stream << "job-" << ordinal;
  return stream.str();
}

}  // namespace

absl::StatusOr<std::string> Scheduler::Submit(JobRequest request) {
  if (request.bundle_id.empty()) {
    return absl::InvalidArgumentError("bundle_id is required");
  }
  if (request.bundle_path.empty()) {
    return absl::InvalidArgumentError("bundle_path is required");
  }
  if (request.entrypoint.empty()) {
    return absl::InvalidArgumentError("entrypoint is required");
  }
  if (request.docker_image.empty()) {
    return absl::InvalidArgumentError("docker_image is required");
  }
  if (request.gpu_count < 1) {
    return absl::InvalidArgumentError("gpu_count must be at least one");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  Job job;
  job.job_id = MakeJobId(next_job_id_++);
  job.request = std::move(request);
  job.runtime.state = JobState::kQueued;
  job.runtime.status_message = "Queued";
  job.submitted_at = std::chrono::system_clock::now();

  const std::string job_id = job.job_id;
  queues_[QueueIndex(job.request.priority)].push_back(job_id);
  jobs_.emplace(job_id, std::move(job));
  return job_id;
}

absl::StatusOr<JobStatusView> Scheduler::GetStatus(const std::string& job_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = jobs_.find(job_id);
  if (it == jobs_.end()) {
    return absl::NotFoundError("unknown job id");
  }
  return MakeViewLocked(it->second);
}

absl::StatusOr<CancelAction> Scheduler::Cancel(const std::string& job_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = jobs_.find(job_id);
  if (it == jobs_.end()) {
    return absl::NotFoundError("unknown job id");
  }

  Job& job = it->second;
  switch (job.runtime.state) {
    case JobState::kQueued:
      job.runtime.state = JobState::kCanceled;
      job.runtime.status_message = "Canceled before dispatch";
      return CancelAction{MakeViewLocked(job), false};
    case JobState::kDispatching:
    case JobState::kRunning:
      job.runtime.cancel_requested = true;
      job.runtime.status_message = "Cancellation requested";
      return CancelAction{MakeViewLocked(job), true};
    case JobState::kSucceeded:
    case JobState::kFailed:
    case JobState::kCanceled:
      return CancelAction{MakeViewLocked(job), false};
  }

  return absl::InternalError("unexpected job state");
}

std::vector<JobStatusView> Scheduler::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<JobStatusView> views;
  views.reserve(jobs_.size());
  for (const auto& [job_id, job] : jobs_) {
    (void)job_id;
    views.push_back(MakeViewLocked(job));
  }
  return views;
}

void Scheduler::TryDispatch(manager::GpuManager& gpu_manager, worker::Worker& worker) {
  while (true) {
    std::optional<DispatchCandidate> candidate;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      candidate = PopNextCandidateLocked();
      if (!candidate.has_value()) {
        return;
      }
    }

    auto gpu_ids = gpu_manager.TryLockGpus(
        candidate->snapshot.job_id,
        candidate->snapshot.request.gpu_count,
        candidate->snapshot.request.preferred_gpu_ids);
    if (!gpu_ids.ok()) {
      if (gpu_ids.status().code() == absl::StatusCode::kResourceExhausted) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = jobs_.find(candidate->snapshot.job_id);
        if (it != jobs_.end() && it->second.runtime.state == JobState::kDispatching) {
          it->second.runtime.state = JobState::kQueued;
          it->second.runtime.status_message = "Waiting for available GPUs";
          RequeueLocked(it->first, candidate->priority);
        }
        return;
      }

      std::lock_guard<std::mutex> lock(mutex_);
      auto it = jobs_.find(candidate->snapshot.job_id);
      if (it != jobs_.end()) {
        FailJobLocked(it->second, std::string(gpu_ids.status().message()));
      }
      continue;
    }

    auto launch = worker.StartJob(candidate->snapshot, *gpu_ids);
    if (!launch.ok()) {
      const absl::Status unlock_status = gpu_manager.UnlockGpus(candidate->snapshot.job_id);
      (void)unlock_status;
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = jobs_.find(candidate->snapshot.job_id);
      if (it != jobs_.end()) {
        FailJobLocked(it->second, std::string(launch.status().message()));
      }
      continue;
    }

    const absl::Status started_status = OnWorkerStarted(candidate->snapshot.job_id, *gpu_ids, launch->pid);
    (void)started_status;
  }
}

absl::Status Scheduler::OnWorkerStarted(
    const std::string& job_id,
    const std::vector<int>& gpu_ids,
    pid_t worker_pid) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = jobs_.find(job_id);
  if (it == jobs_.end()) {
    return absl::NotFoundError("unknown job id");
  }

  Job& job = it->second;
  job.runtime.state = JobState::kRunning;
  job.runtime.assigned_gpu_ids = gpu_ids;
  job.runtime.worker_pid = worker_pid;
  job.runtime.status_message = job.runtime.cancel_requested ? "Cancellation requested" : "Running";
  return absl::OkStatus();
}

absl::Status Scheduler::OnWorkerFinished(
    const std::string& job_id,
    int exit_code,
    const std::string& status_message) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = jobs_.find(job_id);
  if (it == jobs_.end()) {
    return absl::NotFoundError("unknown job id");
  }

  Job& job = it->second;
  job.runtime.exit_code = exit_code;
  if (job.runtime.cancel_requested) {
    job.runtime.state = JobState::kCanceled;
  } else if (exit_code == 0) {
    job.runtime.state = JobState::kSucceeded;
  } else {
    job.runtime.state = JobState::kFailed;
  }
  job.runtime.status_message = status_message;
  job.runtime.worker_pid = -1;
  return absl::OkStatus();
}

std::size_t Scheduler::QueueIndex(JobPriority priority) {
  switch (priority) {
    case JobPriority::kLow:
      return 0;
    case JobPriority::kMedium:
      return 1;
    case JobPriority::kHigh:
      return 2;
  }

  return 1;
}

std::optional<Scheduler::DispatchCandidate> Scheduler::PopNextCandidateLocked() {
  for (std::size_t index = queues_.size(); index-- > 0;) {
    auto& queue = queues_[index];
    while (!queue.empty()) {
      const std::string job_id = queue.front();
      queue.pop_front();

      auto it = jobs_.find(job_id);
      if (it == jobs_.end()) {
        continue;
      }

      Job& job = it->second;
      if (job.runtime.state != JobState::kQueued) {
        continue;
      }
      if (job.runtime.cancel_requested) {
        job.runtime.state = JobState::kCanceled;
        job.runtime.status_message = "Canceled before dispatch";
        continue;
      }

      job.runtime.state = JobState::kDispatching;
      job.runtime.status_message = "Allocating GPUs";
      return DispatchCandidate{job, job.request.priority};
    }
  }

  return std::nullopt;
}

JobStatusView Scheduler::MakeViewLocked(const Job& job) const {
  JobStatusView view;
  view.job_id = job.job_id;
  view.state = job.runtime.state;
  view.assigned_gpu_ids = job.runtime.assigned_gpu_ids;
  view.exit_code = job.runtime.exit_code;
  view.status_message = job.runtime.status_message;
  view.cancel_requested = job.runtime.cancel_requested;

  std::size_t position = 0;
  for (std::size_t index = queues_.size(); index-- > 0;) {
    for (const auto& queued_job_id : queues_[index]) {
      auto it = jobs_.find(queued_job_id);
      if (it == jobs_.end() || it->second.runtime.state != JobState::kQueued) {
        continue;
      }
      ++position;
      if (queued_job_id == job.job_id) {
        view.queue_position = position;
        return view;
      }
    }
  }

  view.queue_position = 0;
  return view;
}

void Scheduler::RequeueLocked(const std::string& job_id, JobPriority priority) {
  queues_[QueueIndex(priority)].push_front(job_id);
}

void Scheduler::FailJobLocked(Job& job, const std::string& message) {
  job.runtime.state = job.runtime.cancel_requested ? JobState::kCanceled : JobState::kFailed;
  job.runtime.exit_code = -1;
  job.runtime.status_message = message;
  job.runtime.worker_pid = -1;
}

}  // namespace gpu::scheduler
