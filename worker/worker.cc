#include "worker/worker.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"

namespace gpu::worker {
namespace {

constexpr const char* kWorkspaceMount = "/workspace";

int ExitCodeFromStatus(int status) {
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }
  return -1;
}

}  // namespace

Worker::Worker(WorkerConfig config, LogStreamer* log_streamer)
    : config_(std::move(config)), log_streamer_(log_streamer) {}

absl::Status Worker::Initialize() {
  auto docker_binary = ResolveDockerBinary();
  if (!docker_binary.ok()) {
    return docker_binary.status();
  }
  docker_binary_ = *docker_binary;
  return ValidateConfiguredImages();
}

absl::StatusOr<LaunchResult> Worker::StartJob(
    const scheduler::Job& job,
    const std::vector<int>& gpu_ids) {
  if (docker_binary_.empty()) {
    return absl::FailedPreconditionError("worker not initialized");
  }
  if (!std::filesystem::exists(job.request.bundle_path)) {
    return absl::NotFoundError("bundle path does not exist");
  }

  auto docker_args = BuildDockerArgs(job, gpu_ids);
  if (!docker_args.ok()) {
    return docker_args.status();
  }

  int stdout_pipe[2] = {-1, -1};
  int stderr_pipe[2] = {-1, -1};
  if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
    return absl::InternalError("failed to create worker pipes");
  }

  const pid_t pid = fork();
  if (pid < 0) {
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[0]);
    close(stderr_pipe[1]);
    return absl::InternalError("fork failed");
  }

  if (pid == 0) {
    close(stdout_pipe[0]);
    close(stderr_pipe[0]);
    dup2(stdout_pipe[1], STDOUT_FILENO);
    dup2(stderr_pipe[1], STDERR_FILENO);
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    std::vector<char*> argv;
    argv.reserve(docker_args->size() + 2);
    argv.push_back(const_cast<char*>(docker_binary_.c_str()));
    for (std::string& arg : *docker_args) {
      argv.push_back(arg.data());
    }
    argv.push_back(nullptr);

    execv(docker_binary_.c_str(), argv.data());
    _exit(127);
  }

  close(stdout_pipe[1]);
  close(stderr_pipe[1]);
  SetNonBlocking(stdout_pipe[0]);
  SetNonBlocking(stderr_pipe[0]);

  if (log_streamer_ != nullptr) {
    log_streamer_->EnsureJob(job.job_id);
  }

  RunningJob running_job;
  running_job.job_id = job.job_id;
  running_job.pid = pid;
  running_job.gpu_ids = gpu_ids;
  running_job.stdout_fd = stdout_pipe[0];
  running_job.stderr_fd = stderr_pipe[0];

  std::lock_guard<std::mutex> lock(mutex_);
  running_jobs_[job.job_id] = std::move(running_job);
  return LaunchResult{pid};
}

absl::Status Worker::CancelJob(const std::string& job_id) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = running_jobs_.find(job_id);
    if (it == running_jobs_.end()) {
      return absl::NotFoundError("unknown running job");
    }
    it->second.cancel_requested = true;
  }

  return RunDockerKill(job_id);
}

std::vector<WorkerEvent> Worker::PollEvents() {
  std::vector<RunningJob> jobs;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [job_id, running_job] : running_jobs_) {
      (void)job_id;
      jobs.push_back(running_job);
    }
  }

  std::vector<WorkerEvent> events;
  for (const RunningJob& job : jobs) {
    DrainFd(job.stdout_fd, job.job_id, LogSource::kStdout, log_streamer_);
    DrainFd(job.stderr_fd, job.job_id, LogSource::kStderr, log_streamer_);

    int status = 0;
    const pid_t wait_result = waitpid(job.pid, &status, WNOHANG);
    if (wait_result == 0) {
      continue;
    }
    if (wait_result < 0) {
      continue;
    }

    RunningJob finished_job;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = running_jobs_.find(job.job_id);
      if (it == running_jobs_.end()) {
        continue;
      }
      finished_job = it->second;
      running_jobs_.erase(it);
    }

    close(finished_job.stdout_fd);
    close(finished_job.stderr_fd);
    if (log_streamer_ != nullptr) {
      log_streamer_->MarkComplete(finished_job.job_id);
    }

    WorkerEvent event;
    event.job_id = finished_job.job_id;
    event.gpu_ids = finished_job.gpu_ids;
    event.exit_code = ExitCodeFromStatus(status);
    event.canceled = finished_job.cancel_requested;
    if (event.canceled) {
      event.status_message = "Canceled";
    } else if (event.exit_code == 0) {
      event.status_message = "Completed successfully";
    } else {
      event.status_message = "Worker exited with code " + std::to_string(event.exit_code);
    }
    events.push_back(std::move(event));
  }

  return events;
}

absl::StatusOr<std::string> Worker::ResolveDockerBinary() const {
  if (config_.docker_path.find('/') != std::string::npos) {
    if (access(config_.docker_path.c_str(), X_OK) == 0) {
      return config_.docker_path;
    }
    return absl::NotFoundError("docker binary is not executable");
  }

  const char* raw_path = std::getenv("PATH");
  if (raw_path == nullptr) {
    return absl::NotFoundError("PATH is not set");
  }

  for (const auto path_entry : absl::StrSplit(raw_path, ':')) {
    const std::filesystem::path candidate = std::filesystem::path(std::string(path_entry)) / config_.docker_path;
    if (std::filesystem::exists(candidate) && access(candidate.c_str(), X_OK) == 0) {
      return candidate.string();
    }
  }

  return absl::NotFoundError("docker binary not found in PATH");
}

absl::Status Worker::ValidateConfiguredImages() const {
  for (const auto& image : config_.allowed_images) {
    const pid_t pid = fork();
    if (pid < 0) {
      return absl::InternalError("failed to validate docker images");
    }

    if (pid == 0) {
      execl(
          docker_binary_.c_str(),
          docker_binary_.c_str(),
          "image",
          "inspect",
          image.c_str(),
          static_cast<char*>(nullptr));
      _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0 || ExitCodeFromStatus(status) != 0) {
      return absl::FailedPreconditionError("docker image is not available locally: " + image);
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::string>> Worker::BuildDockerArgs(
    const scheduler::Job& job,
    const std::vector<int>& gpu_ids) const {
  if (!config_.allowed_images.empty()) {
    const bool image_allowed =
        std::find(config_.allowed_images.begin(), config_.allowed_images.end(), job.request.docker_image) !=
        config_.allowed_images.end();
    if (!image_allowed) {
      return absl::PermissionDeniedError("docker image is not in the allowlist");
    }
  }

  std::vector<std::string> args;
  args.push_back("run");
  args.push_back("--rm");
  args.push_back("--name");
  args.push_back(ContainerName(job.job_id));
  args.push_back("--gpus");
  args.push_back("device=" + JoinGpuIds(gpu_ids));
  args.push_back("-v");
  args.push_back(job.request.bundle_path + ":" + kWorkspaceMount);
  args.push_back("-w");
  args.push_back(kWorkspaceMount);
  args.push_back(job.request.docker_image);
  args.push_back(job.request.entrypoint);
  args.insert(args.end(), job.request.args.begin(), job.request.args.end());
  return args;
}

absl::Status Worker::RunDockerKill(const std::string& job_id) const {
  const pid_t pid = fork();
  if (pid < 0) {
    return absl::InternalError("failed to fork docker kill");
  }

  if (pid == 0) {
    execl(
        docker_binary_.c_str(),
        docker_binary_.c_str(),
        "kill",
        ContainerName(job_id).c_str(),
        static_cast<char*>(nullptr));
    _exit(127);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    return absl::InternalError("failed waiting for docker kill");
  }
  if (ExitCodeFromStatus(status) != 0) {
    return absl::InternalError("docker kill failed");
  }
  return absl::OkStatus();
}

std::string Worker::ContainerName(const std::string& job_id) {
  return job_id;
}

std::string Worker::JoinGpuIds(const std::vector<int>& gpu_ids) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < gpu_ids.size(); ++index) {
    if (index != 0) {
      stream << ',';
    }
    stream << gpu_ids[index];
  }
  return stream.str();
}

void Worker::DrainFd(int fd, const std::string& job_id, LogSource source, LogStreamer* log_streamer) {
  if (fd < 0 || log_streamer == nullptr) {
    return;
  }

  std::array<char, 4096> buffer{};
  while (true) {
    const ssize_t bytes_read = read(fd, buffer.data(), buffer.size());
    if (bytes_read > 0) {
      log_streamer->Append(job_id, source, std::string(buffer.data(), static_cast<std::size_t>(bytes_read)));
      continue;
    }
    if (bytes_read == 0) {
      return;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return;
    }
    return;
  }
}

void Worker::SetNonBlocking(int fd) {
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags >= 0) {
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  }
}

}  // namespace gpu::worker




