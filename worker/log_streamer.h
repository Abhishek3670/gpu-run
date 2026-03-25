#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace gpu::worker {

enum class LogSource {
  kStdout,
  kStderr,
};

struct LogEntry {
  std::int64_t sequence = 0;
  std::int64_t timestamp_unix_ms = 0;
  LogSource source = LogSource::kStdout;
  std::string payload;
};

struct LogReadResult {
  std::vector<LogEntry> entries;
  bool complete = false;
};

class LogStreamer {
 public:
  explicit LogStreamer(std::size_t max_entries_per_job = 1024)
      : max_entries_per_job_(max_entries_per_job) {}

  void EnsureJob(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    logs_.try_emplace(job_id);
  }

  void Append(const std::string& job_id, LogSource source, std::string payload) {
    std::lock_guard<std::mutex> lock(mutex_);
    JobLog& job_log = logs_[job_id];
    job_log.next_sequence += 1;
    job_log.entries.push_back(LogEntry{
        job_log.next_sequence,
        CurrentUnixMillis(),
        source,
        std::move(payload),
    });
    while (job_log.entries.size() > max_entries_per_job_) {
      job_log.entries.pop_front();
    }
  }

  void MarkComplete(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    logs_[job_id].complete = true;
  }

  [[nodiscard]] absl::StatusOr<LogReadResult> ReadSince(
      const std::string& job_id,
      std::int64_t last_sequence) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = logs_.find(job_id);
    if (it == logs_.end()) {
      return absl::NotFoundError("unknown job id");
    }

    LogReadResult result;
    result.complete = it->second.complete;
    for (const auto& entry : it->second.entries) {
      if (entry.sequence > last_sequence) {
        result.entries.push_back(entry);
      }
    }
    return result;
  }

 private:
  struct JobLog {
    std::deque<LogEntry> entries;
    std::int64_t next_sequence = 0;
    bool complete = false;
  };

  [[nodiscard]] static std::int64_t CurrentUnixMillis() {
    const auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  }

  std::size_t max_entries_per_job_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, JobLog> logs_;
};

}  // namespace gpu::worker
