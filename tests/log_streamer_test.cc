#include <cassert>
#include <string>

#include "worker/log_streamer.h"

int main() {
  gpu::worker::LogStreamer log_streamer(4);
  log_streamer.EnsureJob("job-1");
  log_streamer.Append("job-1", gpu::worker::LogSource::kStdout, "hello");
  log_streamer.Append("job-1", gpu::worker::LogSource::kStderr, "world");

  auto first_read = log_streamer.ReadSince("job-1", 0);
  assert(first_read.ok());
  assert(first_read->entries.size() == 2);
  assert(!first_read->complete);

  log_streamer.MarkComplete("job-1");
  auto second_read = log_streamer.ReadSince("job-1", first_read->entries.back().sequence);
  assert(second_read.ok());
  assert(second_read->entries.empty());
  assert(second_read->complete);
  return 0;
}
