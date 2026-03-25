#include <cassert>
#include <string>

#include "scheduler/scheduler.h"

int main() {
  gpu::scheduler::Scheduler scheduler;

  gpu::scheduler::JobRequest low;
  low.bundle_id = "bundle-low";
  low.bundle_path = "/tmp/bundle-low";
  low.entrypoint = "./run.sh";
  low.docker_image = "test-image";
  low.priority = gpu::scheduler::JobPriority::kLow;

  gpu::scheduler::JobRequest high = low;
  high.bundle_id = "bundle-high";
  high.bundle_path = "/tmp/bundle-high";
  high.priority = gpu::scheduler::JobPriority::kHigh;

  auto low_id = scheduler.Submit(low);
  auto high_id = scheduler.Submit(high);
  assert(low_id.ok());
  assert(high_id.ok());

  auto high_status = scheduler.GetStatus(*high_id);
  auto low_status = scheduler.GetStatus(*low_id);
  assert(high_status.ok());
  assert(low_status.ok());
  assert(high_status->queue_position == 1);
  assert(low_status->queue_position == 2);

  auto cancellation = scheduler.Cancel(*low_id);
  assert(cancellation.ok());
  assert(cancellation->view.state == gpu::scheduler::JobState::kCanceled);
  return 0;
}
