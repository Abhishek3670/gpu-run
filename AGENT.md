# AGENT.md — GPU Execution Server

This file defines agent behavior, code conventions, build instructions, and task guidance for AI coding assistants working on this project.

---

## Project Overview

A high-performance, client-server GPU dispatch system written entirely in C++. Linux clients submit GPU workloads (training jobs, compute tasks) to a Windows 11 host running WSL2 over LAN. The server manages a priority queue, allocates NVIDIA GPUs, and launches isolated Docker containers for each job.

**Think of it as a lightweight Slurm — no orchestration frameworks, no Python, no cloud.**

---

## Repository Layout

```
/
├── AGENT.md               ← you are here
├── CMakeLists.txt         ← root build file, wires all targets
├── proto/
│   └── gpu_service.proto  ← single source of truth for all gRPC API contracts
├── server/
│   ├── main.cc            ← gRPC server entry point
│   ├── server.h / .cc     ← async completion-queue server
│   └── auth.h / .cc       ← optional token-based auth middleware
├── scheduler/
│   ├── scheduler.h / .cc  ← priority queue, job state machine
│   └── job.h              ← Job struct definition
├── gpu_manager/
│   ├── gpu_manager.h / .cc ← NVML wrapper, GPU lock/unlock
│   └── metrics.h          ← GPUMetrics struct
├── worker/
│   ├── worker.h / .cc     ← Docker launcher, log capture
│   └── log_streamer.h     ← gRPC log streaming adapter
└── cli/
    ├── main.cc            ← gpu-run entry point
    └── client.h / .cc     ← gRPC stub wrapper, command dispatch
```

---

## Non-Negotiable Rules

These constraints are hard limits. Do not work around them under any circumstance.

| Rule | Detail |
|---|---|
| **No Python** | Zero Python in any execution path. Not in scripts, not in helpers, not in build steps. |
| **No Kubernetes** | No container orchestration. Docker CLI invocation only. |
| **No cloud calls** | All networking is LAN-only. No external API calls at runtime. |
| **C++ only** | All components (server, scheduler, GPU manager, worker, CLI) are C++17 or later. |
| **Latency budget** | Job dispatch from CLI submit to Docker container start must be under 5ms. |
| **One container per job** | Never share a container between jobs. Isolation is mandatory. |

---

## Build System

**Toolchain:** CMake 3.20+, gRPC C++, Protobuf, NVML

```bash
# First-time setup (WSL2 Ubuntu)
sudo apt install cmake build-essential
# Install gRPC C++ from source or vcpkg
# Install CUDA Toolkit (includes NVML headers)

# Generate proto stubs
cd proto/
protoc --grpc_out=. --cpp_out=. \
  --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) \
  gpu_service.proto

# Build all targets
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Binaries produced
# build/server/gpu-server
# build/cli/gpu-run
```

**When modifying `gpu_service.proto`:** always regenerate stubs before touching any `.cc` files. The proto file is the contract — never edit generated files directly.

---

## Component Responsibilities

### gRPC Server (`server/`)
- Runs as an async server using gRPC completion queues (never synchronous gRPC)
- Accepts `SubmitJob`, `GetStatus`, `StreamLogs`, `ListGPUs` RPCs
- Validates requests before passing to the scheduler
- Binds only to LAN interface — never `0.0.0.0` unless explicitly configured

### Scheduler (`scheduler/`)
- Single in-memory priority queue (low / medium / high)
- Thread-safe — all public methods must be safe for concurrent access
- Owns the canonical job state machine: `QUEUED → DISPATCHED → RUNNING → DONE | FAILED`
- Never directly touches Docker or GPUs — delegates to Worker and GPU Manager

### GPU Manager (`gpu_manager/`)
- Wraps NVML for device enumeration, memory, and utilization queries
- Maintains a mutex-protected map of GPU ID → locked/available state
- `LockGPU(id)` and `UnlockGPU(id)` are the only state-mutation methods
- Falls back to `nvidia-smi` subprocess parsing if NVML init fails

### Worker Engine (`worker/`)
- Launches Docker via `docker run --gpus device=<id> ...` subprocess
- Bind-mounts the user script path into the container at `/workspace`
- Captures `stdout` and `stderr` via pipe, not log files
- Feeds captured output into the gRPC `StreamLogs` server-streaming RPC
- On container exit, reports exit code back to the scheduler

### CLI (`cli/`)
- Thin gRPC stub wrapper — no business logic
- All commands print structured output to stdout (JSON-friendly where possible)
- `gpu-run logs <job_id>` must stream — do not buffer the full log before printing
- Exit codes: `0` success, `1` server error, `2` invalid arguments

---

## Code Conventions

### General
- C++17 minimum. Use `std::optional`, `std::variant`, structured bindings freely.
- No raw `new` / `delete`. Use `std::unique_ptr` and `std::shared_ptr`.
- No global mutable state outside of the scheduler's queue (which is explicitly mutex-guarded).
- Prefer `[[nodiscard]]` on functions returning error codes or status objects.

### Error Handling
- Use `absl::Status` / `absl::StatusOr<T>` for all fallible operations.
- Never silently swallow errors. Log and propagate.
- gRPC handlers must always return a `grpc::Status` — never let an exception escape.

### Threading
- The scheduler queue is the only shared mutable state across threads.
- GPU manager lock map is also shared — protect with `std::mutex`.
- Do not use `std::thread` directly — use the gRPC completion queue threading model.

### Logging
- Use `absl::log` (or a thin wrapper) for all server-side logging.
- Log levels: `INFO` for job lifecycle events, `WARNING` for recoverable errors, `ERROR` for failures requiring attention.
- Never log GPU memory addresses or user script contents.

### Naming
- Files: `snake_case.cc` / `snake_case.h`
- Classes: `PascalCase`
- Methods and functions: `PascalCase` (Google style)
- Member variables: `trailing_underscore_`
- Constants: `kConstantName`

---

## gRPC API Contract

Defined in `proto/gpu_service.proto`. Do not change method signatures without updating both server and CLI stubs.

| Method | Type | Key fields |
|---|---|---|
| `SubmitJob` | Unary | `script_path`, `task_type`, `gpu_id`, `docker_image`, `priority` → `job_id`, `status` |
| `GetStatus` | Unary | `job_id` → `status`, `gpu_id`, `progress` |
| `StreamLogs` | Server streaming | `job_id` → `log_chunk` (repeated) |
| `ListGPUs` | Unary | (empty) → `gpu_id`, `memory_usage`, `utilization`, `available` |

**Task types:** `TRAINING` (long-running, holds GPU until complete) or `COMPUTE` (short-lived, GPU released immediately on exit).

**Priority values:** `LOW = 0`, `MEDIUM = 1`, `HIGH = 2`

---

## Docker Integration

```bash
# Template for worker container launch
docker run --rm \
  --gpus device=<gpu_id> \
  --name job-<job_id> \
  -v <script_dir>:/workspace \
  -w /workspace \
  <docker_image> \
  python3 <script_name>   # or any interpreter — the worker does not care
```

- Container name must be `job-<job_id>` for tracking and forced kill on timeout.
- Worker must call `docker kill job-<job_id>` if the job is cancelled while running.
- Pre-warm containers by pulling images at server startup — do not pull on first job submission.

---

## Performance Guidelines

- **Hot path:** `SubmitJob` RPC → scheduler enqueue → GPU lock → Docker launch. Profile this end-to-end. Target: under 5ms from RPC receipt to `docker run` call issued.
- **Avoid protobuf allocation in hot path.** Reuse message objects where possible.
- **Log streaming:** write to the gRPC stream as soon as bytes arrive from the pipe. Do not buffer more than one line.
- **CPU thread pinning:** pin the scheduler dispatch thread to a dedicated core if the host has 4+ cores.
- **Docker cold-start:** the 5ms budget does not include Docker image pull time. Images must already be present on the host.

---

## Validation Checklist

Before marking any component complete, verify:

- [ ] Single training job submits, runs, and streams logs end-to-end
- [ ] Two concurrent jobs from separate CLI processes allocate separate GPUs
- [ ] `gpu-run list-gpus` reflects locked GPUs correctly during a running job
- [ ] Cancelling a job via Ctrl-C on the CLI triggers `docker kill` on the server
- [ ] Server survives a crashed container without hanging or corrupting queue state
- [ ] Dispatch latency benchmark passes: median under 5ms over 100 submissions
- [ ] Auth token rejection returns `grpc::StatusCode::UNAUTHENTICATED`, not a crash

---

## Common Pitfalls

**Do not use synchronous gRPC.** The server must use the async completion queue API. Synchronous gRPC blocks the thread per RPC and will not scale to concurrent jobs.

**Do not pass `gpu_id = -1` to Docker.** Always validate that the GPU manager returned a valid locked ID before constructing the `docker run` command.

**Do not assume Docker is in PATH.** Resolve the Docker binary path at server startup and cache it. Fail fast if Docker is not found.

**Do not use `std::system()` for Docker.** Use `posix_spawn` or `fork/exec` with pipe capture for `stdout`/`stderr`. `std::system()` does not give you a pipe handle.

**Proto enums start at 0.** `Priority::LOW = 0` means an unset `priority` field silently becomes LOW. Consider starting meaningful values at 1 and treating 0 as `UNSPECIFIED` with a validation error.

---

## Agent Task Guidance

When implementing a feature or fixing a bug, follow this order:

1. **Read the proto first.** All API shapes are defined there. If the proto needs changing, update it and regenerate stubs before writing any other code.
2. **Implement the scheduler change** (if any) before the server handler — the handler delegates to the scheduler.
3. **Write the header before the implementation.** The `.h` file is the interface contract; get it right before filling in the body.
4. **Test with the CLI last.** The CLI is a thin client — if the server works, the CLI should be straightforward.
5. **Never leave a mutex locked across an I/O call.** Release the lock, do the I/O, reacquire if needed.