# GPU Dispatch Server

Native C++ GPU execution server for LAN-connected Linux clients. The system is a
lightweight Slurm-style control plane that stages job bundles, schedules GPU
workloads, and launches one Docker container per job on a Windows 11 host
running WSL2 Ubuntu.

## Layout

- `proto/`: protobuf and gRPC contract
- `server/`: async gRPC server and optional bearer-token auth
- `scheduler/`: in-memory priority queue and job state machine
- `gpu_manager/`: NVML-backed GPU discovery and lock tracking
- `worker/`: Docker launcher and bounded log buffer
- `cli/`: `gpu-run` client
- `scripts/`: Windows helper scripts for exposing WSL over LAN
- `tests/`: lightweight unit tests wired through CTest

## Build

Build inside WSL2 Ubuntu with the required C++ toolchain:

```bash
sudo apt install build-essential cmake protobuf-compiler-grpc
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Required dependencies:

- gRPC C++
- Protobuf
- Abseil
- CUDA Toolkit with NVML headers and `libnvidia-ml`
- Docker CLI available in WSL2

## Binaries

- `build/gpu-server`
- `build/gpu-run`

## Getting Started With `gpu-run`

The quickest path is to run both the server and the CLI inside WSL first, then
switch the CLI to a LAN address once the local flow works.

### 1. Make sure a usable image exists locally

`gpu-server` only accepts images that:

- already exist on the host
- are explicitly allowlisted with `--allow-image`
- can run the entrypoint you pass from `gpu-run`

For a first smoke test, a simple shell-capable image is enough:

```bash
docker image inspect ubuntu:22.04 >/dev/null 2>&1 || docker pull ubuntu:22.04
```

### 2. Build the binaries

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 3. Start the server

```bash
mkdir -p bundles
./build/gpu-server \
  --bind 127.0.0.1:50051 \
  --bundle-root ./bundles \
  --allow-image ubuntu:22.04
```

For a remote Linux client on your LAN, bind to the server's reachable address
instead of `127.0.0.1`, then point `gpu-run --server` at that host and port.

If you enable auth on the server, pass the same token to the CLI with
`--token <value>`.

### 4. Check GPU visibility

```bash
./build/gpu-run --server 127.0.0.1:50051 list-gpus
```

This should return JSON with each GPU's id, memory usage, utilization, and
lock state.

### 5. Submit your first job

The repository includes a tiny workload at `tests/e2e_workload/run.sh` that is
safe to use for the first submission.

```bash
./build/gpu-run \
  --server 127.0.0.1:50051 \
  run \
  --script ./tests/e2e_workload/run.sh \
  --image ubuntu:22.04 \
  --entrypoint /bin/sh \
  -- ./run.sh
```

Expected output is JSON with a `job_id`, for example:

```json
{"job_id":"job-1","state":"QUEUED"}
```

Notes:

- `--script` uploads the local file or directory to the server. It is not a
  server-side filesystem path.
- Use `--entrypoint` plus `--` when the container should invoke a shell or
  another launcher before your uploaded script.
- `--gpus`, `--prefer-gpus`, `--priority`, and `--task` are optional for the
  first run.

### 6. Follow the job

Poll status:

```bash
./build/gpu-run --server 127.0.0.1:50051 status <job_id>
```

Stream logs:

```bash
./build/gpu-run --server 127.0.0.1:50051 logs <job_id>
```

For the bundled smoke workload, the logs should contain:

- `E2E_START`
- `E2E_DONE`

### 7. Cancel a running job

```bash
./build/gpu-run --server 127.0.0.1:50051 cancel <job_id>
```

The server will issue `docker kill job-<id>` for active jobs and release the
GPU lock when the worker exits.

### 8. Submit your own workload

Common patterns:

- Single executable file:
  use `--script /path/to/file` and either make the file executable or set
  `--entrypoint` explicitly.
- Directory upload:
  use `--script /path/to/project_dir` and pass an explicit `--entrypoint`
  because the server mounts the whole directory at `/workspace`.
- Remote client:
  keep the same command shape, but change `--server` to the host's LAN address.

## End-To-End Smoke Test

An integration test is wired through CTest and runs when a suitable Docker image
is already cached locally.

Requirements:

- Set `GPU_DISPATCH_E2E_IMAGE` to a local image name.
- The image must support `/bin/sh`.
- The image must work with `docker run --gpus`.

Run it with:

```bash
GPU_DISPATCH_E2E_IMAGE=ubuntu:22.04 ctest --test-dir build -R gpu_dispatch_e2e --output-on-failure
```

If `GPU_DISPATCH_E2E_IMAGE` is unset or the image is not present locally, the
test is skipped.

## Notes

- The control plane is fully native C++; no Python is used in the build,
  scheduler, server, CLI, or worker runtime.
- Submitted workloads may use any runtime inside the user container image.
- The older Python files in this repository are legacy artifacts and are not
  part of the C++ build graph.
