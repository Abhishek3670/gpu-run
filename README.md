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

For a remote Linux client that only needs `gpu-run`, build the client binary
without the server-side NVML and Docker pieces:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGPU_DISPATCH_BUILD_SERVER=OFF \
  -DBUILD_TESTING=OFF
cmake --build build --target gpu-run --parallel
```

If the remote client is missing the C++ dependencies needed for `gpu-run`,
install them first:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  pkg-config \
  libabsl-dev \
  libprotobuf-dev \
  protobuf-compiler \
  libgrpc++-dev \
  protobuf-compiler-grpc
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
For remote access, use `0.0.0.0:50051` so the service is reachable beyond the
WSL loopback interface:

```bash
pkill gpu-server || true
mkdir -p bundles
./build/gpu-server \
  --bind 0.0.0.0:50051 \
  --bundle-root ./bundles \
  --allow-image ubuntu:22.04 \
  --token your-token
```

Validate the listen socket inside WSL:

```bash
ss -ltnp | grep 50051
```

Expected shape:

```text
LISTEN ... *:50051 ... gpu-server
```

If you enable auth on the server, pass the same token to the CLI with
`--token <value>`.

### 3a. Expose WSL to the LAN from Windows

Resolve the current WSL IP from Ubuntu:

```bash
hostname -I
```

Then run the helper from an elevated PowerShell on Windows and pass the actual
IP value, not the literal string `WslIp` and not angle-bracket placeholders:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\windows_expose_wsl.ps1 `
  -ListenAddress 0.0.0.0 `
  -ListenPort 50051 `
  -WslIp <wsl-ip> `
  -WslPort 50051 `
  -AllowedSubnet <lan-subnet-cidr>
```

Validate the port forward on Windows:

```powershell
netsh interface portproxy show v4tov4
```

Expected shape:

```text
Listen on ipv4:             Connect to ipv4:

Address         Port        Address         Port
--------------- ----------  --------------- ----------
0.0.0.0         50051       <wsl-ip>     50051
```

Then confirm that the Windows host is listening on its LAN IP:

```powershell
Test-NetConnection -ComputerName <windows-lan-ip> -Port 50051
```

Use that Windows LAN IP from the remote Linux client:

```bash
./build/gpu-run --server <windows-lan-ip>:50051 --token your-token list-gpus
```

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
  If you are compiling on a client-only machine, use
  `-DGPU_DISPATCH_BUILD_SERVER=OFF -DBUILD_TESTING=OFF` during CMake
  configuration.

## Remote Client Troubleshooting

- `The source directory ... does not appear to contain CMakeLists.txt`:
  you ran `cmake` outside the cloned repository. `cd` into the repo root first
  and confirm `ls CMakeLists.txt` succeeds before configuring.
- `Unable to (re)create the private pkgRedirects directory`:
  the repo or `build/` directory is not writable by your current user. Fix the
  ownership, remove the stale build directory, then re-run CMake:

  ```bash
  sudo chown -R $USER:$USER /path/to/GPU_server
  rm -rf build
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DGPU_DISPATCH_BUILD_SERVER=OFF \
    -DBUILD_TESTING=OFF
  ```

- `Could not find a package configuration file provided by "absl"`:
  the client-side development packages are missing. Install the packages listed
  above, then re-run the configure step.
- `Could not find NVML_LIBRARY` on a remote Linux client:
  you are configuring the full server build on a machine that only needs the
  CLI. Reconfigure with `-DGPU_DISPATCH_BUILD_SERVER=OFF -DBUILD_TESTING=OFF`.
- `docker pull ubuntu:22.04` on the wrong machine:
  pull runnable images on the WSL server, not on the remote client. The remote
  client only uploads scripts and calls gRPC.
- `Connection refused` from the remote Linux client:
  this is usually a TCP routing or bind problem, not an auth problem. Check all
  of the following:

  ```bash
  # inside WSL
  ss -ltnp | grep 50051
  hostname -I
  ```

  ```powershell
  # inside elevated PowerShell on Windows
  netsh interface portproxy show v4tov4
  Test-NetConnection -ComputerName <windows-lan-ip> -Port 50051
  ```

  Fixes:

  - restart `gpu-server` with `--bind 0.0.0.0:50051`, not `127.0.0.1:50051`
  - ensure only one `gpu-server` is running before you restart it
  - recreate the portproxy with the current WSL IP from `hostname -I`
  - verify `portproxy show v4tov4` contains a numeric WSL IP, not a literal
    placeholder such as `WslIp`

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
