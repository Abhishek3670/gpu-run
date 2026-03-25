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

## Notes

- The control plane is fully native C++; no Python is used in the build,
  scheduler, server, CLI, or worker runtime.
- Submitted workloads may use any runtime inside the user container image.
- The older Python files in this repository are legacy artifacts and are not
  part of the C++ build graph.
