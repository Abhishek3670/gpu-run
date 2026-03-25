# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-03-25

### Added
- Native C++17/CMake project layout for the GPU dispatch server and `gpu-run` CLI.
- Protobuf and gRPC contract for bundle upload, job submission, status, logs, GPU listing, and cancellation.
- In-memory scheduler, NVML-backed GPU manager, Docker worker engine, async gRPC server, and CLI client.
- Root-level WSL/Docker helper script and lightweight CTest coverage for scheduler and log streaming behavior.

### Changed
- Replaced the repo's primary Python-service documentation with the native C++ GPU dispatch system build and usage flow.
- Wired CMake to support Ubuntu 22.04-style gRPC packaging and WSL NVML library discovery.
