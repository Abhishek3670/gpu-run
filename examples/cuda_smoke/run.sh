#!/usr/bin/env bash

set -euo pipefail

matrix_size="${1:-4096}"
iterations="${2:-40}"
warmup="${3:-5}"

echo "CUDA_SMOKE_START"
echo "Requested matrix_size=${matrix_size} iterations=${iterations} warmup=${warmup}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not available in the container"
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is not available in the container"
  echo "Use a CUDA devel image, not plain ubuntu:22.04"
  exit 1
fi

echo "Initial GPU state"
nvidia-smi

echo "Compiling CUDA benchmark"
nvcc -O3 -std=c++17 -lineinfo \
  -gencode arch=compute_52,code=compute_52 \
  -o /tmp/matmul_bench ./matmul_bench.cu

echo "Starting GPU telemetry"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv -l 1 &
monitor_pid=$!

cleanup() {
  kill "${monitor_pid}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Running CUDA benchmark"
/tmp/matmul_bench "${matrix_size}" "${iterations}" "${warmup}"

sleep 1
echo "Final GPU state"
nvidia-smi
echo "CUDA_SMOKE_DONE"
