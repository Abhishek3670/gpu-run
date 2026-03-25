#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kTile = 16;

void CheckCuda(cudaError_t result, const char* expr) {
  if (result == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(result));
}

#define CHECK_CUDA(expr) CheckCuda((expr), #expr)

__global__ void MatMulKernel(const float* a, const float* b, float* c, int n) {
  __shared__ float tile_a[kTile][kTile];
  __shared__ float tile_b[kTile][kTile];

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  float value = 0.0f;
  const int tiles = (n + kTile - 1) / kTile;
  for (int tile = 0; tile < tiles; ++tile) {
    const int a_col = tile * kTile + threadIdx.x;
    const int b_row = tile * kTile + threadIdx.y;

    tile_a[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;

    __syncthreads();

    for (int k = 0; k < kTile; ++k) {
      value += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    c[row * n + col] = value;
  }
}

int ParseInt(const char* raw, const char* name) {
  const int value = std::stoi(raw);
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int n = argc > 1 ? ParseInt(argv[1], "matrix_size") : 4096;
    const int iterations = argc > 2 ? ParseInt(argv[2], "iterations") : 40;
    const int warmup = argc > 3 ? ParseInt(argv[3], "warmup") : 5;

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices found");
    }

    CHECK_CUDA(cudaSetDevice(0));

    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));

    const std::size_t element_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    const std::size_t bytes = element_count * sizeof(float);

    std::vector<float> host_a(element_count);
    std::vector<float> host_b(element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
      host_a[index] = static_cast<float>((index % 251) * 0.01f);
      host_b[index] = static_cast<float>((index % 127) * 0.02f);
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    CHECK_CUDA(cudaMalloc(&device_a, bytes));
    CHECK_CUDA(cudaMalloc(&device_b, bytes));
    CHECK_CUDA(cudaMalloc(&device_c, bytes));

    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(kTile, kTile);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    for (int index = 0; index < warmup; ++index) {
      MatMulKernel<<<grid, block>>>(device_a, device_b, device_c, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int index = 0; index < iterations; ++index) {
      MatMulKernel<<<grid, block>>>(device_a, device_b, device_c, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::vector<float> host_c(element_count);
    CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    const std::size_t stride = std::max<std::size_t>(1, element_count / 1024);
    for (std::size_t index = 0; index < element_count; index += stride) {
      checksum += static_cast<double>(host_c[index]);
    }

    const double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n) *
        static_cast<double>(iterations);
    const double gflops = flops / (static_cast<double>(elapsed_ms) * 1.0e6);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CUDA device: " << properties.name << '\n';
    std::cout << "Matrix size: " << n << " x " << n << '\n';
    std::cout << "Iterations: " << iterations << " (warmup " << warmup << ")\n";
    std::cout << "Elapsed: " << elapsed_ms << " ms total\n";
    std::cout << "Average kernel time: " << (elapsed_ms / static_cast<float>(iterations)) << " ms\n";
    std::cout << "Throughput: " << gflops << " GFLOP/s\n";
    std::cout << "Checksum sample: " << checksum << '\n';

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "cuda_smoke failed: " << error.what() << '\n';
    return 1;
  }
}


