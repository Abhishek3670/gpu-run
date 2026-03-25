#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "server/server.h"

namespace {

void PrintUsage() {
  std::cerr << "Usage: gpu-server [--bind host:port] [--bundle-root path]"
            << " [--token value] [--allow-image image] [--docker path] [--nvidia-smi path]\n";
}

}  // namespace

int main(int argc, char** argv) {
  gpu::server::ServerConfig config;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    const auto require_value = [&](const char* /*flag*/) -> std::string {
      if (index + 1 >= argc) {
        PrintUsage();
        std::exit(2);
      }
      ++index;
      return argv[index];
    };

    if (arg == "--bind") {
      config.bind_address = require_value("--bind");
    } else if (arg == "--bundle-root") {
      config.bundle_root = require_value("--bundle-root");
    } else if (arg == "--token") {
      config.auth_token = require_value("--token");
    } else if (arg == "--allow-image") {
      config.allowed_images.push_back(require_value("--allow-image"));
    } else if (arg == "--docker") {
      config.docker_path = require_value("--docker");
    } else if (arg == "--nvidia-smi") {
      config.nvidia_smi_path = require_value("--nvidia-smi");
    } else {
      PrintUsage();
      return 2;
    }
  }

  gpu::server::GpuServer server(config);
  const absl::Status status = server.Run();
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 1;
  }
  return 0;
}
