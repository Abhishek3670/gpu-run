#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "absl/status/status.h"
#include "server/server.h"

namespace {

constexpr std::string_view kDefaultAllowlistPath = "config/images.allowlist";

std::string Trim(std::string_view input) {
  std::size_t begin = 0;
  while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin])) != 0) {
    ++begin;
  }

  std::size_t end = input.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
    --end;
  }

  return std::string(input.substr(begin, end - begin));
}

void AddUniqueImage(
    const std::string& image,
    std::unordered_set<std::string>* seen,
    std::vector<std::string>* images) {
  if (seen->insert(image).second) {
    images->push_back(image);
  }
}

absl::Status LoadAllowlistFile(const std::filesystem::path& path, std::vector<std::string>* images) {
  std::ifstream input(path);
  if (!input.is_open()) {
    return absl::NotFoundError("failed to open allowlist file: " + path.string());
  }

  std::unordered_set<std::string> seen(images->begin(), images->end());
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }
    AddUniqueImage(trimmed, &seen, images);
  }

  return absl::OkStatus();
}

void MergeAllowlistImages(
    const std::vector<std::string>& extra_images,
    std::vector<std::string>* images) {
  std::unordered_set<std::string> seen(images->begin(), images->end());
  for (const auto& image : extra_images) {
    AddUniqueImage(image, &seen, images);
  }
}

void PrintUsage() {
  std::cerr << "Usage: gpu-server [--bind host:port] [--bundle-root path]"
            << " [--token value] [--allowlist path] [--allow-image image]"
            << " [--docker path] [--nvidia-smi path]\n";
}

}  // namespace

int main(int argc, char** argv) {
  gpu::server::ServerConfig config;
  std::filesystem::path allowlist_path = kDefaultAllowlistPath;
  std::vector<std::string> cli_allow_images;
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
    } else if (arg == "--allowlist") {
      allowlist_path = require_value("--allowlist");
    } else if (arg == "--allow-image") {
      cli_allow_images.push_back(require_value("--allow-image"));
    } else if (arg == "--docker") {
      config.docker_path = require_value("--docker");
    } else if (arg == "--nvidia-smi") {
      config.nvidia_smi_path = require_value("--nvidia-smi");
    } else {
      PrintUsage();
      return 2;
    }
  }

  const absl::Status allowlist_status = LoadAllowlistFile(allowlist_path, &config.allowed_images);
  if (!allowlist_status.ok()) {
    std::cerr << allowlist_status.message() << '\n';
    return 1;
  }
  MergeAllowlistImages(cli_allow_images, &config.allowed_images);

  gpu::server::GpuServer server(config);
  const absl::Status status = server.Run();
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 1;
  }
  return 0;
}

