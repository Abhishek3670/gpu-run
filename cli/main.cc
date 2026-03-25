#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "cli/client.h"

namespace {

std::string EscapeJson(std::string_view input) {
  std::string output;
  output.reserve(input.size());
  for (const char ch : input) {
    switch (ch) {
      case '\\':
        output += "\\\\";
        break;
      case '"':
        output += "\\\"";
        break;
      case '\n':
        output += "\\n";
        break;
      default:
        output.push_back(ch);
        break;
    }
  }
  return output;
}

std::vector<int> ParseGpuIds(const std::string& input) {
  std::vector<int> ids;
  std::stringstream stream(input);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      ids.push_back(std::stoi(token));
    }
  }
  return ids;
}

void PrintUsage() {
  std::cerr << "Usage: gpu-run [--server host:port] [--token value] <command> ...\n"
            << "Commands:\n"
            << "  list-gpus\n"
            << "  run --script <path> --image <image> [--task training|compute] [--gpus N] [--prefer-gpus ids] [--priority low|medium|high] [--entrypoint cmd] [-- arg...]\n"
            << "  status <job_id>\n"
            << "  logs <job_id>\n"
            << "  cancel <job_id>\n";
}

std::string RequireValue(int argc, char** argv, int* index) {
  if (*index + 1 >= argc) {
    PrintUsage();
    std::exit(2);
  }
  *index += 1;
  return argv[*index];
}

gpu::TaskType ParseTaskType(const std::string& value) {
  if (value == "training") {
    return gpu::TaskType::TRAINING;
  }
  if (value == "compute") {
    return gpu::TaskType::COMPUTE;
  }
  return gpu::TaskType::TASK_TYPE_UNSPECIFIED;
}

gpu::Priority ParsePriority(const std::string& value) {
  if (value == "high") {
    return gpu::Priority::HIGH;
  }
  if (value == "low") {
    return gpu::Priority::LOW;
  }
  if (value == "medium") {
    return gpu::Priority::MEDIUM;
  }
  return gpu::Priority::PRIORITY_UNSPECIFIED;
}

}  // namespace

int main(int argc, char** argv) {
  gpu::cli::ClientOptions client_options;
  int index = 1;
  for (; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--server") {
      client_options.server_address = RequireValue(argc, argv, &index);
    } else if (arg == "--token") {
      client_options.bearer_token = RequireValue(argc, argv, &index);
    } else {
      break;
    }
  }

  if (index >= argc) {
    PrintUsage();
    return 2;
  }

  const std::string command = argv[index++];
  gpu::cli::GpuRunClient client(client_options);

  if (command == "list-gpus") {
    auto gpus = client.ListGpus();
    if (!gpus.ok()) {
      std::cerr << gpus.status().message() << '\n';
      return 1;
    }

    std::cout << "[\n";
    for (std::size_t gpu_index = 0; gpu_index < gpus->size(); ++gpu_index) {
      const auto& gpu = (*gpus)[gpu_index];
      std::cout << "  {\"gpu_id\":" << gpu.gpu_id
                << ",\"model_name\":\"" << EscapeJson(gpu.model_name)
                << "\",\"total_memory_bytes\":" << gpu.total_memory_bytes
                << ",\"used_memory_bytes\":" << gpu.used_memory_bytes
                << ",\"utilization_percent\":" << gpu.utilization_percent
                << ",\"available\":" << (gpu.available ? "true" : "false")
                << ",\"locked_job_id\":\"" << EscapeJson(gpu.locked_job_id) << "\"}";
      std::cout << (gpu_index + 1 == gpus->size() ? "\n" : ",\n");
    }
    std::cout << "]\n";
    return 0;
  }

  if (command == "status") {
    if (index >= argc) {
      PrintUsage();
      return 2;
    }

    auto status = client.GetStatus(argv[index]);
    if (!status.ok()) {
      std::cerr << status.status().message() << '\n';
      return 1;
    }

    std::cout << "{\"job_id\":\"" << EscapeJson(status->job_id)
              << "\",\"state\":\"" << EscapeJson(status->state)
              << "\",\"queue_position\":" << status->queue_position
              << ",\"exit_code\":" << status->exit_code
              << ",\"status_message\":\"" << EscapeJson(status->status_message)
              << "\",\"assigned_gpu_ids\":[";
    for (std::size_t gpu_index = 0; gpu_index < status->assigned_gpu_ids.size(); ++gpu_index) {
      std::cout << status->assigned_gpu_ids[gpu_index];
      if (gpu_index + 1 != status->assigned_gpu_ids.size()) {
        std::cout << ',';
      }
    }
    std::cout << "]}\n";
    return 0;
  }

  if (command == "logs") {
    if (index >= argc) {
      PrintUsage();
      return 2;
    }
    const absl::Status status = client.StreamLogs(argv[index], std::cout);
    if (!status.ok()) {
      std::cerr << status.message() << '\n';
      return 1;
    }
    return 0;
  }

  if (command == "cancel") {
    if (index >= argc) {
      PrintUsage();
      return 2;
    }
    const absl::Status status = client.CancelJob(argv[index]);
    if (!status.ok()) {
      std::cerr << status.message() << '\n';
      return 1;
    }
    std::cout << "{\"job_id\":\"" << EscapeJson(argv[index]) << "\",\"state\":\"CANCELED\"}\n";
    return 0;
  }

  if (command == "run") {
    gpu::cli::RunOptions options;
    for (; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--") {
        ++index;
        break;
      }
      if (arg == "--script") {
        options.script_path = RequireValue(argc, argv, &index);
      } else if (arg == "--image") {
        options.docker_image = RequireValue(argc, argv, &index);
      } else if (arg == "--task") {
        options.task_type = ParseTaskType(RequireValue(argc, argv, &index));
        if (options.task_type == gpu::TaskType::TASK_TYPE_UNSPECIFIED) {
          PrintUsage();
          return 2;
        }
      } else if (arg == "--gpus") {
        options.gpu_count = std::stoi(RequireValue(argc, argv, &index));
      } else if (arg == "--prefer-gpus") {
        options.preferred_gpu_ids = ParseGpuIds(RequireValue(argc, argv, &index));
      } else if (arg == "--priority") {
        options.priority = ParsePriority(RequireValue(argc, argv, &index));
        if (options.priority == gpu::Priority::PRIORITY_UNSPECIFIED) {
          PrintUsage();
          return 2;
        }
      } else if (arg == "--entrypoint") {
        options.entrypoint = RequireValue(argc, argv, &index);
      } else {
        PrintUsage();
        return 2;
      }
    }

    for (; index < argc; ++index) {
      options.args.push_back(argv[index]);
    }

    if (options.script_path.empty() || options.docker_image.empty()) {
      PrintUsage();
      return 2;
    }

    auto job_id = client.RunJob(options);
    if (!job_id.ok()) {
      std::cerr << job_id.status().message() << '\n';
      return 1;
    }

    std::cout << "{\"job_id\":\"" << EscapeJson(*job_id) << "\",\"state\":\"QUEUED\"}\n";
    return 0;
  }

  PrintUsage();
  return 2;
}
