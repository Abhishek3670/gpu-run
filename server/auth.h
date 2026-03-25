#pragma once

#include <optional>
#include <string>

#include <grpcpp/grpcpp.h>

#include "absl/status/status.h"

namespace gpu::server {

class TokenAuth {
 public:
  explicit TokenAuth(std::optional<std::string> bearer_token = std::nullopt);

  [[nodiscard]] bool enabled() const;
  [[nodiscard]] absl::Status Validate(const grpc::ServerContext& context) const;

 private:
  std::optional<std::string> bearer_token_;
};

}  // namespace gpu::server
