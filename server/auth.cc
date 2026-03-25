#include "server/auth.h"

#include <string>
#include <utility>

#include "absl/status/status.h"

namespace gpu::server {

TokenAuth::TokenAuth(std::optional<std::string> bearer_token)
    : bearer_token_(std::move(bearer_token)) {}

bool TokenAuth::enabled() const {
  return bearer_token_.has_value() && !bearer_token_->empty();
}

absl::Status TokenAuth::Validate(const grpc::ServerContext& context) const {
  if (!enabled()) {
    return absl::OkStatus();
  }

  const auto metadata = context.client_metadata().find("authorization");
  if (metadata == context.client_metadata().end()) {
    return absl::UnauthenticatedError("missing authorization metadata");
  }

  const std::string value(metadata->second.data(), metadata->second.length());
  const std::string expected = "Bearer " + *bearer_token_;
  if (value != expected && value != *bearer_token_) {
    return absl::UnauthenticatedError("invalid bearer token");
  }

  return absl::OkStatus();
}

}  // namespace gpu::server
