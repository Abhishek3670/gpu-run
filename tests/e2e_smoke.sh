#!/usr/bin/env bash

set -euo pipefail

SERVER_BIN="$1"
CLIENT_BIN="$2"
WORKLOAD_SCRIPT="$3"

IMAGE="${GPU_DISPATCH_E2E_IMAGE:-}"
if [[ -z "${IMAGE}" ]]; then
  echo "Skipping gpu_dispatch_e2e: GPU_DISPATCH_E2E_IMAGE is not set"
  exit 77
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Skipping gpu_dispatch_e2e: Docker image ${IMAGE} is not available locally"
  exit 77
fi

TEST_ROOT="$(mktemp -d)"
SERVER_LOG="${TEST_ROOT}/gpu-server.log"
BUNDLE_ROOT="${TEST_ROOT}/bundles"
PORT="$((55000 + ($$ % 1000)))"
SERVER_ADDR="127.0.0.1:${PORT}"
JOB_ID=""
SERVER_PID=""

cleanup() {
  local exit_code=$?
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ $exit_code -ne 0 && -f "${SERVER_LOG}" ]]; then
    echo "--- gpu-server log ---"
    cat "${SERVER_LOG}"
  fi
  rm -rf "${TEST_ROOT}"
}
trap cleanup EXIT

mkdir -p "${BUNDLE_ROOT}"
"${SERVER_BIN}" \
  --bind "${SERVER_ADDR}" \
  --bundle-root "${BUNDLE_ROOT}" \
  --allow-image "${IMAGE}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

wait_for_server() {
  local attempt
  for attempt in $(seq 1 20); do
    if timeout 5s "${CLIENT_BIN}" --server "${SERVER_ADDR}" list-gpus >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "gpu-server did not become ready in time"
  return 1
}

extract_json_field() {
  local field="$1"
  sed -n "s/.*\"${field}\":\"\([^\"]*\)\".*/\1/p"
}

wait_for_server

run_output="$("${CLIENT_BIN}" \
  --server "${SERVER_ADDR}" \
  run \
  --script "${WORKLOAD_SCRIPT}" \
  --image "${IMAGE}" \
  --entrypoint /bin/sh \
  -- ./run.sh)"

JOB_ID="$(printf "%s\n" "${run_output}" | extract_json_field job_id)"
if [[ -z "${JOB_ID}" ]]; then
  echo "Failed to parse job_id from run output: ${run_output}"
  exit 1
fi

saw_running=0
for attempt in $(seq 1 30); do
  status_json="$(timeout 5s "${CLIENT_BIN}" --server "${SERVER_ADDR}" status "${JOB_ID}")"
  state="$(printf "%s\n" "${status_json}" | extract_json_field state)"
  if [[ "${state}" == "RUNNING" ]]; then
    saw_running=1
    break
  fi
  if [[ "${state}" == "SUCCEEDED" ]]; then
    break
  fi
  if [[ "${state}" == "FAILED" || "${state}" == "CANCELED" ]]; then
    echo "Job entered unexpected terminal state: ${status_json}"
    exit 1
  fi
  sleep 1
done

if [[ "${saw_running}" -eq 1 ]]; then
  list_output="$(timeout 5s "${CLIENT_BIN}" --server "${SERVER_ADDR}" list-gpus)"
  if ! printf "%s\n" "${list_output}" | grep -q "\"locked_job_id\":\"${JOB_ID}\""; then
    echo "Expected GPU lock for job ${JOB_ID}, got: ${list_output}"
    exit 1
  fi
fi

final_state=""
for attempt in $(seq 1 45); do
  status_json="$(timeout 5s "${CLIENT_BIN}" --server "${SERVER_ADDR}" status "${JOB_ID}")"
  final_state="$(printf "%s\n" "${status_json}" | extract_json_field state)"
  if [[ "${final_state}" == "SUCCEEDED" ]]; then
    break
  fi
  if [[ "${final_state}" == "FAILED" || "${final_state}" == "CANCELED" ]]; then
    echo "Job ended in unexpected state: ${status_json}"
    exit 1
  fi
  sleep 1
done

if [[ "${final_state}" != "SUCCEEDED" ]]; then
  echo "Job did not reach SUCCEEDED in time"
  exit 1
fi

logs_output="$(timeout 10s "${CLIENT_BIN}" --server "${SERVER_ADDR}" logs "${JOB_ID}")"
if ! printf "%s\n" "${logs_output}" | grep -q "E2E_START"; then
  echo "Missing E2E_START in logs: ${logs_output}"
  exit 1
fi
if ! printf "%s\n" "${logs_output}" | grep -q "E2E_DONE"; then
  echo "Missing E2E_DONE in logs: ${logs_output}"
  exit 1
fi

echo "gpu_dispatch_e2e passed for image ${IMAGE}"
