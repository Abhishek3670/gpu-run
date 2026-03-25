cmake_minimum_required(VERSION 3.20)

project(gpu_dispatch_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

find_package(absl CONFIG REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(gRPC QUIET)

if(NOT gRPC_FOUND)
  pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)
  if(NOT TARGET gRPC::grpc++)
    add_library(gRPC::grpc++ INTERFACE IMPORTED)
    target_link_libraries(gRPC::grpc++ INTERFACE PkgConfig::GRPCPP)
  endif()

  find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin REQUIRED)
  if(NOT TARGET gRPC::grpc_cpp_plugin)
    add_executable(gRPC::grpc_cpp_plugin IMPORTED GLOBAL)
    set_target_properties(
      gRPC::grpc_cpp_plugin
      PROPERTIES IMPORTED_LOCATION ${GRPC_CPP_PLUGIN_EXECUTABLE}
    )
  endif()
endif()

find_path(NVML_INCLUDE_DIR nvml.h HINTS /usr/local/cuda/include /usr/include)
find_library(
  NVML_LIBRARY
  NAMES nvidia-ml libnvidia-ml.so.1
  HINTS /usr/lib/wsl/lib /usr/lib/x86_64-linux-gnu
  REQUIRED
)

if(NOT NVML_INCLUDE_DIR)
  message(FATAL_ERROR "NVML headers were not found. Install the CUDA toolkit in WSL2.")
endif()

set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/gpu_service.proto)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

set(PROTO_SRCS
  ${GENERATED_DIR}/gpu_service.pb.cc
  ${GENERATED_DIR}/gpu_service.grpc.pb.cc
)
set(PROTO_HDRS
  ${GENERATED_DIR}/gpu_service.pb.h
  ${GENERATED_DIR}/gpu_service.grpc.pb.h
)

add_custom_command(
  OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
  COMMAND protobuf::protoc
  ARGS
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
    --cpp_out=${GENERATED_DIR}
    --grpc_out=${GENERATED_DIR}
    --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
  VERBATIM
)

add_library(gpu_proto ${PROTO_SRCS})
target_include_directories(gpu_proto PUBLIC ${GENERATED_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  gpu_proto
  PUBLIC
    protobuf::libprotobuf
    gRPC::grpc++
    absl::status
    absl::statusor
)

add_library(scheduler_lib scheduler/scheduler.cc)
target_include_directories(scheduler_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  scheduler_lib
  PUBLIC
    absl::status
    absl::statusor
)

add_library(gpu_manager_lib gpu_manager/gpu_manager.cc)
target_include_directories(gpu_manager_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${NVML_INCLUDE_DIR})
target_link_libraries(
  gpu_manager_lib
  PUBLIC
    absl::status
    absl::statusor
    absl::strings
    Threads::Threads
    ${NVML_LIBRARY}
)

add_library(worker_lib worker/worker.cc)
target_include_directories(worker_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  worker_lib
  PUBLIC
    scheduler_lib
    absl::status
    absl::statusor
    absl::strings
    absl::time
    Threads::Threads
)

add_library(server_lib server/auth.cc server/server.cc)
target_include_directories(server_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  server_lib
  PUBLIC
    gpu_proto
    scheduler_lib
    gpu_manager_lib
    worker_lib
    absl::status
    absl::statusor
    absl::strings
    absl::time
    gRPC::grpc++
)

add_library(client_lib cli/client.cc)
target_include_directories(client_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  client_lib
  PUBLIC
    gpu_proto
    absl::status
    absl::statusor
    absl::strings
    gRPC::grpc++
)

add_executable(gpu-server server/main.cc)
target_link_libraries(gpu-server PRIVATE server_lib)

add_executable(gpu-run cli/main.cc)
target_link_libraries(gpu-run PRIVATE client_lib)

if(BUILD_TESTING)
  add_executable(scheduler_test tests/scheduler_test.cc)
  target_link_libraries(scheduler_test PRIVATE scheduler_lib gpu_manager_lib worker_lib)
  add_test(NAME scheduler_test COMMAND scheduler_test)

  add_executable(log_streamer_test tests/log_streamer_test.cc)
  target_link_libraries(log_streamer_test PRIVATE worker_lib)
  add_test(NAME log_streamer_test COMMAND log_streamer_test)
endif()
