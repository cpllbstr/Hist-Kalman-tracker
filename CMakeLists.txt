cmake_minimum_required(VERSION 3.11)
project( DisplayImage )
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED )


set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${protobuf_VERSION}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)

include(FetchContent)
FetchContent_Declare(
    gRPC
    GIT_REPOSITORY https://github.com/grpc/grpc
    GIT_TAG        v1.27.3
)
FetchContent_GetProperties(gRPC)
if(NOT gRPC_POPULATED)
    FetchContent_Populate(gRPC)
    add_subdirectory(${grpc_SOURCE_DIR} ${grpc_BINARY_DIR})
endif()


FetchContent_Declare(
    TOML
    GIT_REPOSITORY https://github.com/ToruNiina/toml11
    GIT_TAG        v3.3.1
)
FetchContent_GetProperties(TOML)
if(NOT TOML_POPULATED)
    FetchContent_Populate(TOML)
    add_subdirectory(${toml_SOURCE_DIR} ${toml_BINARY_DIR})
endif()

aux_source_directory(${CMAKE_SOURCE_DIR} SRCS)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
add_executable(client main.cpp ./grpc/yolo_grpc.grpc.pb.cc ./grpc/yolo_grpc.pb.cc)
target_include_directories(client PUBLIC ./grpc  ./headers ${toml_SOURCE_DIR})
target_link_libraries( client ${OpenCV_LIBS} grpc++)