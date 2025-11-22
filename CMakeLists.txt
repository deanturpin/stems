cmake_minimum_required(VERSION 3.25)
project(stems VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Find dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /opt/homebrew/include /usr/local/include /usr/include
    PATH_SUFFIXES onnxruntime onnxruntime/core/session
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS /opt/homebrew/lib /usr/local/lib /usr/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found")
endif()

message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${SNDFILE_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Main executable
add_executable(stems
    src/main.cxx
    src/audio_validator.cxx
)

target_link_libraries(stems
    ${SNDFILE_LDFLAGS}
    ${ONNXRUNTIME_LIBRARY}
)

# Installation
install(TARGETS stems DESTINATION bin)

# Tests (placeholder for now)
# enable_testing()
# add_subdirectory(tests)
