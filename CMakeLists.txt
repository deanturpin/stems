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

# ONNX Runtime - will be needed soon
# find_package(onnxruntime REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${SNDFILE_INCLUDE_DIRS}
)

# Main executable
add_executable(stems
    src/main.cxx
    src/audio_validator.cxx
)

target_link_libraries(stems
    ${SNDFILE_LDFLAGS}
)

# Installation
install(TARGETS stems DESTINATION bin)

# Tests (placeholder for now)
# enable_testing()
# add_subdirectory(tests)
