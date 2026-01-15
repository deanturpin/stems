# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Audio stem separation tool using Demucs v4 via ONNX Runtime. Separates audio into vocals, drums, bass, and other instruments. Native C++ implementation starting with CLI, potential web service later.

**Platform Support**: macOS and latest Linux only (no Windows support)
**Language**: Latest C++ standard (currently C++23, will track C++26 when available)
**Active Development**: Track progress in [issue #25](https://github.com/deanturpin/projects/issues/25)

## Architecture

Currently focused on native CLI tool. Future web service architecture:
- **Processing Core**: ONNX Runtime backend with multi-threaded batch processing
- **CLI**: Direct interface to processing core
- **Server** (future): Crow C++ HTTP server with WebSocket for progress updates
- **Web UI** (future): HTML/CSS/JS drag-and-drop interface

Core separation uses Demucs v4 (htdemucs) model exported to ONNX format.

## Building and Development

### Build System

Top-level Makefile wraps CMake for simple workflow:
```bash
make                  # Configure CMake and build all targets
make stems            # Build CLI tool only
make test             # Run unit tests
make clean            # Clean build artifacts
```

CMake handles the actual build configuration and compilation.

### CLI Usage
```bash
stems input.wav --output ./stems
stems input.flac --output ./stems
stems *.wav --batch --jobs 4
```

**Supported Formats**: WAV, FLAC, AIFF (lossless only - no MP3 or lossy formats)

### Dependencies
- Latest C++ compiler with C++23 support (GCC 13+, Clang 16+)
- CMake 3.25+
- ONNX Runtime (model inference)
- libsndfile (audio I/O for WAV/FLAC/AIFF)
- FFTW3 (Fast Fourier Transform library)

**Note**: Track latest C++ standard and Linux versions - update compiler requirements as new standards become available.

CMakeLists.txt uses pkg-config to find libsndfile and FFTW3, and manual find_path/find_library for ONNX Runtime (checks /opt/homebrew, /usr/local, /usr).

### Platform-specific Setup
```bash
# macOS
brew install cmake onnxruntime libsndfile fftw

# Ubuntu/Debian
apt install cmake libonnxruntime-dev libsndfile1-dev libfftw3-dev g++-13

# Arch Linux
pacman -S cmake onnxruntime libsndfile fftw
```

## Code Structure

- `src/`: Implementation files (main.cxx, stem_processor.cxx, onnx_model.cxx, stft.cxx, audio_validator.cxx, audio_writer.cxx)
- `include/`: Public headers matching src/ structure
- `models/`: ONNX model files (downloaded separately, not in git - see models/README.md)
- `docs/`: Architecture and design documentation (SIGNAL_PROCESSING.md, ROADMAP.md)
- `tests/`: Unit tests (placeholder structure)
- `CMakeLists.txt`: CMake build configuration with compiler warnings (-Wall -Wextra -Wpedantic -Werror)
- `Makefile`: Top-level convenience wrapper
- `watch.sh`: Development helper for continuous builds

## Key Implementation Details

### Core Architecture Components

The processing pipeline follows this flow:
1. **AudioValidator** (`audio_validator.h/cxx`) - Validates input file format and extracts metadata
2. **STFT** (`stft.h/cxx`) - Short-Time Fourier Transform preprocessing using FFTW3
3. **OnnxModel** (`onnx_model.h/cxx`) - ONNX Runtime wrapper for model inference
4. **StemProcessor** (`stem_processor.h/cxx`) - Orchestrates the complete separation pipeline
5. **AudioWriter** (`audio_writer.h/cxx`) - Writes separated stems to disk

### Error Handling Strategy
- Uses `std::expected<T, Error>` throughout (C++23 feature, no exceptions)
- Each component defines its own error enum
- Error messages via `constexpr` functions for zero runtime overhead
- Fail fast on validation errors before expensive processing

### Model Integration
- ONNX Runtime C++ API for model inference
- Models in `models/` directory (excluded from git due to size)
- Primary model: htdemucs.onnx or htdemucs.ort (~300MB)
- Model path defaults to `models/htdemucs.onnx`, can be overridden via CLI
- See `models/README.md` for conversion instructions using sevagh/demucs.onnx

### Audio Processing
- Input/output via libsndfile (WAV supported, FLAC/AIFF coming soon)
- 4-stem output: vocals, drums, bass, other
- STFT preprocessing uses FFTW3 (selected for performance, see `docs/SIGNAL_PROCESSING.md`)
- Validation happens before processing to fail fast on unsupported formats
- Currently single-file processing, batch mode planned

### Future Web Service (not current priority)
- Crow C++ HTTP server
- REST endpoints for job submission and status
- WebSocket for real-time progress updates
- Job queue system for concurrent processing control

### Performance Targets
- CPU: Process 4-min song in 5-8 minutes (htdemucs)
- GPU (CUDA): Process 4-min song in 30-60 seconds
- Support configurable parallel job limits

## Build and Development Workflow

1. Top-level `make` handles everything via CMake
2. Native builds for macOS (arm64/x86_64) and Linux (x86_64)
3. No Windows support planned
4. Docker containerisation deferred until web service implementation
