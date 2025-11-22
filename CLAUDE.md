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
- ONNX Runtime
- libsndfile (audio I/O, supports WAV/FLAC/AIFF)

**Note**: Track latest C++ standard and Linux versions - update compiler requirements as new standards become available.

### Platform-specific Setup
```bash
# macOS
brew install cmake onnxruntime libsndfile

# Ubuntu/Debian
apt install cmake libonnxruntime-dev libsndfile1-dev

# Arch Linux
pacman -S cmake onnxruntime libsndfile
```

## Code Structure

- `src/`: Core stem separation logic (ONNX wrapper, audio processing, CLI)
- `include/`: Public headers
- `models/`: ONNX model files (downloaded separately, not in git)
- `tests/`: Unit tests
- `CMakeLists.txt`: CMake build configuration
- `Makefile`: Top-level convenience wrapper

## Key Implementation Details

### Model Integration
- Use ONNX Runtime C++ API for model inference
- Models live in `models/` directory (excluded from git due to size)
- Primary model: htdemucs.onnx (~300MB)
- Alternative: mdx_extra.onnx (faster, lower quality)

### Audio Processing
- Input/output via libsndfile (WAV, FLAC, AIFF only - reject MP3 and lossy formats)
- 4-stem output: vocals, drums, bass, other
- Multi-threaded batch processing for parallel jobs
- Validate input format before processing (fail fast on lossy formats)

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
