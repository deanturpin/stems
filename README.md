# Stems

High-quality audio stem separation using Demucs v4 via ONNX Runtime. Separate audio tracks into vocals, drums, bass, and other instruments with true multi-core parallelisation.

## Features

- **High Quality**: State-of-the-art Demucs v4 (htdemucs) model with 9.0+ dB SDR
- **Fast Processing**: Multi-threaded C++ implementation, no Python GIL limitations
- **Multiple Interfaces**:
  - CLI for developers and automation
  - Web UI for general use (local or remote)
  - Desktop GUI (planned)
- **Flexible Deployment**:
  - Run locally for privacy and offline use
  - Deploy as web service on VPS
  - Queue system for batch processing
- **4-Stem Separation**: Vocals, drums, bass, other

## Architecture

```
┌─────────────┐
│   Web UI    │  HTML/CSS/JS (works local and remote)
└──────┬──────┘
       │ HTTP/WebSocket
┌──────┴──────┐
│  API Server │  Crow C++ HTTP server
└──────┬──────┘
       │
┌──────┴──────┐
│ C++ Backend │  ONNX Runtime + Demucs model
└─────────────┘  Multi-threaded processing
```

## Library Comparison

| Library | Quality (SDR) | Speed | Parallel | ONNX Support | Notes |
|---------|--------------|-------|----------|--------------|-------|
| **Demucs v4 (htdemucs)** | **9.2 dB** | Slower | ✅ Excellent | ✅ Yes | **Selected** - Best quality, multi-threaded C++ |
| Spleeter | 7.0-7.3 dB | Fast | ⚠️ Limited | ❌ No | Python/TensorFlow, less accurate |
| Open-Unmix | 8.2-8.4 dB | Medium | ⚠️ Limited | ⚠️ Partial | Good quality, PyTorch-based |
| MDX-Net | 8.5-9.0 dB | Medium | ✅ Good | ✅ Yes | Best for vocals specifically |

**Selection Rationale**: Demucs v4 (htdemucs) chosen for:
- Highest quality separation (9.2+ dB SDR on MUSDB18)
- True multi-core parallelisation via ONNX Runtime C++ (no Python GIL)
- Hybrid architecture (time-domain + transformer) handles phase information better
- Active ONNX conversion support from Mixxx project

## Technology Stack

- **Backend**: C++23 (tracking latest standard) with ONNX Runtime
- **Platforms**: macOS and latest Linux (no Windows support)
- **Build System**: CMake with top-level Makefile wrapper
- **Models**: Demucs v4 (htdemucs) exported to ONNX
- **Audio I/O**: libsndfile (WAV, FLAC, AIFF - lossless only)
- **GPU Support**: CUDA (optional), Metal/CoreML on macOS
- **CPU Optimisation**: Thread pool, parallel batch processing

## Quick Start

### CLI Usage

```bash
# Single file
stems input.wav --output ./stems

# Batch processing
stems *.wav --batch --jobs 4

# Quality/speed trade-off
stems input.wav --model htdemucs      # best quality (slower)
stems input.wav --model mdx_extra     # faster, good quality
```

### Web Service (planned)

```bash
# Start local server
stems-server --port 8080

# Visit http://localhost:8080
# Drag and drop audio files, download stems
```

### Deploy to VPS (planned)

```bash
# Docker deployment
docker pull deanturpin/stems
docker run -p 8080:8080 deanturpin/stems

# Or systemd service
make install-service
```

## Building from Source

### Prerequisites

Requires latest C++ compiler with C++23 support.

```bash
# macOS
brew install cmake onnxruntime libsndfile

# Latest Ubuntu/Debian
apt install cmake libonnxruntime-dev libsndfile1-dev g++-13

# Arch Linux (rolling release - always latest)
pacman -S cmake onnxruntime libsndfile
```

### Build

```bash
git clone https://github.com/deanturpin/stems.git
cd stems
make
```

## Project Structure

```
stems/
├── src/              # Core C++ stem separation logic and CLI
│   ├── main.cxx
│   ├── stem_processor.cxx
│   └── onnx_wrapper.cxx
├── include/          # Public headers
│   ├── stem_processor.h
│   └── onnx_wrapper.h
├── models/           # ONNX model files (download separately)
│   └── htdemucs.onnx
├── tests/            # Unit tests
├── CMakeLists.txt    # CMake build configuration
└── Makefile          # Top-level build wrapper
```

## Performance

### CPU-Only (8-core VPS)
- 4-minute song: ~5-8 minutes (htdemucs)
- 4-minute song: ~3-5 minutes (mdx_extra)
- Parallel batch: 3 jobs concurrently

### With GPU (CUDA)
- 4-minute song: ~30-60 seconds
- 10-50x speedup over CPU

## Roadmap

- [x] Research and select best model (Demucs v4)
- [ ] C++ backend with ONNX Runtime integration
- [ ] CLI interface
- [ ] HTTP API server with job queue
- [ ] Web UI with drag-and-drop
- [ ] Docker deployment
- [ ] Model download and management
- [ ] Progress tracking via WebSocket
- [ ] Email notifications for completed jobs
- [ ] File size and rate limiting
- [ ] Desktop GUI (Qt)
- [ ] Extended stem separation (piano, guitar - via UVR models)
- [ ] GPU acceleration support

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## Licence

MIT Licence - see [LICENCE](LICENCE) for details.

## Acknowledgements

- [Demucs](https://github.com/facebookresearch/demucs) by Meta Research for the outstanding separation model
- [ONNX Runtime](https://onnxruntime.ai/) for cross-platform ML inference
- [Mixxx GSoC 2025](https://mixxx.org/news/2025-10-27-gsoc2025-demucs-to-onnx-dhunstack/) for Demucs ONNX conversion work
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) community for model research

## References

- [Demucs Paper](https://arxiv.org/abs/2111.03600) - Hybrid Spectrogram and Waveform Source Separation
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [Audio Source Separation Benchmarks](https://paperswithcode.com/task/audio-source-separation)
