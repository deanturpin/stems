# Roadmap

## Current Sprint: Core Functionality

**Status: Pipeline Complete - Ready for Model**

- [x] Build system with C++23
- [x] Audio file validation (WAV)
- [x] ONNX Runtime integration
- [x] FFTW3 integration (fftw3f for float precision)
- [x] Compile-time validation with constexpr/static_assert
- [x] STFT/iSTFT implementation (4096 window, 1024 hop, Hann window)
- [x] ONNX tensor preparation (complex-as-channels, dual inputs)
- [x] Full inference pipeline implementation
- [x] iSTFT reconstruction for output stems
- [x] Output WAV file generation (4 stems)
- [ ] **Obtain htdemucs.onnx model file** ← Blocking next steps
- [ ] Verify tensor shapes match actual model expectations
- [ ] End-to-end test with example.wav
- [ ] Implement proper stereo separation (currently duplicating mono for L/R)

## Future Enhancements

### Visualization (#future-viz)
**Priority: Medium**

Add spectrogram and waveform visualization for debugging and quality verification.

**Options:**
1. **Spectrogram images** (PNG/SVG)
   - Input audio spectrogram
   - Each of 4 separated stems
   - Side-by-side comparison
   - Library: stb_image_write (single header) or matplotlib-cpp

2. **Real-time terminal visualization**
   - ASCII spectrum analyzer during processing
   - Progress bars for FFT stages
   - No additional dependencies

3. **Waveform plots**
   - Before/after comparison
   - Overlaid stems

**Implementation:**
- Optional `--visualize` or `--save-spectrograms` flag
- Generate alongside WAV output
- Helpful for tuning separation quality

**Dependencies:**
- Option 1: stb_image_write (BSD, single header)
- Option 2: None (pure terminal output)

### Additional Format Support (#future-formats)
**Priority: High**

- [ ] FLAC support (lossless)
- [ ] AIFF support (lossless)
- Maintain "lossless only" policy

### Performance Optimizations (#future-perf)
**Priority: Low**

- [ ] GPU acceleration (CUDA/Metal)
- [ ] Batch processing with job queue
- [ ] Multi-file parallel processing

### Model Support (#future-models)
**Priority: Medium**

- [ ] 6-stem model support (add piano, guitar)
- [ ] Model auto-download and management
- [ ] Multiple quality tiers (fast/balanced/best)

### User Experience (#future-ux)
**Priority: Medium**

- [ ] Progress reporting via WebSocket (for web UI)
- [ ] Desktop GUI (Qt)
- [ ] Web service deployment

## Non-Goals

- Windows support
- Lossy format support (MP3, AAC, etc.)
- Real-time processing
- Plugin/VST format
