# Session Summary - Stem Separation Implementation

## Overview
Successfully implemented complete audio stem separation pipeline with chunking, overlap-add blending, and support for both 4-stem and 6-stem models.

## Major Accomplishments

### 1. Fixed FFTW Crash (Abort trap: 6)
**Problem**: Program crashed after processing 2-3 audio chunks with "Abort trap: 6"

**Root Cause**: Creating and destroying FFTW plans and buffers on every STFT call caused memory corruption issues

**Solution**:
- Cache FFTW plan and buffers in StftProcessor class
- Initialize once in constructor, destroy in destructor
- Changed from FFTW_ESTIMATE to FFTW_MEASURE for better performance
- Properly manage lifecycle to avoid dangling references

**Files Changed**:
- [include/stft.h](../include/stft.h) - Added cached plan and buffers as member variables
- [src/stft.cxx](../src/stft.cxx) - Implemented initialize_fftw() and updated forward()

**Result**: Successfully processes arbitrarily long audio files (tested with 510-second file, 69 chunks)

### 2. Fixed Stem Ordering Bug
**Problem**: Generated stem files had swapped labels (vocals file contained drums, etc.)

**Root Cause**: Assumed htdemucs outputs vocals first, but actual order is: drums, bass, other, vocals

**Solution**:
- Updated stem_names array in constants.h to match model output order
- Corrected stem extraction indices in stem_processor.cxx
- Fixed static_assert tests for validation

**Files Changed**:
- [include/constants.h](../include/constants.h) - Reordered stem_names array
- [src/stem_processor.cxx](../src/stem_processor.cxx) - Fixed extraction order

**References**:
- [Demucs GitHub Repository](https://github.com/facebookresearch/demucs)
- [PyTorch Hybrid Demucs Tutorial](https://docs.pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html)

### 3. Implemented Audio Chunking with Overlap-Add
**Challenge**: htdemucs ONNX model has fixed input size (343,980 samples ≈ 7.8 seconds)

**Solution**:
- Split audio into fixed-size chunks with overlap (5% = 17,199 samples)
- Extract chunks with zero-padding for edge cases
- Blend chunks using linear crossfade to prevent artifacts
- Process each chunk through STFT → Model → Output

**Implementation**:
```cpp
// Chunking parameters
constexpr auto chunk_size = 343980uz;      // Model input size
constexpr auto chunk_overlap = 17199uz;    // 5% overlap for smooth blending
auto const step = chunk_size - overlap;

// Process each chunk
for (auto chunk_idx = 0uz; chunk_idx < num_chunks; ++chunk_idx) {
    auto const offset = chunk_idx * step;

    // Extract chunk with padding
    auto const left_chunk = extract_chunk(left, offset, chunk_size);

    // Process through STFT and model
    // ...

    // Blend into output with crossfading
    blend_chunk(output, chunk_audio, offset, overlap, is_first, is_last);
}
```

**Files Changed**:
- [include/constants.h](../include/constants.h) - Added chunking constants
- [src/stem_processor.cxx](../src/stem_processor.cxx) - Implemented extract_chunk() and blend_chunk()

### 4. Added 6-Stem Model Support
**Feature**: Support for htdemucs_6s model with additional guitar and piano separation

**Implementation**:
- Updated constants.h with stem_names_6 array
- Modified download_model.sh to accept parameter for model selection
- Maintains backward compatibility with 4-stem default

**Usage**:
```bash
./scripts/download_model.sh     # 4-stem (default): drums, bass, other, vocals
./scripts/download_model.sh 6   # 6-stem: drums, bass, other, vocals, guitar, piano
```

**Note**: Piano separation quality is experimental and may have artifacts

**References**:
- [Unleash the Power of Demucs v4](https://lame.buanzo.org/max4live_blog/unleash-the-power-of-demucs-v4-in-your-productions.html)
- [Demucs Release Notes](https://github.com/facebookresearch/demucs/blob/main/docs/release.md)

### 5. Set Up Release Infrastructure
**Feature**: Automated binary builds and GitHub releases

**Implementation**:
- GitHub Actions workflow for macOS and Linux builds
- Triggers on `release` branch pushes and version tags (v*)
- Automatic creation of GitHub releases with platform tarballs

**Workflow**: [.github/workflows/release.yml](../.github/workflows/release.yml)

**Branch Strategy**:
- `main`: Development and integration work
- `release`: Stable production deployments (triggers builds)
- Tags `v*`: Create GitHub releases with binaries

## Technical Details

### Audio Processing Pipeline
1. **Validation**: Check file format (WAV/FLAC/AIFF, lossless only)
2. **Loading**: Read audio data via libsndfile
3. **De-interleaving**: Split stereo into left/right channels
4. **Chunking**: Split into fixed-size chunks with overlap
5. **STFT**: Compute spectrograms using FFTW3
6. **Inference**: Run htdemucs model via ONNX Runtime
7. **Blending**: Overlap-add with linear crossfade
8. **Output**: Write separated stems as 16-bit WAV files

### Performance Characteristics
- **Processing Speed**: ~69 chunks for 510-second audio (varies with hardware)
- **Memory Usage**: Efficient (one chunk at a time)
- **Output Quality**: 44.1kHz, 16-bit PCM stereo WAV

## Current Limitations

### Known TODOs
1. **Right Channel**: Currently duplicating left channel instead of processing separately
2. **Dynamic Model Detection**: Hardcoded to expect 4 stems
3. **Progress Reporting**: No real-time progress updates
4. **CLI Options**: Limited configuration options

### Code Comments
See `TODO` markers in:
- [src/stem_processor.cxx:137](../src/stem_processor.cxx#L137) - Right channel processing
- [src/onnx_model.cxx:227](../src/onnx_model.cxx#L227) - Interface cleanup for time-domain audio

## Testing Results

### Test File: example.wav
- **Format**: WAV, 44.1kHz stereo
- **Duration**: 510.09 seconds (8.5 minutes)
- **Frames**: 22,495,146
- **Chunks**: 69 chunks with 17,199 sample overlap

### Output Files
All stems generated successfully:
- `example_drums.wav` - 86MB
- `example_bass.wav` - 86MB
- `example_other.wav` - 86MB
- `example_vocals.wav` - 86MB

## Next Steps

1. **Test 6-Stem Model**: Complete download and test htdemucs_6s
2. **Create Release**: Tag v0.1.0 and trigger automated builds
3. **Right Channel**: Implement proper stereo processing
4. **Documentation**: Update README with usage examples
5. **Performance**: Optimize chunk size and overlap parameters

## References

### Documentation
- [Demucs Paper (arXiv)](https://arxiv.org/abs/2111.03600) - Hybrid Spectrogram and Waveform Source Separation
- [ONNX Runtime Docs](https://onnxruntime.ai/docs/)
- [FFTW Documentation](https://www.fftw.org/fftw3_doc/)

### Related Projects
- [sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx) - ONNX conversion reference
- [sevagh/demucs.cpp](https://github.com/sevagh/demucs.cpp) - C++ reference implementation
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) - GUI application

---

*Last Updated: 2026-01-16*
*Model: htdemucs (4-stem), htdemucs_6s (6-stem)*
*Platform: macOS and Linux (C++23)*
