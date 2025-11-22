# Signal Processing Library Options

For Demucs stem separation, we need STFT (Short-Time Fourier Transform) and iSTFT implementations. The Demucs ONNX model expects spectrograms as input, so STFT preprocessing must happen in C++ before inference.

## FFT Library Comparison

| Library | Speed | Size | License | SIMD | Complexity | Notes |
|---------|-------|------|---------|------|------------|-------|
| **FFTW3** | Fastest (588ns) | ~2MB | GPL/Commercial | ✅ Yes | High | Industry standard, requires planning |
| **pffft** | Fast (1255ns) | 32KB | BSD-like | ✅ Yes | Low | Good balance, single-header friendly |
| **KissFFT** | Slow (6553ns) | 20KB | BSD | ❌ No | Very Low | Simplest, no vectorisation |
| **Eigen::FFT** | Medium | Large | MPL2 | ⚠️ Optional | Medium | Uses KissFFT or FFTW backend |

*Benchmark data from [project-gemmi/benchmarking-fft](https://github.com/project-gemmi/benchmarking-fft) for n=512 complex-to-complex transforms*

## Recommended Approach: Use Existing Demucs.cpp STFT

The [sevagh/demucs.cpp](https://github.com/sevagh/demucs.cpp) and [sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx) projects already implement STFT/iSTFT for Demucs with:

- ✅ Complex-as-channels representation (required for Demucs)
- ✅ Proper padding and windowing
- ✅ Tested and working with Demucs models
- ✅ Uses Eigen for linear algebra
- ✅ MIT license

### Implementation Strategy

**Option 1: Vendor demucs.cpp STFT code**
- Copy STFT/iSTFT implementation from demucs.cpp
- ~500 lines of tested code
- Requires Eigen dependency
- Pros: Battle-tested, known to work
- Cons: Eigen dependency adds complexity

**Option 2: Use pffft + custom STFT**
- Implement STFT/iSTFT wrapper around pffft
- Lightweight (32KB)
- Pros: Small footprint, BSD license
- Cons: Need to implement windowing, overlap-add ourselves

**Option 3: Use FFTW3**
- Best performance
- Pros: Fastest, widely used
- Cons: GPL license (may require commercial license), larger size

## Demucs STFT Requirements

Based on [Demucs ONNX documentation](https://github.com/sevagh/demucs.onnx):

1. **Input**: Time-domain stereo waveform
2. **Processing**:
   - Apply STFT with proper window (Hann)
   - Convert to magnitude spectrogram
   - Complex-as-channels representation (real/imag as separate channels)
3. **ONNX Model Input**: Dual inputs
   - Time-domain waveform
   - Spectrogram (magnitude with complex-as-channels)
4. **Output**: 4 spectrograms (one per stem)
5. **Post-processing**: Apply iSTFT to each stem

## Decision: FFTW3

**Selected: FFTW3** - Industry standard, fastest performance

Integrated as of 2024-11-22:
- Version: 3.3.10
- Installed via Homebrew
- Linked through pkg-config

## Previous Recommendation

**Alternative: Vendor demucs.cpp STFT**

Rationale:
- Proven to work with Demucs ONNX models
- Saves implementation time
- We already need Eigen for other linear algebra
- Can optimise later if needed

Implementation plan:
1. Add Eigen as dependency
2. Extract STFT/iSTFT code from demucs.cpp
3. Adapt to our codebase style
4. Add unit tests with known inputs/outputs

## References

- [FFTW Benchmarks](https://www.fftw.org/benchfft/ffts.html)
- [FFT Library Comparison (STFC)](https://epubs.stfc.ac.uk/manifestation/45434584/RAL-TR-2020-003.pdf)
- [Fastest FFT in C++ - Signal Processing Stack Exchange](https://dsp.stackexchange.com/questions/24375/fastest-implementation-of-fft-in-c)
- [sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx)
- [sevagh/demucs.cpp](https://github.com/sevagh/demucs.cpp)
- [Mixxx GSOC 2025 - Demucs to ONNX](https://mixxx.org/news/2025-10-27-gsoc2025-demucs-to-onnx-dhunstack/)
