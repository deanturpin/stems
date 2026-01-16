# Model Limitations and Solutions

## Fixed Input Size (Not a Bug)

The ONNX-exported Demucs model has a **fixed input size of 343,980 samples** (approximately 7.8 seconds at 44.1kHz sample rate). This is not a conversion error but rather a fundamental limitation of how PyTorch models export to ONNX with the current tooling.

### Why Fixed Size?

1. **Demucs Architecture**: The htdemucs model uses padding and reshape operations that depend on specific tensor dimensions
2. **PyTorch ONNX Export**: Dynamic shapes require the model code to be fully dynamic, but Demucs has hardcoded padding calculations
3. **torch.export Constraints**: The new PyTorch exporter (torch 2.x+) is stricter about dynamic dimensions and fails when the model specialises tensor sizes

### Industry Standard Solution: Audio Chunking

All production Demucs implementations (including the official Python version) use **chunking** to process long audio:

1. **Split** long audio into overlapping segments of ~7.8 seconds
2. **Process** each chunk through the model independently
3. **Blend** overlapping regions using crossfading to avoid artifacts
4. **Concatenate** processed chunks back into full-length stems

This approach has several advantages:
- **Memory efficiency**: Process large files without loading entire audio into GPU/CPU at once
- **Parallelisation**: Multiple chunks can be processed concurrently
- **Progress tracking**: Report progress per-chunk for long files
- **Error recovery**: Failed chunks can be retried without reprocessing entire file

## Implementation Status

### Current Status (Fixed Size Model)
- ✓ Model loads successfully with external data files
- ✓ Accepts 343,980 sample (7.8 second) input chunks
- ✗ Fails on longer audio with dimension mismatch error

### Required: Chunking Implementation

The separation pipeline needs to be updated to:

```cpp
// Pseudocode for chunking approach
auto constexpr chunk_size = 343980uz;  // Model's fixed input size
auto constexpr overlap = 17199uz;       // ~0.39s overlap for crossfade

for (auto offset = 0uz; offset < total_samples; offset += chunk_size - overlap) {
    auto const chunk_length = std::min(chunk_size, total_samples - offset);

    // Extract chunk with padding if needed
    auto chunk_left = extract_chunk(audio_left, offset, chunk_length, chunk_size);
    auto chunk_right = extract_chunk(audio_right, offset, chunk_length, chunk_size);

    // Compute STFT for chunk
    auto spec_left = stft_forward(chunk_left);
    auto spec_right = stft_forward(chunk_right);

    // Run inference on chunk
    auto stems = model.infer(chunk_left, chunk_right, spec_left, spec_right);

    // Apply crossfade and accumulate
    blend_chunk_into_output(stems, offset, chunk_length, overlap);
}
```

### Crossfading

Overlapping regions should use smooth crossfading:
- Linear fade-out on previous chunk's end
- Linear fade-in on current chunk's start
- Overlap typically 5-10% of chunk size

This eliminates audible clicks or discontinuities at chunk boundaries.

## Alternative: Dynamic Shape Export (Blocked)

Attempts to export with `dynamic_axes` fail because:

```
Constraints violated (L['mix'].size()[2])!
The model code specialized it to be a constant (343980).
```

The Demucs model would need significant refactoring to support fully dynamic shapes:
- Remove hardcoded padding calculations
- Use symbolic tensor dimensions throughout
- Test with various input sizes

This is a substantial engineering effort and goes beyond the scope of using the model for audio separation. **Chunking is the correct solution**.

## References

- Official Demucs implementation: Uses `apply_model` with chunking/overlap
- ONNX Runtime limitations: Fixed shapes are common for complex models
- PyTorch export docs: Dynamic shapes require fully symbolic operations
