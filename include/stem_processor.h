#pragma once

#include "onnx_model.h"
#include "stft.h"
#include <expected>
#include <vector>

namespace stems {

// Stem separation processing errors
enum class ProcessingError {
    StftFailed,
    InferenceFailed,
    InvalidAudio,
    OutputGenerationFailed
};

// Convert ProcessingError to human-readable string
constexpr std::string_view error_message(ProcessingError error) {
    switch (error) {
        case ProcessingError::StftFailed:
            return "STFT processing failed";
        case ProcessingError::InferenceFailed:
            return "ONNX inference failed";
        case ProcessingError::InvalidAudio:
            return "Invalid audio format or data";
        case ProcessingError::OutputGenerationFailed:
            return "Failed to generate output stems";
    }
    return "Unknown error";
}

// Compile-time tests
static_assert(error_message(ProcessingError::StftFailed) == "STFT processing failed");
static_assert(error_message(ProcessingError::InferenceFailed) == "ONNX inference failed");
static_assert(!error_message(ProcessingError::OutputGenerationFailed).empty());

// Separated audio stems
struct SeparatedStems {
    std::vector<float> vocals;
    std::vector<float> drums;
    std::vector<float> bass;
    std::vector<float> other;
};

// Main stem separation processor
class StemProcessor {
public:
    explicit StemProcessor(OnnxModel);

    // Separate stereo audio into 4 stems
    // Input: interleaved stereo audio samples
    // Output: 4 separated stems (each stereo)
    std::expected<SeparatedStems, ProcessingError> process(
        std::vector<float> const&,
        int sample_rate,
        int channels
    );

private:
    OnnxModel model_;
    StftProcessor stft_;

    // Prepare input tensors for ONNX model
    std::expected<std::vector<Ort::Value>, ProcessingError> prepare_inputs(
        std::vector<float> const&,
        Spectrogram const&
    );

    // Extract stems from ONNX output tensors
    std::expected<SeparatedStems, ProcessingError> extract_stems(
        std::vector<Ort::Value>&
    );
};

} // namespace stems
