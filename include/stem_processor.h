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

// Separated audio stems (supports both 4 and 6 stem models)
struct SeparatedStems {
    std::vector<float> drums;
    std::vector<float> bass;
    std::vector<float> other;
    std::vector<float> vocals;
    std::vector<float> guitar;  // Only used for 6-stem model
    std::vector<float> piano;   // Only used for 6-stem model

    // Get stem by index (0=drums, 1=bass, 2=other, 3=vocals, 4=guitar, 5=piano)
    std::vector<float>* get_stem(std::size_t index) {
        switch (index) {
            case 0: return &drums;
            case 1: return &bass;
            case 2: return &other;
            case 3: return &vocals;
            case 4: return &guitar;
            case 5: return &piano;
            default: return nullptr;
        }
    }
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
};

} // namespace stems
