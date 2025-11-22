#pragma once

#include <onnxruntime_cxx_api.h>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace stems {

// ONNX model loading errors
enum class ModelError {
    FileNotFound,
    LoadFailed,
    InvalidModel,
    InferenceFailed
};

// ONNX model wrapper for Demucs htdemucs
class OnnxModel {
public:
    // Load model from file path
    static std::expected<OnnxModel, ModelError> load(std::string_view);

    // Run inference on audio data
    // Input: interleaved stereo audio samples (float)
    // Output: 4 stems (vocals, drums, bass, other), each stereo
    std::expected<std::vector<std::vector<float>>, ModelError> infer(
        std::vector<float> const&,
        int sample_rate,
        int channels
    );

    // Get model info
    std::string_view model_path() const { return model_path_; }

private:
    OnnxModel(
        std::unique_ptr<Ort::Env>,
        std::unique_ptr<Ort::Session>,
        std::string
    );

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string model_path_;
};

// Convert ModelError to human-readable string
constexpr std::string_view error_message(ModelError error) {
    switch (error) {
        case ModelError::FileNotFound:
            return "Model file not found";
        case ModelError::LoadFailed:
            return "Failed to load model";
        case ModelError::InvalidModel:
            return "Invalid model format";
        case ModelError::InferenceFailed:
            return "Inference failed";
    }
    return "Unknown error";
}

// Compile-time tests
static_assert(error_message(ModelError::FileNotFound) == "Model file not found");
static_assert(error_message(ModelError::LoadFailed) == "Failed to load model");
static_assert(!error_message(ModelError::InferenceFailed).empty());

} // namespace stems
