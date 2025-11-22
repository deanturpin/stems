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
std::string_view error_message(ModelError);

} // namespace stems
