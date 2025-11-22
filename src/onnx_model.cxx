#include "onnx_model.h"
#include <filesystem>
#include <print>

namespace stems {

namespace {

// Check if file exists
bool file_exists(std::string_view path) {
    return std::filesystem::exists(path);
}

} // anonymous namespace

OnnxModel::OnnxModel(
    std::unique_ptr<Ort::Env> env,
    std::unique_ptr<Ort::Session> session,
    std::string path
) : env_(std::move(env)),
    session_(std::move(session)),
    model_path_(std::move(path)) {}

std::expected<OnnxModel, ModelError> OnnxModel::load(std::string_view model_path) {
    if (!file_exists(model_path))
        return std::unexpected(ModelError::FileNotFound);

    try {
        // Create ONNX Runtime environment
        auto env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "stems");

        // Configure session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(0); // Use all available threads
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Load model
        auto session = std::make_unique<Ort::Session>(
            *env,
            model_path.data(),
            session_options
        );

        // Verify model loaded successfully
        auto const num_inputs = session->GetInputCount();
        auto const num_outputs = session->GetOutputCount();

        std::println("Model loaded successfully:");
        std::println("  Inputs: {}", num_inputs);
        std::println("  Outputs: {}", num_outputs);

        return OnnxModel(
            std::move(env),
            std::move(session),
            std::string{model_path}
        );

    } catch (Ort::Exception const& e) {
        std::println(stderr, "ONNX Runtime error: {}", e.what());
        return std::unexpected(ModelError::LoadFailed);
    } catch (...) {
        return std::unexpected(ModelError::LoadFailed);
    }
}

std::expected<std::vector<std::vector<float>>, ModelError> OnnxModel::infer(
    std::vector<float> const& audio_data,
    int sample_rate,
    int channels
) {
    // Placeholder implementation - will be completed in next step
    // This requires STFT preprocessing and proper tensor preparation
    std::println("Inference not yet implemented");
    std::println("  Audio samples: {}", audio_data.size());
    std::println("  Sample rate: {}", sample_rate);
    std::println("  Channels: {}", channels);

    return std::unexpected(ModelError::InferenceFailed);
}

std::string_view error_message(ModelError error) {
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

} // namespace stems
