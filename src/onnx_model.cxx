#include "onnx_model.h"
#include <algorithm>
#include <filesystem>
#include <print>

namespace stems {

namespace {

// Minimum expected file size for htdemucs model (300MB typical, allow 100MB minimum)
constexpr auto min_model_size = 100'000'000uz; // 100 MB

// Check if file exists and has reasonable size
std::expected<void, ModelError> validate_model_file(std::string_view path) {
    if (!std::filesystem::exists(path))
        return std::unexpected(ModelError::FileNotFound);

    auto total_size = std::filesystem::file_size(path);

    // Check for external data file (ONNX models can store weights separately)
    auto const data_path = std::filesystem::path{path}.string() + ".data";
    if (std::filesystem::exists(data_path))
        total_size += std::filesystem::file_size(data_path);

    if (total_size < min_model_size) {
        std::println(stderr, "Model file is too small: {} bytes", total_size);
        std::println(stderr, "Expected at least {} bytes (~300MB for htdemucs)", min_model_size);
        std::println(stderr, "");
        std::println(stderr, "The model file appears to be corrupted or incomplete.");
        std::println(stderr, "Please regenerate it using the instructions in models/README.md:");
        std::println(stderr, "  1. Run ./scripts/download_model.sh");
        std::println(stderr, "  2. Or follow manual conversion steps in models/README.md");
        return std::unexpected(ModelError::InvalidModel);
    }

    return {};
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
    // Validate model file exists and has reasonable size
    if (auto const validation = validate_model_file(model_path); !validation)
        return std::unexpected(validation.error());

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

std::expected<std::vector<Spectrogram>, ModelError> OnnxModel::infer(
    std::vector<float> const& audio_left,
    std::vector<float> const& audio_right,
    Spectrogram const& spec_left,
    Spectrogram const& spec_right
) {
    try {
        // Demucs htdemucs model expects dual inputs:
        // 1. Time-domain waveform: [batch, channels, time]
        // 2. Spectrogram: [batch, channels, freq, time] with complex-as-channels

        auto const num_samples = audio_left.size();
        auto const num_frames = spec_left.num_frames;
        auto const num_bins = spec_left.num_bins;

        std::println("Preparing ONNX tensors:");
        std::println("  Time-domain: [1, 2, {}]", num_samples);
        std::println("  Spectrogram: [1, 2, {}, {}]", num_bins, num_frames);

        // Prepare time-domain input tensor [1, 2, time]
        // Interleave left and right channels
        auto waveform_data = std::vector<float>(2 * num_samples);
        for (auto i = 0uz; i < num_samples; ++i) {
            waveform_data[i] = audio_left[i];                    // Channel 0 (left)
            waveform_data[num_samples + i] = audio_right[i];     // Channel 1 (right)
        }

        auto const waveform_shape = std::array<int64_t, 3>{1, 2, static_cast<int64_t>(num_samples)};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        auto waveform_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            waveform_data.data(),
            waveform_data.size(),
            waveform_shape.data(),
            waveform_shape.size()
        );

        // Prepare spectrogram input tensor [1, 4, freq, time]
        // Complex-as-channels: real_left, imag_left, real_right, imag_right
        auto const spec_size = num_bins * num_frames;
        auto spectrogram_data = std::vector<float>(4 * spec_size);

        // Channel 0: real components (left)
        std::copy(spec_left.real.begin(), spec_left.real.end(),
                  spectrogram_data.begin());

        // Channel 1: imaginary components (left)
        std::copy(spec_left.imag.begin(), spec_left.imag.end(),
                  spectrogram_data.begin() + spec_size);

        // Channel 2: real components (right)
        std::copy(spec_right.real.begin(), spec_right.real.end(),
                  spectrogram_data.begin() + 2 * spec_size);

        // Channel 3: imaginary components (right)
        std::copy(spec_right.imag.begin(), spec_right.imag.end(),
                  spectrogram_data.begin() + 3 * spec_size);

        auto const spec_shape = std::array<int64_t, 4>{
            1, 4,
            static_cast<int64_t>(num_bins),
            static_cast<int64_t>(num_frames)
        };

        auto spectrogram_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            spectrogram_data.data(),
            spectrogram_data.size(),
            spec_shape.data(),
            spec_shape.size()
        );

        // Prepare input/output names
        auto const input_names = std::array{
            "waveform",
            "spectrogram"
        };

        auto const output_names = std::array{
            "vocals",
            "drums",
            "bass",
            "other"
        };

        // Prepare input tensor array
        auto input_tensors = std::vector<Ort::Value>{};
        input_tensors.push_back(std::move(waveform_tensor));
        input_tensors.push_back(std::move(spectrogram_tensor));

        std::println("Running ONNX inference...");

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size()
        );

        std::println("Inference complete, extracting {} stems", output_tensors.size());

        // Extract output spectrograms (4 stems)
        auto result = std::vector<Spectrogram>{};
        result.reserve(4);

        for (auto& tensor : output_tensors) {
            auto* data = tensor.GetTensorMutableData<float>();
            auto const shape_info = tensor.GetTensorTypeAndShapeInfo();
            auto const shape = shape_info.GetShape();

            // Expected output shape: [1, 4, freq, time]
            // Extract complex-as-channels for each stem
            auto stem_spec = Spectrogram{
                .real = std::vector<float>(spec_size),
                .imag = std::vector<float>(spec_size),
                .num_frames = num_frames,
                .num_bins = num_bins
            };

            // For now, just copy the real components (channel 0)
            // TODO: Properly handle complex-as-channels output
            std::copy(data, data + spec_size, stem_spec.real.begin());
            std::copy(data + spec_size, data + 2 * spec_size, stem_spec.imag.begin());

            result.push_back(std::move(stem_spec));
        }

        return result;

    } catch (Ort::Exception const& e) {
        std::println(stderr, "ONNX inference error: {}", e.what());
        return std::unexpected(ModelError::InferenceFailed);
    } catch (...) {
        std::println(stderr, "Unknown error during inference");
        return std::unexpected(ModelError::InferenceFailed);
    }
}

} // namespace stems
