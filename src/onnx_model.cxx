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
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

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

        // Verify input spectrogram sizes match expected dimensions
        if (spec_left.real.size() != spec_size or spec_left.imag.size() != spec_size or
            spec_right.real.size() != spec_size or spec_right.imag.size() != spec_size) {
            std::println(stderr, "Spectrogram size mismatch!");
            return std::unexpected(ModelError::InferenceFailed);
        }

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

        // Prepare input/output names (must match ONNX model's actual names)
        auto const input_names = std::array{
            "input",  // Time-domain waveform input
            "x"       // Spectrogram input
        };

        auto const output_names = std::array{
            "output",   // Primary output (spectrograms for all 4 stems)
            "add_67"    // Secondary output (time-domain waveforms)
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

        std::println("Inference complete, got {} output tensors", output_tensors.size());

        // The model outputs TWO tensors:
        // output_tensors[0] = "output" [1, 4, 4, 2048, 336] - spectrogram outputs
        // output_tensors[1] = "add_67" [1, 4, 2, 343980] - time-domain waveforms (what we want!)
        //
        // We use the time-domain output directly to avoid iSTFT conversion
        // Shape: [batch=1, stems=4, channels=2, samples=343980]

        if (output_tensors.size() < 2) {
            std::println(stderr, "Expected 2 output tensors, got {}", output_tensors.size());
            return std::unexpected(ModelError::InferenceFailed);
        }

        // Get the time-domain output (add_67)
        auto& time_domain_output = output_tensors[1];
        auto* data = time_domain_output.GetTensorMutableData<float>();
        auto const shape_info = time_domain_output.GetTensorTypeAndShapeInfo();
        auto const shape = shape_info.GetShape();

        // Verify shape: [batch, stems, channels, samples]
        // Expect shape[0]=1 (batch), shape[1]=4 or 6 (stems), shape[2]=2 (stereo), shape[3]=samples
        if (shape.size() != 4 or shape[0] != 1 or shape[2] != 2) {
            std::println(stderr, "Unexpected time-domain output shape: [{}, {}, {}, {}]",
                         shape.size() > 0 ? shape[0] : 0,
                         shape.size() > 1 ? shape[1] : 0,
                         shape.size() > 2 ? shape[2] : 0,
                         shape.size() > 3 ? shape[3] : 0);
            return std::unexpected(ModelError::InferenceFailed);
        }

        auto const num_stems = static_cast<std::size_t>(shape[1]);

        // Validate number of stems (4 or 6)
        if (num_stems != 4 and num_stems != 6) {
            std::println(stderr, "Unexpected number of stems: {} (expected 4 or 6)", num_stems);
            return std::unexpected(ModelError::InferenceFailed);
        }
        auto const num_channels = static_cast<std::size_t>(shape[2]);
        auto const samples_per_stem = static_cast<std::size_t>(shape[3]);

        std::println("Extracted {} stems with {} channels, {} samples each",
                     num_stems, num_channels, samples_per_stem);

        // For now, convert to "fake" spectrograms just to match the interface
        // TODO: Change interface to return time-domain audio directly
        auto result = std::vector<Spectrogram>{};
        result.reserve(num_stems);

        for (auto stem_idx = 0uz; stem_idx < num_stems; ++stem_idx) {
            // Extract left channel for this stem and store as "real" component
            // Shape indexing: data[batch * (stems * channels * samples) + stem * (channels * samples) + channel * samples + sample]
            auto const stem_offset = stem_idx * num_channels * samples_per_stem;
            auto const left_offset = stem_offset + 0 * samples_per_stem;  // Channel 0 (left)

            // Store time-domain audio in the real component (misuse of Spectrogram structure)
            auto stem_spec = Spectrogram{
                .real = std::vector<float>(data + left_offset, data + left_offset + samples_per_stem),
                .imag = std::vector<float>(samples_per_stem, 0.0f),  // Unused
                .num_frames = 1,  // Dummy value
                .num_bins = samples_per_stem  // Store sample count here (hack)
            };

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
