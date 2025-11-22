#include "stem_processor.h"
#include "constants.h"
#include <print>

namespace stems {

namespace {

// De-interleave stereo audio into separate left/right channels
std::pair<std::vector<float>, std::vector<float>> deinterleave_stereo(
    std::vector<float> const& interleaved
) {
    auto const num_samples = interleaved.size() / 2uz;
    auto left = std::vector<float>(num_samples);
    auto right = std::vector<float>(num_samples);

    for (auto i = 0uz; i < num_samples; ++i) {
        left[i] = interleaved[i * 2uz];
        right[i] = interleaved[i * 2uz + 1uz];
    }

    return {left, right};
}

// Interleave separate left/right channels into stereo
[[maybe_unused]] std::vector<float> interleave_stereo(
    std::vector<float> const& left,
    std::vector<float> const& right
) {
    auto const num_samples = left.size();
    auto interleaved = std::vector<float>(num_samples * 2uz);

    for (auto i = 0uz; i < num_samples; ++i) {
        interleaved[i * 2uz] = left[i];
        interleaved[i * 2uz + 1uz] = right[i];
    }

    return interleaved;
}

} // anonymous namespace

StemProcessor::StemProcessor(OnnxModel model)
    : model_(std::move(model)), stft_{} {}

std::expected<SeparatedStems, ProcessingError> StemProcessor::process(
    std::vector<float> const& audio,
    int sample_rate,
    int channels
) {
    if (channels != 2) {
        std::println(stderr, "Only stereo audio is supported (got {} channels)", channels);
        return std::unexpected(ProcessingError::InvalidAudio);
    }

    std::println("Processing {} samples at {} Hz ({} channels)",
                 audio.size(), sample_rate, channels);

    // De-interleave stereo input
    auto const [left, right] = deinterleave_stereo(audio);

    // Compute STFT for left channel
    auto stft_result = stft_.forward(left);
    if (!stft_result) {
        std::println(stderr, "STFT failed: {}", error_message(stft_result.error()));
        return std::unexpected(ProcessingError::StftFailed);
    }

    auto const& spectrogram = stft_result.value();
    std::println("STFT completed: {} frames x {} bins",
                 spectrogram.num_frames, spectrogram.num_bins);

    // Prepare inputs for ONNX model
    auto inputs_result = prepare_inputs(audio, spectrogram);
    if (!inputs_result)
        return std::unexpected(inputs_result.error());

    // Run inference
    auto inference_result = model_.infer(audio, sample_rate, channels);
    if (!inference_result) {
        std::println(stderr, "ONNX inference failed");
        return std::unexpected(ProcessingError::InferenceFailed);
    }

    // For now, return placeholder stems until full inference pipeline is complete
    // This will be replaced with actual ONNX output processing
    auto const num_samples = audio.size();
    return SeparatedStems{
        .vocals = std::vector<float>(num_samples, 0.0f),
        .drums = std::vector<float>(num_samples, 0.0f),
        .bass = std::vector<float>(num_samples, 0.0f),
        .other = std::vector<float>(num_samples, 0.0f)
    };
}

std::expected<std::vector<Ort::Value>, ProcessingError> StemProcessor::prepare_inputs(
    std::vector<float> const& audio,
    Spectrogram const& spec
) {
    // Demucs htdemucs model expects dual inputs:
    // 1. Time-domain waveform: [batch, channels, time]
    // 2. Magnitude spectrogram: [batch, channels, freq, time]
    //
    // This is a placeholder - full implementation requires:
    // - Proper tensor shape preparation
    // - Complex-as-channels representation
    // - Batch dimension handling

    std::println("Preparing ONNX inputs (placeholder):");
    std::println("  Audio samples: {}", audio.size());
    std::println("  Spectrogram: {} frames x {} bins", spec.num_frames, spec.num_bins);

    // Return empty for now - will be implemented after verifying model input shapes
    return std::vector<Ort::Value>{};
}

std::expected<SeparatedStems, ProcessingError> StemProcessor::extract_stems(
    [[maybe_unused]] std::vector<Ort::Value>& outputs
) {
    // Extract and process ONNX output tensors
    // Apply iSTFT to convert spectrograms back to time domain
    // This is a placeholder for the complete implementation

    std::println("Extracting stems from ONNX outputs (placeholder)");

    return std::unexpected(ProcessingError::OutputGenerationFailed);
}

} // namespace stems
