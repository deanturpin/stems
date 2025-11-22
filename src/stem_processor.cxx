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
std::vector<float> interleave_stereo(
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

    // Compute STFT for both channels
    std::println("Computing STFT for left channel...");
    auto stft_left_result = stft_.forward(left);
    if (!stft_left_result) {
        std::println(stderr, "STFT (left) failed: {}", error_message(stft_left_result.error()));
        return std::unexpected(ProcessingError::StftFailed);
    }

    std::println("Computing STFT for right channel...");
    auto stft_right_result = stft_.forward(right);
    if (!stft_right_result) {
        std::println(stderr, "STFT (right) failed: {}", error_message(stft_right_result.error()));
        return std::unexpected(ProcessingError::StftFailed);
    }

    auto const& spec_left = stft_left_result.value();
    auto const& spec_right = stft_right_result.value();
    std::println("STFT completed: {} frames x {} bins",
                 spec_left.num_frames, spec_left.num_bins);

    // Run ONNX inference
    std::println("Running stem separation inference...");
    auto inference_result = model_.infer(left, right, spec_left, spec_right);
    if (!inference_result) {
        std::println(stderr, "ONNX inference failed");
        return std::unexpected(ProcessingError::InferenceFailed);
    }

    auto const& stem_spectrograms = inference_result.value();
    if (stem_spectrograms.size() != 4) {
        std::println(stderr, "Expected 4 stems, got {}", stem_spectrograms.size());
        return std::unexpected(ProcessingError::OutputGenerationFailed);
    }

    // Apply iSTFT to each stem to convert back to time domain
    std::println("Converting stems to time domain...");
    auto vocals_result = stft_.inverse(stem_spectrograms[0]);
    auto drums_result = stft_.inverse(stem_spectrograms[1]);
    auto bass_result = stft_.inverse(stem_spectrograms[2]);
    auto other_result = stft_.inverse(stem_spectrograms[3]);

    if (!vocals_result || !drums_result || !bass_result || !other_result) {
        std::println(stderr, "iSTFT conversion failed for one or more stems");
        return std::unexpected(ProcessingError::OutputGenerationFailed);
    }

    // Interleave left and right channels for each stem
    // For now, using mono output (same data for both channels)
    // TODO: Process right channel separately and properly interleave
    auto const& vocals = vocals_result.value();
    auto const& drums = drums_result.value();
    auto const& bass = bass_result.value();
    auto const& other = other_result.value();

    std::println("Stem separation complete!");
    std::println("  Vocals: {} samples", vocals.size());
    std::println("  Drums: {} samples", drums.size());
    std::println("  Bass: {} samples", bass.size());
    std::println("  Other: {} samples", other.size());

    return SeparatedStems{
        .vocals = interleave_stereo(vocals, vocals),  // Duplicate mono to stereo
        .drums = interleave_stereo(drums, drums),
        .bass = interleave_stereo(bass, bass),
        .other = interleave_stereo(other, other)
    };
}

} // namespace stems
