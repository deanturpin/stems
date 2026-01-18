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

// Extract chunk from audio with zero-padding if needed
std::vector<float> extract_chunk(
    std::vector<float> const& audio,
    std::size_t offset,
    std::size_t target_size
) {
    auto chunk = std::vector<float>(target_size, 0.0f);
    auto const available = offset < audio.size()
        ? std::min(target_size, audio.size() - offset)
        : 0uz;

    if (available > 0)
        std::copy_n(audio.begin() + offset, available, chunk.begin());

    return chunk;
}

// Apply overlap-add with linear crossfade to avoid artifacts
// Blends chunk into output buffer at given offset with smooth transitions
void blend_chunk(
    std::vector<float>& output,
    std::vector<float> const& chunk,
    std::size_t offset,
    std::size_t overlap_size,
    bool is_first_chunk,
    bool is_last_chunk
) {
    auto const chunk_size = chunk.size();
    auto const end = std::min(offset + chunk_size, output.size());

    for (auto i = 0uz; i < chunk_size and (offset + i) < end; ++i) {
        auto const output_idx = offset + i;
        auto weight = 1.0f;

        // Fade in at start of chunk (except for first chunk)
        if (!is_first_chunk and i < overlap_size) {
            weight = static_cast<float>(i) / static_cast<float>(overlap_size);
        }

        // Fade out at end of chunk (except for last chunk)
        auto const dist_from_end = chunk_size - i;
        if (!is_last_chunk and dist_from_end <= overlap_size) {
            auto const fade_out_weight = static_cast<float>(dist_from_end) / static_cast<float>(overlap_size);
            weight = std::min(weight, fade_out_weight);
        }

        // Overlap-add: blend with existing content
        output[output_idx] = output[output_idx] * (1.0f - weight) + chunk[i] * weight;
    }
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
    auto const num_samples = left.size();

    // Calculate chunking parameters
    auto constexpr chunk_size = separation::model_chunk_size;
    auto constexpr overlap = separation::chunk_overlap;
    auto const step = chunk_size - overlap;
    auto const num_chunks = (num_samples + step - 1) / step;

    std::println("Chunking audio: {} chunks of {} samples with {} overlap",
                 num_chunks, chunk_size, overlap);

    // Initialize output buffers for all possible stems (support up to 6 stems)
    auto drums_out = std::vector<float>(num_samples, 0.0f);
    auto bass_out = std::vector<float>(num_samples, 0.0f);
    auto other_out = std::vector<float>(num_samples, 0.0f);
    auto vocals_out = std::vector<float>(num_samples, 0.0f);
    auto guitar_out = std::vector<float>(num_samples, 0.0f);
    auto piano_out = std::vector<float>(num_samples, 0.0f);

    auto num_detected_stems = 0uz;  // Will be set after first inference

    // Process each chunk
    for (auto chunk_idx = 0uz; chunk_idx < num_chunks; ++chunk_idx) {
        auto const offset = chunk_idx * step;
        auto const is_first = chunk_idx == 0;
        auto const is_last = chunk_idx == num_chunks - 1;

        std::println("Processing chunk {}/{} (offset: {} samples)",
                     chunk_idx + 1, num_chunks, offset);

        // Extract chunk with padding
        auto const left_chunk = extract_chunk(left, offset, chunk_size);
        auto const right_chunk = extract_chunk(right, offset, chunk_size);

        // Compute STFT for chunk
        auto stft_left_result = stft_.forward(left_chunk);
        auto stft_right_result = stft_.forward(right_chunk);

        if (!stft_left_result or !stft_right_result) {
            std::println(stderr, "STFT failed for chunk {}", chunk_idx + 1);
            return std::unexpected(ProcessingError::StftFailed);
        }

        // Run inference on chunk
        auto const& spec_left = stft_left_result.value();
        auto const& spec_right = stft_right_result.value();

        auto inference_result = model_.infer(left_chunk, right_chunk, spec_left, spec_right);
        if (!inference_result) {
            std::println(stderr, "Inference failed for chunk {}", chunk_idx + 1);
            return std::unexpected(ProcessingError::InferenceFailed);
        }

        auto const& stem_specs = inference_result.value();

        // Detect number of stems from first chunk
        if (chunk_idx == 0) {
            num_detected_stems = stem_specs.size();
            if (num_detected_stems != 4 and num_detected_stems != 6) {
                std::println(stderr, "Unexpected number of stems: {} (expected 4 or 6)", num_detected_stems);
                return std::unexpected(ProcessingError::OutputGenerationFailed);
            }
            std::println("Detected {}-stem model", num_detected_stems);
        }

        // Verify consistent stem count
        if (stem_specs.size() != num_detected_stems) {
            std::println(stderr, "Inconsistent stem count: expected {}, got {}",
                         num_detected_stems, stem_specs.size());
            return std::unexpected(ProcessingError::OutputGenerationFailed);
        }

        // Model outputs time-domain audio directly (stored in spec.real)
        // htdemucs outputs in order: drums, bass, other, vocals [, guitar, piano]
        blend_chunk(drums_out, stem_specs[0].real, offset, overlap, is_first, is_last);
        blend_chunk(bass_out, stem_specs[1].real, offset, overlap, is_first, is_last);
        blend_chunk(other_out, stem_specs[2].real, offset, overlap, is_first, is_last);
        blend_chunk(vocals_out, stem_specs[3].real, offset, overlap, is_first, is_last);

        // Process additional stems for 6-stem model
        if (num_detected_stems == 6) {
            blend_chunk(guitar_out, stem_specs[4].real, offset, overlap, is_first, is_last);
            blend_chunk(piano_out, stem_specs[5].real, offset, overlap, is_first, is_last);
        }
    }

    std::println("Stem separation complete!");
    std::println("  Drums: {} samples", drums_out.size());
    std::println("  Bass: {} samples", bass_out.size());
    std::println("  Other: {} samples", other_out.size());
    std::println("  Vocals: {} samples", vocals_out.size());
    if (num_detected_stems == 6) {
        std::println("  Guitar: {} samples", guitar_out.size());
        std::println("  Piano: {} samples", piano_out.size());
    }

    return SeparatedStems{
        .drums = interleave_stereo(drums_out, drums_out),
        .bass = interleave_stereo(bass_out, bass_out),
        .other = interleave_stereo(other_out, other_out),
        .vocals = interleave_stereo(vocals_out, vocals_out),  // TODO: Process right channel
        .guitar = num_detected_stems == 6 ? interleave_stereo(guitar_out, guitar_out) : std::vector<float>{},
        .piano = num_detected_stems == 6 ? interleave_stereo(piano_out, piano_out) : std::vector<float>{}
    };
}

} // namespace stems
