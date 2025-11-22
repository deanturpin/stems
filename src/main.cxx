#include "audio_validator.h"
#include "audio_writer.h"
#include "onnx_model.h"
#include "stem_processor.h"
#include <filesystem>
#include <print>
#include <span>
#include <string_view>
#include <cstdlib>
#include <sndfile.h>

namespace {

void print_usage(std::string_view program_name) {
    std::println("Usage: {} <audio_file> [model_path]", program_name);
    std::println("\nSupported formats: WAV (FLAC and AIFF coming soon)");
    std::println("\nThis tool separates audio into 4 stems:");
    std::println("  - vocals");
    std::println("  - drums");
    std::println("  - bass");
    std::println("  - other");
    std::println("\nModel path defaults to: models/htdemucs.onnx");
}

void print_audio_info(stems::AudioInfo const& info) {
    std::println("Audio file information:");
    std::println("  Format: {}", info.format_name);
    std::println("  Sample rate: {} Hz", info.sample_rate);
    std::println("  Channels: {}", info.channels);
    std::println("  Frames: {}", info.frames);

    auto const duration_seconds = static_cast<double>(info.frames) / info.sample_rate;
    std::println("  Duration: {:.2f} seconds", duration_seconds);
}

// Load audio file into memory
std::expected<std::vector<float>, stems::ValidationError> load_audio(
    std::string_view path,
    stems::AudioInfo const& info
) {
    auto file_info = SF_INFO{};
    auto* file = sf_open(path.data(), SFM_READ, &file_info);
    if (!file)
        return std::unexpected(stems::ValidationError::FileNotFound);

    auto const total_samples = static_cast<std::size_t>(info.frames * info.channels);
    auto audio_data = std::vector<float>(total_samples);

    auto const frames_read = sf_readf_float(file, audio_data.data(), info.frames);
    sf_close(file);

    if (frames_read != info.frames)
        return std::unexpected(stems::ValidationError::CorruptedFile);

    return audio_data;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    auto const args = std::span(argv, static_cast<std::size_t>(argc));

    if (args.size() < 2 || args.size() > 3) {
        print_usage(args[0]);
        return EXIT_FAILURE;
    }

    auto const input_file = std::string_view{args[1]};
    auto const model_path = (args.size() == 3)
        ? std::string_view{args[2]}
        : std::string_view{"models/htdemucs.onnx"};

    // Validate input file
    auto const info_result = stems::validate_audio_file(input_file);
    if (!info_result) {
        std::println(stderr, "Error: {}", stems::error_message(info_result.error()));
        std::println(stderr, "File: {}", input_file);
        return EXIT_FAILURE;
    }

    auto const& info = *info_result;
    std::println("✓ Valid lossless audio file");
    print_audio_info(info);

    // Load ONNX model
    std::println("\nLoading model: {}", model_path);
    auto model_result = stems::OnnxModel::load(model_path);
    if (!model_result) {
        std::println(stderr, "Error: {}", stems::error_message(model_result.error()));
        return EXIT_FAILURE;
    }

    // Load audio data
    std::println("\nLoading audio data...");
    auto audio_result = load_audio(input_file, info);
    if (!audio_result) {
        std::println(stderr, "Error loading audio: {}",
                     stems::error_message(audio_result.error()));
        return EXIT_FAILURE;
    }

    auto const& audio_data = *audio_result;
    std::println("Loaded {} samples", audio_data.size());

    // Process stems
    std::println("\nSeparating stems...");
    auto processor = stems::StemProcessor{std::move(*model_result)};
    auto stems_result = processor.process(audio_data, info.sample_rate, info.channels);

    if (!stems_result) {
        std::println(stderr, "Error: {}", stems::error_message(stems_result.error()));
        return EXIT_FAILURE;
    }

    // Write output files
    std::println("\nWriting output files...");
    auto const output_path = std::filesystem::path{input_file};
    auto write_result = stems::write_stems(
        output_path,
        *stems_result,
        info.sample_rate,
        info.channels
    );

    if (!write_result) {
        std::println(stderr, "Error: {}", stems::error_message(write_result.error()));
        return EXIT_FAILURE;
    }

    std::println("\n✓ Stem separation complete!");
    return EXIT_SUCCESS;
}
