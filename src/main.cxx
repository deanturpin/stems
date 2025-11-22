#include "audio_validator.h"
#include <print>
#include <span>
#include <string_view>
#include <cstdlib>

namespace {

void print_usage(std::string_view program_name) {
    std::println("Usage: {} <audio_file>", program_name);
    std::println("\nSupported formats: WAV (FLAC and AIFF coming soon)");
    std::println("\nThis tool separates audio into 4 stems:");
    std::println("  - vocals");
    std::println("  - drums");
    std::println("  - bass");
    std::println("  - other");
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

} // anonymous namespace

int main(int argc, char* argv[]) {
    auto const args = std::span(argv, static_cast<std::size_t>(argc));

    if (args.size() != 2) {
        print_usage(args[0]);
        return EXIT_FAILURE;
    }

    auto const input_file = std::string_view{args[1]};

    // Validate input file
    auto const result = stems::validate_audio_file(input_file);

    if (!result) {
        std::println(stderr, "Error: {}", stems::error_message(result.error()));
        std::println(stderr, "File: {}", input_file);
        return EXIT_FAILURE;
    }

    // Print file information
    std::println("✓ Valid lossless audio file");
    print_audio_info(*result);

    std::println("\n[Stem separation not yet implemented]");
    std::println("Ready to process this file once ONNX integration is complete.");

    return EXIT_SUCCESS;
}
