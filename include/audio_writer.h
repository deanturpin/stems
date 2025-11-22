#pragma once

#include "stem_processor.h"
#include <expected>
#include <filesystem>
#include <string_view>

namespace stems {

// Audio writing errors
enum class WriteError {
    FileCreationFailed,
    WriteFailed,
    InvalidFormat,
    InvalidPath
};

// Convert WriteError to human-readable string
constexpr std::string_view error_message(WriteError error) {
    switch (error) {
        case WriteError::FileCreationFailed:
            return "Failed to create output file";
        case WriteError::WriteFailed:
            return "Failed to write audio data";
        case WriteError::InvalidFormat:
            return "Invalid audio format";
        case WriteError::InvalidPath:
            return "Invalid output path";
    }
    return "Unknown error";
}

// Compile-time tests
static_assert(error_message(WriteError::FileCreationFailed) == "Failed to create output file");
static_assert(error_message(WriteError::WriteFailed) == "Failed to write audio data");
static_assert(!error_message(WriteError::InvalidPath).empty());

// Write separated stems to WAV files
// Creates 4 files: {base}_vocals.wav, {base}_drums.wav, {base}_bass.wav, {base}_other.wav
std::expected<void, WriteError> write_stems(
    std::filesystem::path const&,
    SeparatedStems const&,
    int sample_rate,
    int channels
);

} // namespace stems
