#pragma once

#include <expected>
#include <string>
#include <string_view>

namespace stems {

// Audio file validation errors
enum class ValidationError {
    FileNotFound,
    UnsupportedFormat,
    LossyFormat,
    CorruptedFile,
    UnknownError
};

// Audio file information
struct AudioInfo {
    int sample_rate;
    int channels;
    long frames;
    std::string format_name;
};

// Validate audio file format (lossless only: WAV, FLAC, AIFF)
// Returns AudioInfo on success, ValidationError on failure
std::expected<AudioInfo, ValidationError> validate_audio_file(std::string_view);

// Convert ValidationError to human-readable string
constexpr std::string_view error_message(ValidationError error) {
    switch (error) {
        case ValidationError::FileNotFound:
            return "File not found or cannot be opened";
        case ValidationError::UnsupportedFormat:
            return "Unsupported audio format (only WAV supported currently)";
        case ValidationError::LossyFormat:
            return "Lossy format not supported";
        case ValidationError::CorruptedFile:
            return "File appears to be corrupted";
        case ValidationError::UnknownError:
            return "Unknown error occurred";
    }
    return "Unknown error";
}

// Compile-time tests
static_assert(error_message(ValidationError::FileNotFound) ==
              "File not found or cannot be opened");
static_assert(error_message(ValidationError::LossyFormat) ==
              "Lossy format not supported");
static_assert(!error_message(ValidationError::FileNotFound).empty());

} // namespace stems
