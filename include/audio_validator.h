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
std::string_view error_message(ValidationError);

} // namespace stems
