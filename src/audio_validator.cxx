#include "audio_validator.h"
#include <sndfile.h>
#include <array>

namespace stems {

namespace {

// Currently only WAV is supported (FLAC/AIFF coming later)
bool is_supported_format(int format) {
    auto const major_format = format & SF_FORMAT_TYPEMASK;
    return major_format == SF_FORMAT_WAV;
}

} // anonymous namespace

std::expected<AudioInfo, ValidationError> validate_audio_file(std::string_view path) {
    SF_INFO sf_info{};

    // Open file for reading
    auto const file = sf_open(path.data(), SFM_READ, &sf_info);
    if (!file)
        return std::unexpected(ValidationError::FileNotFound);

    // RAII cleanup
    struct FileCloser {
        SNDFILE* f;
        ~FileCloser() { sf_close(f); }
    } closer{file};

    // Check if format is supported (WAV only for now)
    if (!is_supported_format(sf_info.format))
        return std::unexpected(ValidationError::UnsupportedFormat);

    return AudioInfo{
        .sample_rate = sf_info.samplerate,
        .channels = sf_info.channels,
        .frames = sf_info.frames,
        .format_name = "WAV"
    };
}

std::string_view error_message(ValidationError error) {
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

} // namespace stems
