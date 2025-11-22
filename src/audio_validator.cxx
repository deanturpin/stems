#include "audio_validator.h"
#include <array>
#include <sndfile.h>

namespace stems {

namespace {

// Currently only WAV is supported (FLAC/AIFF coming later)
constexpr bool is_supported_format(int format) {
  auto const major_format = format & SF_FORMAT_TYPEMASK;
  return major_format == SF_FORMAT_WAV;
}

static_assert(is_supported_format(SF_FORMAT_WAV));
static_assert(!is_supported_format(SF_FORMAT_FLAC));
static_assert(!is_supported_format(SF_FORMAT_AIFF));

} // anonymous namespace

std::expected<AudioInfo, ValidationError>
validate_audio_file(std::string_view path) {
  SF_INFO sf_info{};

  // Open file for reading
  auto const file = sf_open(path.data(), SFM_READ, &sf_info);
  if (!file)
    return std::unexpected(ValidationError::FileNotFound);

  // RAII cleanup
  struct FileCloser {
    SNDFILE *f;
    ~FileCloser() { sf_close(f); }
  } closer{file};

  // Check if format is supported (WAV only for now)
  if (!is_supported_format(sf_info.format))
    return std::unexpected(ValidationError::UnsupportedFormat);

  // Only stereo is supported for now (mono/multichannel support is future work)
  if (sf_info.channels != 2)
    return std::unexpected(ValidationError::UnsupportedFormat);

  return AudioInfo{.sample_rate = sf_info.samplerate,
                   .channels = sf_info.channels,
                   .frames = sf_info.frames,
                   .format_name = "WAV"};
}

} // namespace stems
