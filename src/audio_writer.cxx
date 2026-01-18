#include "audio_writer.h"
#include "constants.h"
#include <sndfile.h>
#include <print>

namespace stems {

namespace {

// Write single WAV file
std::expected<void, WriteError> write_wav_file(
    std::filesystem::path const& path,
    std::vector<float> const& data,
    int sample_rate,
    int channels
) {
    // Configure WAV format
    auto sf_info = SF_INFO{
        .frames = 0,  // Must be 0 for output files (libsndfile requirement)
        .samplerate = sample_rate,
        .channels = channels,
        .format = SF_FORMAT_WAV | SF_FORMAT_PCM_16,
        .sections = 0,
        .seekable = 0
    };

    // Open file for writing
    auto* file = sf_open(path.c_str(), SFM_WRITE, &sf_info);
    if (!file) {
        std::println(stderr, "Failed to create file: {}", path.string());
        std::println(stderr, "libsndfile error: {}", sf_strerror(nullptr));
        return std::unexpected(WriteError::FileCreationFailed);
    }

    // Write audio data
    auto const expected_frames = static_cast<sf_count_t>(data.size() / static_cast<std::size_t>(channels));
    auto const frames_written = sf_writef_float(
        file,
        data.data(),
        expected_frames
    );

    sf_close(file);

    if (frames_written != expected_frames) {
        std::println(stderr, "Write failed: expected {} frames, wrote {}",
                     expected_frames, frames_written);
        return std::unexpected(WriteError::WriteFailed);
    }

    std::println("Wrote {} samples to {}", data.size(), path.filename().string());
    return {};
}

// Generate output filename for a stem
std::filesystem::path make_stem_path(
    std::filesystem::path const& base_path,
    std::string_view stem_name
) {
    auto path = base_path;
    path.replace_filename(
        path.stem().string() + "_" + std::string{stem_name} + path.extension().string()
    );
    return path;
}

} // anonymous namespace

std::expected<void, WriteError> write_stems(
    std::filesystem::path const& base_path,
    SeparatedStems const& stems,
    int sample_rate,
    int channels
) {
    if (!base_path.has_filename())
        return std::unexpected(WriteError::InvalidPath);

    std::println("Writing stems to {}...", base_path.parent_path().string());

    // Detect number of stems based on non-empty guitar stem
    auto const num_stems = stems.guitar.empty() ? 4uz : 6uz;

    // Write each stem to its own file
    // Order: drums, bass, other, vocals [, guitar, piano]
    auto const stem_files = std::vector<std::pair<std::string_view, std::vector<float> const*>>{
        {separation::stem_name(0, num_stems), &stems.drums},
        {separation::stem_name(1, num_stems), &stems.bass},
        {separation::stem_name(2, num_stems), &stems.other},
        {separation::stem_name(3, num_stems), &stems.vocals},
        {separation::stem_name(4, num_stems), &stems.guitar},
        {separation::stem_name(5, num_stems), &stems.piano}
    };

    auto stems_written = 0uz;
    for (auto i = 0uz; i < num_stems; ++i) {
        auto const& [name, data] = stem_files[i];

        // Skip empty stems (only guitar/piano can be empty for 4-stem model)
        if (data->empty())
            continue;

        auto const path = make_stem_path(base_path, name);
        auto const result = write_wav_file(path, *data, sample_rate, channels);

        if (!result)
            return result;

        ++stems_written;
    }

    std::println("Successfully wrote {} stems", stems_written);
    return {};
}

} // namespace stems
