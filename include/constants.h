#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace stems {

// Audio processing constants
namespace audio {

// Supported sample rates
constexpr std::array supported_sample_rates = {44100, 48000, 96000};

// Standard CD quality
constexpr auto cd_sample_rate = 44100;
constexpr auto stereo_channels = 2;
constexpr auto mono_channels = 1;

// Validate sample rate is supported
constexpr bool is_supported_sample_rate(int rate) {
    for (auto const supported : supported_sample_rates)
        if (rate == supported)
            return true;
    return false;
}

// Compile-time tests
static_assert(is_supported_sample_rate(44100));
static_assert(is_supported_sample_rate(48000));
static_assert(!is_supported_sample_rate(22050));
static_assert(cd_sample_rate == 44100);
static_assert(stereo_channels == 2);

} // namespace audio

// Stem separation constants
namespace separation {

// Model chunk size (fixed input dimension from ONNX export)
// htdemucs model expects exactly 343,980 samples per chunk
constexpr auto model_chunk_size = 343980uz;  // ~7.8 seconds at 44.1kHz

// Overlap between chunks for smooth blending (5% of chunk size)
constexpr auto chunk_overlap = model_chunk_size / 20uz;  // 17,199 samples (~0.39s)

// Stem names for different model variants
// htdemucs (4 stems): drums, bass, other, vocals
constexpr std::array<std::string_view, 4uz> stem_names_4 = {
    "drums",
    "bass",
    "other",
    "vocals"
};

// htdemucs_6s (6 stems): drums, bass, other, vocals, guitar, piano
constexpr std::array<std::string_view, 6uz> stem_names_6 = {
    "drums",
    "bass",
    "other",
    "vocals",
    "guitar",
    "piano"
};

// Default to 4-stem model
constexpr auto num_stems = 4uz;
constexpr auto stem_names = stem_names_4;

// Get stem name by index (for default 4-stem model)
constexpr std::string_view stem_name(std::size_t index) {
    return index < num_stems ? stem_names[index] : "unknown";
}

// Get stem name by index with dynamic stem count
constexpr std::string_view stem_name(std::size_t index, std::size_t total_stems) {
    if (total_stems == 6)
        return index < stem_names_6.size() ? stem_names_6[index] : "unknown";
    return index < stem_names_4.size() ? stem_names_4[index] : "unknown";
}

// Find stem index by name
constexpr std::size_t stem_index(std::string_view name) {
    for (auto i = 0uz; i < num_stems; ++i)
        if (stem_names[i] == name)
            return i;
    return num_stems; // Invalid index
}

// Compile-time tests
static_assert(num_stems == 4);
static_assert(stem_names.size() == num_stems);
static_assert(stem_name(0) == "drums");
static_assert(stem_name(1) == "bass");
static_assert(stem_name(2) == "other");
static_assert(stem_name(3) == "vocals");
static_assert(stem_name(999) == "unknown");
static_assert(stem_index("drums") == 0);
static_assert(stem_index("vocals") == 3);
static_assert(stem_index("invalid") == num_stems);

} // namespace separation

} // namespace stems
