#pragma once

#include <fftw3.h>
#include <expected>
#include <memory>
#include <span>
#include <vector>

namespace stems {

// STFT processing errors
enum class StftError {
    InvalidInput,
    AllocationFailed,
    PlanningFailed,
    InvalidWindowSize
};

// Convert StftError to human-readable string
constexpr std::string_view error_message(StftError error) {
    switch (error) {
        case StftError::InvalidInput:
            return "Invalid input data";
        case StftError::AllocationFailed:
            return "Memory allocation failed";
        case StftError::PlanningFailed:
            return "FFTW plan creation failed";
        case StftError::InvalidWindowSize:
            return "Invalid window size (must be power of 2)";
    }
    return "Unknown error";
}

// Compile-time tests
static_assert(error_message(StftError::InvalidInput) == "Invalid input data");
static_assert(error_message(StftError::PlanningFailed) == "FFTW plan creation failed");
static_assert(!error_message(StftError::InvalidWindowSize).empty());

// STFT parameters for Demucs
namespace stft_params {
    constexpr auto window_size = 4096uz;      // Standard for music separation
    constexpr auto hop_size = 1024uz;         // 75% overlap
    constexpr auto fft_size = window_size;
    constexpr auto num_bins = fft_size / 2 + 1; // For real FFT

    // Verify parameters at compile time
    static_assert(window_size > 0);
    static_assert(hop_size > 0);
    static_assert(hop_size <= window_size);
    static_assert((window_size & (window_size - 1)) == 0, "Window size must be power of 2");
}

// Complex spectrogram representation
// Each frame contains real and imaginary components
struct Spectrogram {
    std::vector<float> real;  // Real components
    std::vector<float> imag;  // Imaginary components
    std::size_t num_frames;
    std::size_t num_bins;
};

// RAII wrapper for FFTW plan
class FftwPlan {
public:
    explicit FftwPlan(fftwf_plan);
    ~FftwPlan();

    // Non-copyable
    FftwPlan(FftwPlan const&) = delete;
    FftwPlan& operator=(FftwPlan const&) = delete;

    // Movable
    FftwPlan(FftwPlan&&) noexcept;
    FftwPlan& operator=(FftwPlan&&) noexcept;

    void execute();
    fftwf_plan get() const { return plan_; }

private:
    fftwf_plan plan_;
};

// Short-Time Fourier Transform processor
class StftProcessor {
public:
    StftProcessor();
    ~StftProcessor() = default;

    // Forward transform: time domain -> frequency domain
    std::expected<Spectrogram, StftError> forward(std::span<float const>);

    // Inverse transform: frequency domain -> time domain
    std::expected<std::vector<float>, StftError> inverse(Spectrogram const&);

private:
    // Hann window for smooth transitions
    std::vector<float> window_;

    // Pre-compute window coefficients
    void compute_hann_window();
};

} // namespace stems
