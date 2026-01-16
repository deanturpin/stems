#include "stft.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <print>

namespace stems {

namespace {

// Check if input data is valid for processing
bool is_valid_input(std::span<float const> data) {
    return !data.empty() && data.size() >= stft_params::window_size;
}

// Compute Hann window coefficients
std::vector<float> make_hann_window(std::size_t size) {
    auto window = std::vector<float>(size);
    auto const pi = std::numbers::pi_v<float>;

    for (auto i = 0uz; i < size; ++i) {
        auto const n = static_cast<float>(i);
        auto const N = static_cast<float>(size);
        window[i] = 0.5f * (1.0f - std::cos(2.0f * pi * n / (N - 1.0f)));
    }

    return window;
}

// Calculate number of frames for STFT with center padding (Demucs style)
// Center padding: audio is padded on both sides so frames are centered on samples
std::size_t calculate_num_frames(std::size_t signal_length) {
    if (signal_length == 0)
        return 0uz;

    // With center padding: num_frames = signal_length / hop_size + 1
    // This matches PyTorch's torch.stft with center=True
    return signal_length / stft_params::hop_size + 1uz;
}

} // anonymous namespace

FftwPlan::FftwPlan(fftwf_plan plan) : plan_(plan) {}

FftwPlan::~FftwPlan() {
    if (plan_)
        fftwf_destroy_plan(plan_);
}

FftwPlan::FftwPlan(FftwPlan&& other) noexcept : plan_(other.plan_) {
    other.plan_ = nullptr;
}

FftwPlan& FftwPlan::operator=(FftwPlan&& other) noexcept {
    if (this != &other) {
        if (plan_)
            fftwf_destroy_plan(plan_);
        plan_ = other.plan_;
        other.plan_ = nullptr;
    }
    return *this;
}

void FftwPlan::execute() {
    fftwf_execute(plan_);
}

StftProcessor::StftProcessor() : window_(make_hann_window(stft_params::window_size)) {
    initialize_fftw();
}

StftProcessor::~StftProcessor() {
    if (forward_plan_)
        fftwf_destroy_plan(forward_plan_);
    if (fftw_input_)
        fftwf_free(fftw_input_);
    if (fftw_output_)
        fftwf_free(fftw_output_);
}

bool StftProcessor::initialize_fftw() {
    // Allocate FFTW buffers once
    fftw_input_ = fftwf_alloc_real(stft_params::fft_size);
    fftw_output_ = fftwf_alloc_complex(stft_params::num_bins);

    if (!fftw_input_ or !fftw_output_)
        return false;

    // Create FFTW plan once
    forward_plan_ = fftwf_plan_dft_r2c_1d(
        static_cast<int>(stft_params::fft_size),
        fftw_input_,
        fftw_output_,
        FFTW_MEASURE  // Use MEASURE for better performance
    );

    return forward_plan_ != nullptr;
}

void StftProcessor::compute_hann_window() {
    window_ = make_hann_window(stft_params::window_size);
}

std::expected<Spectrogram, StftError> StftProcessor::forward(std::span<float const> audio) {
    if (!is_valid_input(audio))
        return std::unexpected(StftError::InvalidInput);

    if (!forward_plan_ or !fftw_input_ or !fftw_output_)
        return std::unexpected(StftError::PlanningFailed);

    auto const num_frames = calculate_num_frames(audio.size());
    if (num_frames == 0uz)
        return std::unexpected(StftError::InvalidInput);

    // Allocate output buffers
    auto spec = Spectrogram{
        .real = std::vector<float>(num_frames * stft_params::num_bins),
        .imag = std::vector<float>(num_frames * stft_params::num_bins),
        .num_frames = num_frames,
        .num_bins = stft_params::num_bins
    };

    // Center padding: pad signal by window_size/2 on each side
    // This matches torch.stft(center=True) behavior
    auto constexpr pad_size = stft_params::window_size / 2;

    // Process each frame with center padding
    for (auto frame_idx = 0uz; frame_idx < num_frames; ++frame_idx) {
        auto const center_pos = frame_idx * stft_params::hop_size;
        auto const start_pos = center_pos > pad_size ? center_pos - pad_size : 0uz;

        // Apply window and copy to FFTW input buffer
        for (auto i = 0uz; i < stft_params::window_size; ++i) {
            // Calculate position in original audio (accounting for padding)
            auto const padded_pos = start_pos + i;
            auto const audio_idx = padded_pos >= pad_size ? padded_pos - pad_size : 0uz;

            // Zero-pad before start or after end
            if (padded_pos < pad_size or audio_idx >= audio.size()) {
                fftw_input_[i] = 0.0f;
            } else {
                fftw_input_[i] = audio[audio_idx] * window_[i];
            }
        }

        // Execute FFT using cached plan
        fftwf_execute(forward_plan_);

        // Copy complex results to spectrogram
        for (auto bin = 0uz; bin < stft_params::num_bins; ++bin) {
            auto const spec_idx = frame_idx * stft_params::num_bins + bin;
            spec.real[spec_idx] = fftw_output_[bin][0]; // Real part
            spec.imag[spec_idx] = fftw_output_[bin][1]; // Imaginary part
        }
    }

    std::println("STFT forward: {} samples -> {} frames x {} bins",
                 audio.size(), num_frames, stft_params::num_bins);

    return spec;
}

std::expected<std::vector<float>, StftError> StftProcessor::inverse(Spectrogram const& spec) {
    if (spec.num_frames == 0uz || spec.num_bins != stft_params::num_bins)
        return std::unexpected(StftError::InvalidInput);

    // Calculate output length
    auto const output_length = (spec.num_frames - 1uz) * stft_params::hop_size
                              + stft_params::window_size;
    auto output = std::vector<float>(output_length, 0.0f);
    auto window_sum = std::vector<float>(output_length, 0.0f);

    // Allocate FFTW buffers
    auto* input = fftwf_alloc_complex(stft_params::num_bins);
    auto* output_buf = fftwf_alloc_real(stft_params::fft_size);

    if (!input || !output_buf) {
        if (input) fftwf_free(input);
        if (output_buf) fftwf_free(output_buf);
        return std::unexpected(StftError::AllocationFailed);
    }

    // Create FFTW plan for complex-to-real transform
    auto* plan = fftwf_plan_dft_c2r_1d(
        static_cast<int>(stft_params::fft_size),
        input,
        output_buf,
        FFTW_ESTIMATE
    );

    if (!plan) {
        fftwf_free(input);
        fftwf_free(output_buf);
        return std::unexpected(StftError::PlanningFailed);
    }

    auto fftw_plan = FftwPlan{plan};

    // Process each frame
    for (auto frame_idx = 0uz; frame_idx < spec.num_frames; ++frame_idx) {
        // Copy complex spectrogram to FFTW input buffer
        for (auto bin = 0uz; bin < stft_params::num_bins; ++bin) {
            auto const spec_idx = frame_idx * stft_params::num_bins + bin;
            input[bin][0] = spec.real[spec_idx];
            input[bin][1] = spec.imag[spec_idx];
        }

        // Execute inverse FFT
        fftw_plan.execute();

        // Overlap-add with windowing
        auto const start_pos = frame_idx * stft_params::hop_size;
        auto const fft_scale = 1.0f / static_cast<float>(stft_params::fft_size);

        for (auto i = 0uz; i < stft_params::window_size; ++i) {
            auto const out_idx = start_pos + i;
            if (out_idx < output_length) {
                output[out_idx] += output_buf[i] * window_[i] * fft_scale;
                window_sum[out_idx] += window_[i] * window_[i];
            }
        }
    }

    // Normalise by window sum to avoid amplitude modulation
    for (auto i = 0uz; i < output_length; ++i)
        if (window_sum[i] > 1e-8f)
            output[i] /= window_sum[i];

    fftwf_free(input);
    fftwf_free(output_buf);

    std::println("STFT inverse: {} frames x {} bins -> {} samples",
                 spec.num_frames, spec.num_bins, output.size());

    return output;
}

} // namespace stems
