#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dsp::noise_reduction::adaptive {

    enum EstimatorType {
        ESTIMATOR_PERCENTILE,
        ESTIMATOR_MCRA,
        ESTIMATOR_GATED_EMA
    };

    // Estimates the noise floor of a channel over time.
    //
    // The stream reaching this code has already been frequency-shifted,
    // resampled and low-pass filtered to the channel bandwidth
    // (radio_module.h:88, rx_vfo.h:24), so everything outside the channel is
    // stopband and there is no adjacent-channel noise reference. The floor must
    // therefore be estimated across time, not across frequency.
    class NoiseEstimator {
    public:
        virtual ~NoiseEstimator() {}

        // frameSeconds is the fixed frame duration produced by FrameMeter.
        // Every time constant in an implementation is derived from it.
        virtual void init(double frameSeconds) = 0;
        virtual void reset() = 0;

        // Consumes one frame and returns the current noise floor estimate in dB.
        virtual float update(float framePowerDb, bool gateOpen) = 0;

        // Fraction of the recent window sitting clearly above the estimate.
        // Above roughly 0.8 the estimate is no longer trustworthy: there is not
        // enough noise-only time in the window. RTLSDR-Airband documents this
        // exact failure for continuous ATIS/AWOS channels.
        virtual float occupancy() const { return 0.0f; }

        virtual const char* name() const = 0;
    };

    // ---------------------------------------------------------------------
    // P - rolling dB histogram percentile. The default.
    //
    // Deterministic memory, exact window expiration, no FFT, robust to
    // impulses, and no distribution-specific bias table to get wrong. The
    // percentile encodes an occupancy assumption: a q-quantile stays
    // noise-representative only while at least q of the window is noise-only,
    // so the 20th percentile tolerates roughly 80% channel occupancy.
    // ---------------------------------------------------------------------
    class PercentileEstimator : public NoiseEstimator {
    public:
        static constexpr float MIN_DB = -140.0f;
        static constexpr float MAX_DB = 0.0f;
        static constexpr float BIN_WIDTH = 0.25f;
        static constexpr int BIN_COUNT = (int)((MAX_DB - MIN_DB) / BIN_WIDTH);   // 560
        static constexpr float OCCUPANCY_MARGIN_DB = 6.0f;

        void init(double frameSeconds) override {
            _frameSeconds = frameSeconds;
            rebuild();
        }

        void setWindow(double seconds) {
            _windowSeconds = seconds;
            rebuild();
        }

        void setPercentile(float p) {
            _percentile = std::clamp(p, 0.01f, 0.99f);
        }

        void reset() override {
            std::fill(_bins.begin(), _bins.end(), 0);
            std::fill(_ring.begin(), _ring.end(), (uint16_t)0);
            _head = 0;
            _filled = 0;
            _lastEstimate = MIN_DB;
        }

        // Maps a dB value onto a bin index, hard-clamped at both ends.
        // The leading !(db > MIN_DB) also catches NaN, which would otherwise
        // fall through every comparison and produce an out-of-range index.
        static int binOf(float db) {
            if (!(db > MIN_DB)) { return 0; }
            if (db >= MAX_DB) { return BIN_COUNT - 1; }
            int idx = (int)((db - MIN_DB) / BIN_WIDTH);
            if (idx < 0) { idx = 0; }
            if (idx >= BIN_COUNT) { idx = BIN_COUNT - 1; }
            return idx;
        }

        float update(float framePowerDb, bool /*gateOpen*/) override {
            const int idx = binOf(framePowerDb);

            if (_filled == _windowFrames) {
                const uint16_t old = _ring[_head];
                if (_bins[old] > 0) { _bins[old]--; }
            }
            else {
                _filled++;
            }
            _ring[_head] = (uint16_t)idx;
            _head++;
            if (_head >= _windowFrames) { _head = 0; }
            _bins[idx]++;

            int target = (int)std::lround((double)_percentile * (double)_filled);
            if (target < 1) { target = 1; }

            int cum = 0;
            for (int b = 0; b < BIN_COUNT; b++) {
                cum += _bins[b];
                if (cum >= target) {
                    _lastEstimate = MIN_DB + ((float)b + 0.5f) * BIN_WIDTH;
                    break;
                }
            }
            return _lastEstimate;
        }

        float occupancy() const override {
            if (_filled == 0) { return 0.0f; }
            const int thr = binOf(_lastEstimate + OCCUPANCY_MARGIN_DB);
            int above = 0;
            for (int b = thr; b < BIN_COUNT; b++) { above += _bins[b]; }
            return (float)above / (float)_filled;
        }

        const char* name() const override { return "percentile"; }

    private:
        void rebuild() {
            _windowFrames = (int)std::lround(_windowSeconds / _frameSeconds);
            if (_windowFrames < 1) { _windowFrames = 1; }
            _ring.assign((size_t)_windowFrames, (uint16_t)0);
            _bins.assign((size_t)BIN_COUNT, 0);
            reset();
        }

        double _frameSeconds = 0.010;
        double _windowSeconds = 10.0;
        float _percentile = 0.20f;
        int _windowFrames = 1000;
        std::vector<uint16_t> _ring;
        std::vector<int32_t> _bins;
        int _head = 0;
        int _filled = 0;
        float _lastEstimate = MIN_DB;
    };
}
