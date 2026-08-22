#pragma once
#include <cmath>
#include <vector>
#include "../../types.h"

namespace dsp::noise_reduction::amnr {

    inline float psdAlphaFor(double tauSeconds, double frameSeconds) {
        if (tauSeconds <= 0.0) { return 0.0f; }
        return (float)std::exp(-frameSeconds / tauSeconds);
    }

    // Per-bin noise PSD tracker for the AM noise reduction chain (design
    // section 7, step 2b).
    //
    // Follows the same minimum-guided philosophy as the squelch's scalar
    // McraEstimator, applied per frequency bin: a smoothed periodogram is
    // compared against a slowly tracked local minimum; bins whose energy sits
    // clearly above their minimum are marked speech-present and FROZEN, while
    // absent bins track the observation - downward quickly, upward slowly and
    // only while absent, exactly as the design prescribes. Upward targets are
    // additionally clipped relative to the running estimate so a single
    // impulse frame cannot drag the floor up behind it.
    //
    // The reliability flag drives automatic bypass: it requires warm-up, goes
    // stale after STALE_SECONDS without any noise-only frame, and trips an
    // occupancy-style guard when speech has been present more than
    // OCCUPANCY_LIMIT of the recent past even if no single gap was long -
    // the same ATIS/AWOS failure documented for the squelch estimator.
    class NoisePsdTracker {
    public:
        static constexpr double SMOOTH_TAU_SECONDS = 0.04;
        static constexpr double PRESENCE_TAU_SECONDS = 0.08;
        static constexpr float PRESENCE_RATIO = 5.0f;       // S/min, ~7 dB
        static constexpr double DOWN_TAU_SECONDS = 0.15;
        static constexpr double UP_TAU_SECONDS = 0.50;
        static constexpr float UPWARD_CLIP_FACTOR = 10.0f;   // target <= est*10
        static constexpr double MIN_WINDOW_SECONDS = 1.0;
        static constexpr double WARMUP_SECONDS = 1.0;
        static constexpr double STALE_SECONDS = 5.0;
        static constexpr double OCCUPANCY_LIMIT = 0.80;
        static constexpr double OCCUPANCY_TAU_SECONDS = 3.0;

        void init(int bins, double frameSeconds) {
            _bins = bins;
            _hopSec = frameSeconds;
            _aS = psdAlphaFor(SMOOTH_TAU_SECONDS, _hopSec);
            _aP = psdAlphaFor(PRESENCE_TAU_SECONDS, _hopSec);
            _aDown = psdAlphaFor(DOWN_TAU_SECONDS, _hopSec);
            _aUp = psdAlphaFor(UP_TAU_SECONDS, _hopSec);
            _aOcc = psdAlphaFor(OCCUPANCY_TAU_SECONDS, _hopSec);
            _halfWin = (int)std::lround(MIN_WINDOW_SECONDS / _hopSec / 2.0);
            if (_halfWin < 1) { _halfWin = 1; }
            _warmupFrames = (int)std::lround(WARMUP_SECONDS / _hopSec);
            if (_warmupFrames < 1) { _warmupFrames = 1; }

            _s.assign((size_t)_bins, 0.0f);
            _p.assign((size_t)_bins, 0.0f);
            _minRun.assign((size_t)_bins, 0.0f);
            _min.assign((size_t)_bins, 0.0f);
            _noise.assign((size_t)_bins, 0.0f);
            _binAbsent.assign((size_t)_bins, 0);
            reset();
        }

        void reset() {
            std::fill(_s.begin(), _s.end(), 0.0f);
            std::fill(_p.begin(), _p.end(), 0.0f);
            std::fill(_minRun.begin(), _minRun.end(), 0.0f);
            std::fill(_min.begin(), _min.end(), 0.0f);
            std::fill(_noise.begin(), _noise.end(), 0.0f);
                _halfCount = 0;
            std::fill(_binAbsent.begin(), _binAbsent.end(), 0);
            _frames = 0;
            _framesSinceEvidence = 0;
            _everEvidence = false;
            _occupancy = 0.0f;
            _started = false;
        }

        // Consumes one analysis frame's power spectrum (r2c layout) and
        // writes the frequency-smoothed noise PSD estimate.
        void process(const float* magSq, float* noisePsdOut) {
            // Bootstrap: the minimum tracker must start from the first real
            // observation, not from zero - a zero minimum would mark every
            // later frame as speech-present and freeze the tracker forever.
            if (!_started) {
                _started = true;
                float sum = 0.0f;
                for (int k = 0; k < _bins; k++) {
                    const float v = magSq[k] > 0.0f ? magSq[k] : 0.0f;
                    _s[(size_t)k] = v;
                    _min[(size_t)k] = v;
                    _minRun[(size_t)k] = v;
                    _noise[(size_t)k] = v;
                    _p[(size_t)k] = 0.0f;
                    sum += v;
                }
                _halfCount = 0;
                writeSmoothed(noisePsdOut);
                updateReliability(_bins);
                return;
            }

            // Smoothed periodogram and per-bin presence probability.
            int absentBins = 0;
            for (int k = 0; k < _bins; k++) {
                _s[(size_t)k] = _aS * _s[(size_t)k]
                              + (1.0f - _aS) * magSq[k];
                const bool rawPresent = _s[(size_t)k] > _min[(size_t)k] * PRESENCE_RATIO;
                _p[(size_t)k] = _aP * _p[(size_t)k]
                              + (1.0f - _aP) * (rawPresent ? 1.0f : 0.0f);
                if (!rawPresent) {
                    absentBins++;
                    _binAbsent[(size_t)k]++;
                }
            }

            // Two-half rolling minimum, shared window across bins. A narrow
            // band of speech leaves most bins at their floors, so the per-bin
            // minima keep tracking the noise even mid-transmission; only
            // genuinely broadband presence lifts them all.
            //
            // A committed minimum may RISE only when THAT BIN observed
            // genuine absence during the window - a bin that has been
            // continuously present cannot redefine its own floor, otherwise
            // a sustained carrier slowly becomes the floor and the staleness
            // guard loses its reference. Falls are never gated: floors must
            // always be able to drop.
            _halfCount++;
            if (_halfCount >= _halfWin) {
                const int riseFrames = _halfWin / 2;
                for (int k = 0; k < _bins; k++) {
                    float committed = _minRun[(size_t)k];
                    if (_binAbsent[(size_t)k] < riseFrames && committed > _min[(size_t)k]) {
                        committed = _min[(size_t)k];
                    }
                    _min[(size_t)k] = committed;
                    _minRun[(size_t)k] = _s[(size_t)k];
                    _binAbsent[(size_t)k] = 0;
                }
                _halfCount = 0;
            }
            for (int k = 0; k < _bins; k++) {
                if (_s[(size_t)k] < _minRun[(size_t)k]) { _minRun[(size_t)k] = _s[(size_t)k]; }
            }

            // Per-bin gated update. Present bins are FROZEN - the design
            // assigns gross broadband shifts to the reliability bypass, not
            // to floor chasing; absent bins track downward fast and upward
            // slowly, with the upward target clipped so impulses cannot lift
            // the floor behind them.
            for (int k = 0; k < _bins; k++) {
                if (_p[(size_t)k] >= 0.5f) { continue; }
                float target = _s[(size_t)k];
                if (_noise[(size_t)k] > 0.0f && target > _noise[(size_t)k] * UPWARD_CLIP_FACTOR) {
                    target = _noise[(size_t)k] * UPWARD_CLIP_FACTOR;
                }
                const float a = (target < _noise[(size_t)k]) ? _aDown : _aUp;
                const float updated = a * _noise[(size_t)k] + (1.0f - a) * target;
                _noise[(size_t)k] = (updated > 0.0f) ? updated : 0.0f;   // NaN guard
            }

            writeSmoothed(noisePsdOut);
            updateReliability(absentBins);
        }

        bool reliable() const {
            return _everEvidence
                && _frames >= _warmupFrames
                && (double)_framesSinceEvidence * _hopSec <= STALE_SECONDS
                && _occupancy <= (float)OCCUPANCY_LIMIT;
        }

        float occupancy() const { return _occupancy; }
    private:
        void writeSmoothed(float* noisePsdOut) {
            // Frequency smoothing: 3-tap weighted average with indices hard
            // clamped at both ends.
            for (int k = 0; k < _bins; k++) {
                const int left = (k > 0) ? k - 1 : 0;
                const int right = (k < _bins - 1) ? k + 1 : _bins - 1;
                noisePsdOut[k] = 0.25f * _noise[(size_t)left]
                               + 0.50f * _noise[(size_t)k]
                               + 0.25f * _noise[(size_t)right];
            }
        }

        void updateReliability(int absentBins) {
            _frames++;
            // A frame counts as absence evidence when most bins sit near
            // their own floors. This survives narrowband carriers (quiet
            // bins still observe the floor) and only fails when presence is
            // genuinely broadband - which is exactly when the floor stops
            // being observable.
            const bool evidence = absentBins * 2 >= _bins;
            if (evidence) {
                _framesSinceEvidence = 0;
                _everEvidence = true;
            } else if (_everEvidence) {
                _framesSinceEvidence++;
            }
            _occupancy = _aOcc * _occupancy
                       + (1.0f - _aOcc) * (evidence ? 0.0f : 1.0f);
        }

        int _bins = 0;
        double _hopSec = 256.0 / 48000.0;
        float _aS = 0.0f, _aP = 0.0f, _aDown = 0.0f, _aUp = 0.0f, _aOcc = 0.0f;
        int _halfWin = 1;
        int _warmupFrames = 1;
        std::vector<int> _binAbsent;

        std::vector<float> _s, _p, _minRun, _min, _noise;
        int _halfCount = 0;
        long long _frames = 0;
        long long _framesSinceEvidence = 0;
        bool _everEvidence = false;
        bool _started = false;
        float _occupancy = 0.0f;
    };
}
