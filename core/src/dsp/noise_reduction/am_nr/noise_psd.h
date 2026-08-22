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
    // Presence-gated recursive averaging in the IMCRA family (Cohen and
    // Berdugo 2002; Cohen 2003): a smoothed periodogram is compared against
    // a slowly tracked per-bin reference; bins whose energy sits clearly
    // above that reference are marked speech-present and FROZEN, while
    // absent bins track the observation - downward quickly, upward slowly,
    // exactly as the design prescribes - with upward targets clipped
    // relative to the running estimate so an impulse cannot drag the floor
    // up behind it.
    //
    // The presence reference is a slow conditional mean: the average of each
    // bin's smoothed power over frames that themselves looked absent. A
    // minimum-based reference was tried first and rejected with evidence:
    // on chi-square(2) periodogram bins the tracked minimum lands near the
    // lower quantile unless it is inflated by a bias factor tied to the
    // window geometry, and any such threshold either freezes on phantom
    // speech or chases carriers once it is raised. An unbiased conditional
    // mean needs no compensation factor at all, and the harness carries a
    // chi-square convergence test to keep it honest.
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
        // Three times the conditional mean (~5 dB) sits above virtually all
        // chi-square(2) fluctuation of pure noise while real speech towers
        // far beyond it.
        static constexpr float PRESENCE_RATIO = 3.0f;
        static constexpr double PRESENCE_MEAN_TAU_SECONDS = 1.0;
        // Both directions share one absence-gated time constant. An
        // asymmetric pair (faster falls) was tried and rejected on evidence:
        // on a stationary chi-square input it biases the stationary estimate
        // well below the true mean, which the harness convergence test
        // measures. The design mandates a slow, absence-gated rise and a
        // freeze during presence; it does not ask for a faster fall.
        static constexpr double DOWN_TAU_SECONDS = 0.50;
        static constexpr double UP_TAU_SECONDS = 0.50;
        static constexpr float UPWARD_CLIP_FACTOR = 10.0f;   // target <= est*10
        static constexpr double WARMUP_SECONDS = 1.0;
        static constexpr double STALE_SECONDS = 5.0;
        static constexpr double OCCUPANCY_LIMIT = 0.80;
        static constexpr double OCCUPANCY_TAU_SECONDS = 3.0;
        // A bin whose reference trapped low classifies as present forever
        // and can never learn its way out through the gated path. After
        // this long without a single update it takes one unconditional
        // resync toward the observation; healthy bins never reach this.
        static constexpr double ESCAPE_SECONDS = 4.0;

        void init(int bins, double frameSeconds) {
            _bins = bins;
            _hopSec = frameSeconds;
            _aS = psdAlphaFor(SMOOTH_TAU_SECONDS, _hopSec);
            _aP = psdAlphaFor(PRESENCE_TAU_SECONDS, _hopSec);
            _aMean = psdAlphaFor(PRESENCE_MEAN_TAU_SECONDS, _hopSec);
            _aDown = psdAlphaFor(DOWN_TAU_SECONDS, _hopSec);
            _aUp = psdAlphaFor(UP_TAU_SECONDS, _hopSec);
            _aOcc = psdAlphaFor(OCCUPANCY_TAU_SECONDS, _hopSec);
            _warmupFrames = (int)std::lround(WARMUP_SECONDS / _hopSec);
            if (_warmupFrames < 1) { _warmupFrames = 1; }
            _escapeFrames = (int)std::lround(ESCAPE_SECONDS / _hopSec);
            if (_escapeFrames < 1) { _escapeFrames = 1; }

            _s.assign((size_t)_bins, 0.0f);
            _p.assign((size_t)_bins, 0.0f);
            _meanSlow.assign((size_t)_bins, 0.0f);
            _noise.assign((size_t)_bins, 0.0f);
            _sinceUpdate.assign((size_t)_bins, 0);
            reset();
        }

        void reset() {
            std::fill(_s.begin(), _s.end(), 0.0f);
            std::fill(_p.begin(), _p.end(), 0.0f);
            std::fill(_meanSlow.begin(), _meanSlow.end(), 0.0f);
            std::fill(_noise.begin(), _noise.end(), 0.0f);
            _frames = 0;
            _framesSinceEvidence = 0;
            _everEvidence = false;
            _occupancy = 0.0f;
            _started = false;
            std::fill(_sinceUpdate.begin(), _sinceUpdate.end(), 0);
        }

        // Consumes one analysis frame's power spectrum (r2c layout) and
        // writes the frequency-smoothed noise PSD estimate.
        void process(const float* magSq, float* noisePsdOut) {
            // Bootstrap from the first real observation: zero references
            // would classify every later frame as speech-present and freeze
            // the tracker forever.
            if (!_started) {
                _started = true;
                for (int k = 0; k < _bins; k++) {
                    const float v = magSq[k] > 0.0f ? magSq[k] : 0.0f;
                    _s[(size_t)k] = v;
                    _meanSlow[(size_t)k] = v;
                    _noise[(size_t)k] = v;
                    _p[(size_t)k] = 0.0f;
                }
                writeSmoothed(noisePsdOut);
                updateReliability(_bins);
                return;
            }

            // Smoothed periodogram, per-bin presence probability, and the
            // frame-level absence verdict. Classification runs against each
            // bin's slow conditional mean; the FRAME is absent when most
            // bins sit near their floors - which survives narrowband
            // carriers and fails only when presence is genuinely broadband,
            // i.e., exactly when the floor stops being observable.
            int absentBins = 0;
            for (int k = 0; k < _bins; k++) {
                _s[(size_t)k] = _aS * _s[(size_t)k]
                              + (1.0f - _aS) * magSq[k];
                const bool rawPresent = _s[(size_t)k] > _meanSlow[(size_t)k] * PRESENCE_RATIO;
                _p[(size_t)k] = _aP * _p[(size_t)k]
                              + (1.0f - _aP) * (rawPresent ? 1.0f : 0.0f);
                if (!rawPresent) { absentBins++; }
            }
            const bool frameAbsent = absentBins * 2 >= _bins;

            // Per-bin conditioned update, IMCRA-style: on absent FRAMES,
            // bins whose own presence probability has settled track the
            // observation - downward and upward share one absence-gated time
            // constant (an asymmetric pair was tried and rejected on
            // evidence: it biases the stationary estimate low on chi-square
            // input), with upward targets clipped so an impulse cannot lift
            // the floor behind it. Present bins are FROZEN, and so is the
            // conditional-mean reference that gates them: gross broadband
            // shifts belong to the reliability bypass, not to floor
            // chasing, and a continuously present bin must never redefine
            // its own floor. On present FRAMES nothing moves anywhere.
            if (frameAbsent) {
                for (int k = 0; k < _bins; k++) {
                    // Escape hatch: a bin locked present by an unlucky early
                    // reference cannot recover through the gated path, so it
                    // resyncs unconditionally after a long silence of
                    // updates. Healthy bins reset this counter constantly
                    // and never trigger it.
                    if (_sinceUpdate[(size_t)k] >= _escapeFrames) {
                        const float v = _s[(size_t)k] > 0.0f ? _s[(size_t)k] : 0.0f;
                        _noise[(size_t)k] = v;
                        _meanSlow[(size_t)k] = v;
                        _p[(size_t)k] *= _aP;
                        _sinceUpdate[(size_t)k] = 0;
                        continue;
                    }
                    if (_p[(size_t)k] >= 0.5f) {
                        _sinceUpdate[(size_t)k]++;
                        continue;
                    }
                    _sinceUpdate[(size_t)k] = 0;
                    float target = _s[(size_t)k];
                    if (_noise[(size_t)k] > 0.0f && target > _noise[(size_t)k] * UPWARD_CLIP_FACTOR) {
                        target = _noise[(size_t)k] * UPWARD_CLIP_FACTOR;
                    }
                    const float a = (target < _noise[(size_t)k]) ? _aDown : _aUp;
                    const float updated = a * _noise[(size_t)k] + (1.0f - a) * target;
                    _noise[(size_t)k] = (updated > 0.0f) ? updated : 0.0f;   // NaN guard

                    // The reference tracks the full observation, not just its
                    // quiet part: gating it by its own threshold would let it
                    // only ever learn values below three times itself, a
                    // self-truncating fixed point on heavy-tailed input. The
                    // presence-probability gate above is what keeps a
                    // continuously present bin from redefining its floor.
                    _meanSlow[(size_t)k] += (1.0f - _aMean)
                                          * (_s[(size_t)k] - _meanSlow[(size_t)k]);
                }
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

        // The smoothed periodogram the tracker itself runs on. The gain rule
        // must consume the same signal: raw per-frame powers of active speech
        // are chi-square distributed over enormous dynamic range, and letting
        // the enhancement react to them gates speech components on and off
        // within syllables.
        const std::vector<float>& smoothedPower() const { return _s; }

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
        float _aS = 0.0f, _aP = 0.0f, _aMean = 0.0f;
        float _aDown = 0.0f, _aUp = 0.0f, _aOcc = 0.0f;
        int _warmupFrames = 1;
        int _escapeFrames = 1;

        std::vector<float> _s, _p, _meanSlow, _noise;
        std::vector<int> _sinceUpdate;
        long long _frames = 0;
        long long _framesSinceEvidence = 0;
        bool _everEvidence = false;
        bool _started = false;
        float _occupancy = 0.0f;
    };
}
