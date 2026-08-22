#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include <dsp/types.h>

// M_PI is not guaranteed by the standard and is absent under MSVC without
// _USE_MATH_DEFINES, so it is spelled out here rather than assumed.
#define SG_TWO_PI 6.283185307179586

// Deterministic synthetic scenes with ground-truth transmission labels.
// Realism is not the goal; repeatability is. Every sweep run must see
// byte-identical input or estimator comparisons mean nothing.
namespace sg {

    // mulberry32. Small, fast, and identical on every platform and compiler,
    // which std::mt19937 distributions are not.
    class Rng {
    public:
        explicit Rng(uint32_t seed) : _s(seed) {}

        uint32_t next() {
            _s += 0x6D2B79F5u;
            uint32_t t = _s;
            t = (t ^ (t >> 15)) * (t | 1u);
            t ^= t + (t ^ (t >> 7)) * (t | 61u);
            return t ^ (t >> 14);
        }

        float uniform() { return (float)(next() >> 8) / 16777216.0f; }

        // Box-Muller. One value per call; the spare is cached.
        float gaussian() {
            if (_hasSpare) { _hasSpare = false; return _spare; }
            float u1 = uniform();
            if (u1 < 1e-7f) { u1 = 1e-7f; }
            float u2 = uniform();
            float mag = std::sqrt(-2.0f * std::log(u1));
            _spare = mag * std::sin(6.2831853f * u2);
            _hasSpare = true;
            return mag * std::cos(6.2831853f * u2);
        }

    private:
        uint32_t _s;
        bool _hasSpare = false;
        float _spare = 0.0f;
    };

    struct Label {
        long start;   // first sample of the transmission, inclusive
        long end;     // one past the last sample
    };

    struct SceneConfig {
        double samplerate = 15000.0;
        double durationSeconds = 60.0;
        float noiseFloorDbfs = -70.0f;   // mean power of the noise-only stream
        float cnrDb = 10.0f;             // peak transmission power above the floor
        double onSeconds = 3.0;
        double offSeconds = 2.0;
        double syllabicHz = 3.5;         // envelope modulation inside a burst
        uint32_t seed = 1u;
    };

    struct Scene {
        std::vector<dsp::complex_t> iq;
        std::vector<Label> labels;
        double samplerate = 0.0;
    };

    inline Scene generate(const SceneConfig& cfg) {
        Scene s;
        s.samplerate = cfg.samplerate;
        const long n = (long)std::llround(cfg.durationSeconds * cfg.samplerate);
        s.iq.resize((size_t)n);

        // Noise amplitude per component. Total mean power of re + im is
        // 2 * sigma^2, so sigma = sqrt(P / 2).
        const double noisePower = std::pow(10.0, cfg.noiseFloorDbfs / 10.0);
        const float sigma = (float)std::sqrt(noisePower / 2.0);
        const double sigPeakPower = noisePower * std::pow(10.0, cfg.cnrDb / 10.0);
        const float sigPeakAmp = (float)std::sqrt(sigPeakPower);

        const long onLen = (long)std::llround(cfg.onSeconds * cfg.samplerate);
        const long offLen = (long)std::llround(cfg.offSeconds * cfg.samplerate);
        const long period = onLen + offLen;

        // Start with silence so an estimator has something to converge on.
        for (long start = offLen; start + onLen <= n; start += period) {
            s.labels.push_back({ start, start + onLen });
        }

        Rng rng(cfg.seed);
        size_t labelIdx = 0;
        for (long i = 0; i < n; i++) {
            float re = sigma * rng.gaussian();
            float im = sigma * rng.gaussian();

            while (labelIdx < s.labels.size() && i >= s.labels[labelIdx].end) { labelIdx++; }
            if (labelIdx < s.labels.size() && i >= s.labels[labelIdx].start) {
                const long into = i - s.labels[labelIdx].start;
                const double t = (double)into / cfg.samplerate;
                // Syllabic envelope: never quite zero, so the burst is one
                // transmission rather than a train of short ones.
                const double env = 0.25 + 0.75 * 0.5 * (1.0 - std::cos(SG_TWO_PI * cfg.syllabicHz * t));
                const float amp = sigPeakAmp * (float)env;
                const double phase = SG_TWO_PI * 1200.0 * t;
                re += amp * (float)std::cos(phase);
                im += amp * (float)std::sin(phase);
            }
            s.iq[(size_t)i] = dsp::complex_t{ re, im };
        }
        return s;
    }
}
