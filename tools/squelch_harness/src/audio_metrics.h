#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include "signal_gen.h"
#include "wav_io.h"

// Audio-chain scoring for the AM voice bandpass and noise reduction blocks.
// These metrics are computed against ground truth (the clean scene and its
// labels), with block latency removed using the delay constants the blocks
// themselves report - never by detecting alignment from the signals.
namespace metrics {

    inline double medianOf(std::vector<double> v) {
        if (v.empty()) { return 0.0; }
        size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        return v[mid];
    }

    // Median power change, in dB, inside the noise-only regions: how much of
    // the hiss the chain removes. Negative means reduction. Segment edges are
    // excluded so hang/overhang behaviour does not contaminate the figure.
    // gapMarginSeconds trims both ends of every gap.
    inline double noiseReductionDb(const std::vector<float>& in,
                                   const std::vector<float>& out,
                                   const std::vector<sg::Label>& txLabels,
                                   double rate,
                                   double gapMarginSeconds) {
        std::vector<char> active((size_t)std::min(in.size(), out.size()), 0);
        for (auto& l : txLabels) {
            for (long i = l.start; i < l.end && i < (long)active.size(); i++) {
                active[(size_t)i] = 1;
            }
        }

        // Collect contiguous gap intervals, trimmed at both ends.
        std::vector<sg::Label> gaps;
        long i = 0;
        const long m = (long)active.size();
        while (i < m) {
            if (active[(size_t)i]) { i++; continue; }
            long start = i;
            while (i < m && !active[(size_t)i]) { i++; }
            gaps.push_back({ start, i });
        }

        const long margin = (long)std::llround(gapMarginSeconds * rate);
        const long segLen = (long)std::llround(0.020 * rate);
        std::vector<double> segDb;
        for (auto& g : gaps) {
            long a = g.start + margin;
            long b = g.end - margin;
            for (long s = a; s + segLen <= b; s += segLen) {
                double pi = 0.0, po = 0.0;
                for (long j = s; j < s + segLen; j++) {
                    pi += (double)in[(size_t)j] * in[(size_t)j];
                    po += (double)out[(size_t)j] * out[(size_t)j];
                }
                if (pi > 1e-12) {
                    segDb.push_back(10.0 * std::log10(po / pi));
                }
            }
        }
        return medianOf(segDb);
    }

    // Median segmental power delta between the clean ground truth and the
    // processed output inside transmissions, after removing the chain's
    // reported latency: how much speech-band energy survives. A value near 0
    // means the chain passes speech essentially untouched.
    inline double inBandFidelityDb(const std::vector<float>& clean,
                                   const std::vector<float>& out,
                                   const std::vector<sg::Label>& txLabels,
                                   int delaySamples) {
        const long n = (long)std::min(clean.size(), out.size());
        const long segLen = 960;   // 20 ms at the canonical 48 kHz test rate
        std::vector<double> segDb;
        for (auto& l : txLabels) {
            for (long s = l.start; s + segLen <= l.end; s += segLen) {
                double pc = 0.0, po = 0.0;
                bool ok = true;
                for (long j = s; j < s + segLen; j++) {
                    const long o = j + delaySamples;
                    if (o >= n || o < 0) { ok = false; break; }
                    pc += (double)clean[(size_t)j] * clean[(size_t)j];
                    po += (double)out[(size_t)o] * out[(size_t)o];
                }
                if (ok && pc > 1e-12) {
                    segDb.push_back(10.0 * std::log10(po / pc));
                }
            }
        }
        return medianOf(segDb);
    }

    // Largest single-sample magnitude change. A guard metric: enhancement is
    // not allowed to introduce clicks that were not in the input.
    inline float maxSampleJump(const std::vector<float>& v) {
        float worst = 0.0f;
        for (size_t i = 1; i < v.size(); i++) {
            float d = std::fabs(v[i] - v[i - 1]);
            if (d > worst) { worst = d; }
        }
        return worst;
    }

    // Writes clean reference, degraded input and processed output as f32 WAVs
    // sharing one base path. Consumed by stoi_check.py and the listening
    // protocol.
    inline bool exportWavTriplet(const std::string& basePath,
                                 double rate,
                                 const std::vector<float>& refClean,
                                 const std::vector<float>& degNoisy,
                                 const std::vector<float>& procOut) {
        return ag::writeFloat32((basePath + "_ref.wav").c_str(), refClean, (int)rate)
            && ag::writeFloat32((basePath + "_deg.wav").c_str(), degNoisy, (int)rate)
            && ag::writeFloat32((basePath + "_proc.wav").c_str(), procOut, (int)rate);
    }
}
