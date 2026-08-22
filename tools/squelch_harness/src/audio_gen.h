#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include "signal_gen.h"

// Deterministic audio-band scenes for the AM chain tests: formant-synthesized
// "speech" inside ground-truth transmission and word labels, plus white
// Gaussian noise at a requested active-SNR. Realism is not the goal;
// repeatability and known ground truth are. Every run must produce
// byte-identical vectors or metric comparisons mean nothing.
//
// The synthesizer is deliberately simple: an impulse train with a wandering
// fundamental through three two-pole resonators (a Klatt-style digital
// resonator bank), shaped by word envelopes. It carries enough spectral
// structure for STOI-style correlation metrics and for human transcription
// in the listening protocol, without pretending to be real speech.
namespace ag {

    struct AudioSceneConfig {
        double rate = 48000.0;
        double durationSeconds = 60.0;
        float snrDb = 15.0f;              // active-speech power to noise power
        int wordsPerTx = 4;
        double txOnSeconds = 3.0;
        double txOffSeconds = 2.0;
        float f0BaseHz = 140.0f;
        uint32_t seed = 1u;
    };

    struct AudioScene {
        std::vector<float> clean;
        std::vector<float> noisy;
        std::vector<sg::Label> txLabels;
        std::vector<sg::Label> wordLabels;
        double rate = 0.0;
    };

    namespace agdetail {

        // Two-pole digital resonator (Klatt form): y[n] = x[n] + a1*y[n-1]
        // + a2*y[n-2]. The impulse response decays at bw and rings at f.
        // Output is scaled by (1-r) so the three formants land at comparable
        // loudness despite different bandwidths.
        struct Resonator {
            double a1 = 0.0, a2 = 0.0;
            double y1 = 0.0, y2 = 0.0;

            void set(double f, double bw, double fs) {
                const double r = std::exp(-3.14159265358979 * bw / fs);
                const double theta = 6.283185307179586 * f / fs;
                a1 = 2.0 * r * std::cos(theta);
                a2 = -r * r;
            }

            inline double run(double x) {
                const double y = x + a1 * y1 + a2 * y2;
                y2 = y1;
                y1 = y;
                return y * (1.0 - std::sqrt(-a2));
            }
        };

        struct FormantVoice {
            Resonator f1, f2, f3;

            void init(double fs) {
                f1.set(500.0, 90.0, fs);
                f2.set(1500.0, 110.0, fs);
                f3.set(2500.0, 140.0, fs);
            }

            // x is one excitation sample (impulses at pitch, zeros elsewhere).
            inline double run(double x) {
                return f1.run(x) + f2.run(x) + f3.run(x);
            }
        };
    }

    inline AudioScene generate(const AudioSceneConfig& cfg) {
        AudioScene s;
        s.rate = cfg.rate;
        const long n = (long)std::llround(cfg.durationSeconds * cfg.rate);
        s.clean.assign((size_t)n, 0.0f);

        sg::Rng rng(cfg.seed);

        // ---- transmission and word layout ---------------------------------
        const long txOn = (long)std::llround(cfg.txOnSeconds * cfg.rate);
        const long txOff = (long)std::llround(cfg.txOffSeconds * cfg.rate);
        const long period = txOn + txOff;
        for (long start = txOff; start + txOn <= n; start += period) {
            s.txLabels.push_back({ start, start + txOn });
        }

        for (auto& t : s.txLabels) {
            long cursor = t.start;
            long room = t.end - t.start;
            long pauseBudget = (long)((cfg.wordsPerTx - 1) * 0.12 * cfg.rate);
            for (int w = 0; w < cfg.wordsPerTx; w++) {
                const bool last = (w == cfg.wordsPerTx - 1);
                long len;
                if (last) {
                    len = t.end - cursor;
                }
                else {
                    len = (long)std::llround((0.25 + rng.uniform() * 0.30) * cfg.rate);
                    if ((long)(cursor - t.start) + len + pauseBudget > room) {
                        len = room - (cursor - t.start) - pauseBudget;
                    }
                    if (len < (long)(0.05 * cfg.rate)) { len = (long)(0.05 * cfg.rate); }
                }
                if (len <= 0 || cursor >= t.end) { break; }
                long end = cursor + len;
                if (end > t.end) { end = t.end; }
                s.wordLabels.push_back({ cursor, end });
                cursor = end;
                if (!last) {
                    cursor += (long)std::llround((0.06 + rng.uniform() * 0.09) * cfg.rate);
                    if (cursor > t.end) { cursor = t.end; }
                }
            }
        }

        // ---- voiced source -------------------------------------------------
        agdetail::FormantVoice voice;
        voice.init(cfg.rate);

        // Raised-cosine edge tables keep the ground truth click-free.
        const int edge = (int)std::llround(0.010 * cfg.rate);

        auto envelopeAt = [&](long i) -> float {
            for (auto& w : s.wordLabels) {
                if (i >= w.start && i < w.end) {
                    float env = 1.0f;
                    if (i - w.start < edge) {
                        env *= 0.5f - 0.5f * std::cos(3.14159265f * (float)(i - w.start) / (float)edge);
                    }
                    if (w.end - i <= edge) {
                        env *= 0.5f - 0.5f * std::cos(3.14159265f * (float)(w.end - i) / (float)edge);
                    }
                    return env;
                }
                if (i < w.start) { break; }
            }
            return 0.0f;
        };

        // Impulse train: phase accumulates in cycles; f0 wanders slowly with a
        // bounded random walk so consecutive runs differ but stay deterministic.
        double f0 = cfg.f0BaseHz;
        double phase = 0.0;
        double outScale = 0.0;

        // First pass: synthesize into the vector, tracking the peak.
        for (long i = 0; i < n; i++) {
            const double t = (double)i / cfg.rate;
            f0 += (rng.uniform() - 0.5) * 2.0;                 // random walk, +/-1 Hz per sample step is far too fast...
            // ...so pull it back toward base with a slow spring.
            f0 += (cfg.f0BaseHz + 20.0 * std::sin(6.283185307179586 * 0.7 * t) - f0) * 0.0005;
            if (f0 < 80.0) { f0 = 80.0; }
            if (f0 > 220.0) { f0 = 220.0; }

            phase += f0 / cfg.rate;
            double exc = 0.0;
            if (phase >= 1.0) { phase -= 1.0; exc = 1.0; }

            float env = envelopeAt(i);
            if (env != 0.0f) {
                double v = voice.run(exc) * (double)env;
                s.clean[(size_t)i] = (float)v;
                double a = v < 0 ? -v : v;
                if (a > outScale) { outScale = a; }
            }
            else {
                // Keep the resonator state warm so word onsets do not thump,
                // but discard the output during silence.
                voice.run(exc);
                s.clean[(size_t)i] = 0.0f;
            }
        }

        // Normalize peak to a fixed level so SNR math has a stable reference.
        if (!(outScale > 0.0)) { outScale = 1.0; }
        const float norm = (float)(0.35 / outScale);
        for (auto& v : s.clean) { v *= norm; }

        // ---- noise at the requested active SNR -----------------------------
        double speechPower = 0.0;
        long speechN = 0;
        for (auto& w : s.wordLabels) {
            for (long i = w.start; i < w.end && i < n; i++) {
                speechPower += (double)s.clean[(size_t)i] * s.clean[(size_t)i];
                speechN++;
            }
        }
        if (speechN == 0) { speechN = 1; }
        speechPower /= (double)speechN;

        const double snrLin = std::pow(10.0, cfg.snrDb / 10.0);
        const double noisePower = (speechPower > 0.0) ? speechPower / snrLin : 0.0;
        const float sigma = (float)std::sqrt(noisePower);

        s.noisy.resize((size_t)n);
        for (long i = 0; i < n; i++) {
            s.noisy[(size_t)i] = s.clean[(size_t)i] + sigma * rng.gaussian();
        }

        return s;
    }
}
