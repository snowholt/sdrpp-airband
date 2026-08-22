#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "audio_gen.h"
#include "audio_metrics.h"
#include "wav_io.h"
#include <dsp/audio/voice_bandpass.h>

using dsp::audio::VoiceBandpass;

// AM voice bandpass evaluation sweep (plan task P2-7).
//
// Runs formant-speech scenes at several SNRs through four AF-chain conditions
// - unprocessed, and the bandpass with 2500, 2700 and 3000 Hz upper cutoffs -
// and reports the noise-only reduction, delay-compensated in-band speech
// fidelity, transient jump guard and CPU cost per condition. WAV triplets and
// an STOI batch manifest are written for the intelligibility gate that is run
// immediately afterwards by stoi_check.py.
//
// Release build only: the ns/sample column is meaningless otherwise.

namespace {

    struct Condition {
        const char* name;
        bool enabled;
        double highHz;      // only used when enabled
    };

    const Condition CONDITIONS[] = {
        { "off",      false, 0.0    },
        { "bp_2500",  true,  2500.0 },
        { "bp_2700",  true,  2700.0 },
        { "bp_3000",  true,  3000.0 },
    };

    struct RunResult {
        double noiseReductionDb = 0.0;
        double fidelityDb = 0.0;
        float maxJump = 0.0f;
        double nsPerSample = 0.0;
        int delaySamples = 0;
    };

    RunResult runOne(const ag::AudioScene& scene,
                     const Condition& cond,
                     std::vector<float>& out) {
        const long n = (long)scene.noisy.size();
        out.assign((size_t)n, 0.0f);

        // The chain blocks are stereo; the scenes are mono. Duplicate into
        // both channels explicitly instead of aliasing the float buffer.
        std::vector<dsp::stereo_t> inStereo((size_t)n);
        for (long i = 0; i < n; i++) {
            inStereo[(size_t)i].l = scene.noisy[(size_t)i];
            inStereo[(size_t)i].r = scene.noisy[(size_t)i];
        }
        std::vector<dsp::stereo_t> outStereo((size_t)n);

        int delay = 0;
        VoiceBandpass bp;

        if (cond.enabled) {
            bp.init(NULL, scene.rate);
            if (cond.highHz != VoiceBandpass::DEFAULT_HIGH_HZ) {
                bp.setCutoffs(VoiceBandpass::DEFAULT_LOW_HZ, cond.highHz);
            }
            delay = bp.delaySamples();
        }

        sg::Rng chunkRng(4242);
        long pos = 0;
        const auto t0 = std::chrono::steady_clock::now();
        while (pos < n) {
            int take = 1 + (int)(chunkRng.uniform() * 8192.0f);
            if (pos + take > n) { take = (int)(n - pos); }
            if (cond.enabled) {
                bp.process(take, inStereo.data() + pos, outStereo.data() + pos);
            } else {
                std::copy(inStereo.begin() + pos, inStereo.begin() + pos + take,
                          outStereo.begin() + pos);
            }
            pos += take;
        }
        const auto t1 = std::chrono::steady_clock::now();

        for (long i = 0; i < n; i++) {
            out[(size_t)i] = outStereo[(size_t)i].l;
        }

        RunResult r;
        r.noiseReductionDb = metrics::noiseReductionDb(scene.noisy, out, scene.txLabels,
                                                       scene.rate, 0.05);
        r.fidelityDb = metrics::inBandFidelityDb(scene.clean, out, scene.txLabels, delay);
        r.maxJump = metrics::maxSampleJump(out);
        r.delaySamples = delay;
        double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        r.nsPerSample = ns / (double)n;
        return r;
    }

}  // namespace

int main(int argc, char** argv) {
    const char* exportDir = "/tmp";
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--export-dir") && i + 1 < argc) { exportDir = argv[++i]; }
    }

    const double snrPoints[] = { 20.0, 15.0, 10.0, 6.0 };

    std::FILE* manifest = NULL;
    std::string manifestPath = std::string(exportDir) + "/phase2_bp_manifest.csv";
    manifest = std::fopen(manifestPath.c_str(), "w");
    if (manifest) {
        std::fprintf(manifest, "tag,ref,test,base1_name,base1_path\n");
    }

    std::printf("%-9s %5s %12s %12s %8s %11s\n",
                "condition", "SNR", "noiseReddB", "fidelitydB", "maxJump", "ns/sample");
    std::printf("------------------------------------------------------------------\n");

    for (double snr : snrPoints) {
        ag::AudioSceneConfig cfg;
        cfg.durationSeconds = 20.0;
        cfg.snrDb = (float)snr;
        cfg.wordsPerTx = 4;
        cfg.seed = 7u;
        ag::AudioScene scene = ag::generate(cfg);

        for (auto& cond : CONDITIONS) {
            std::vector<float> out;
            RunResult r = runOne(scene, cond, out);

            std::printf("%-9s %5.1f %12.2f %12.2f %8.4f %11.1f\n",
                        cond.name, snr, r.noiseReductionDb, r.fidelityDb,
                        r.maxJump, r.nsPerSample);

            // Export time-aligned audio: the bandpass shifts the stream by
            // its reported group delay, and correlation metrics must compare
            // like against like.
            std::vector<float> aligned(out.begin() + r.delaySamples, out.end());
            aligned.resize(scene.clean.size(), 0.0f);

            if (manifest) {
                char base[512];
                std::snprintf(base, sizeof(base), "%s/phase2_bp_%s_snr%.0f",
                              exportDir, cond.name, snr);
                metrics::exportWavTriplet(base, scene.rate, scene.clean, scene.noisy, aligned);
                // Every bandpass variant is gated against the unprocessed
                // baseline of the same scene.
                char refPath[600], procPath[640], basePath[640];
                std::snprintf(refPath, sizeof(refPath), "%s_ref.wav", base);
                std::snprintf(procPath, sizeof(procPath), "%s_proc.wav", base);
                std::snprintf(basePath, sizeof(basePath), "%s/phase2_bp_off_snr%.0f_proc.wav",
                              exportDir, snr);
                if (cond.enabled) {
                    std::fprintf(manifest, "bp_snr%.0f_%s,%s,%s,noisy,%s\n",
                                 snr, cond.name, refPath, procPath, basePath);
                }
            }
        }
        std::printf("\n");
    }

    if (manifest) {
        std::fclose(manifest);
        std::printf("STOI manifest written to %s\n", manifestPath.c_str());
    }
    return 0;
}
