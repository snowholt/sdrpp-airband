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
#include <dsp/noise_reduction/am_nr.h>

using dsp::audio::VoiceBandpass;
using dsp::noise_reduction::AmNR;

// AM audio-chain evaluation sweep (plan task P2-13).
//
// Runs formant-speech scenes at several SNRs through three AF-chain
// conditions - unprocessed, voice bandpass, and bandpass plus capped log-MMSE
// noise reduction - and reports the noise-only reduction, delay-compensated
// in-band speech fidelity, transient jump guard and CPU cost per condition.
// WAV triplets and an STOI batch manifest are written for the release-gate
// scoring run performed immediately afterwards by stoi_check.py.
//
// Release build only: the ns/sample column is meaningless otherwise.

namespace {

    struct Condition {
        const char* name;
        bool bpEnabled;
        bool nrEnabled;
    };

    const Condition CONDITIONS[] = {
        { "off",   false, false },
        { "bp",    true,  false },
        { "bp_nr", true,  true  },
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

        std::vector<dsp::stereo_t> inStereo((size_t)n);
        for (long i = 0; i < n; i++) {
            inStereo[(size_t)i].l = scene.noisy[(size_t)i];
            inStereo[(size_t)i].r = scene.noisy[(size_t)i];
        }

        // Displacement versus availability: the bandpass delays the
        // waveform by its group delay; the STFT-based NR is time-aligned
        // (its latencySamples() is an availability horizon only) and must
        // NOT be added here. Correlation scoring compares waveforms.
        int delay = 0;

        // Pass 1: voice bandpass.
        std::vector<dsp::stereo_t> afterBp((size_t)n);
        sg::Rng chunkRng(4242);
        const auto t0 = std::chrono::steady_clock::now();
        if (cond.bpEnabled) {
            VoiceBandpass bp;
            bp.init(NULL, scene.rate);
            delay += bp.delaySamples();
            long pos = 0;
            while (pos < n) {
                int take = 1 + (int)(chunkRng.uniform() * 8192.0f);
                if (pos + take > n) { take = (int)(n - pos); }
                bp.process(take, inStereo.data() + pos, afterBp.data() + pos);
                pos += take;
            }
        } else {
            afterBp = inStereo;
        }

        // Pass 2: noise reduction. The block emits what is final per call,
        // so samples are appended as they arrive.
        std::vector<dsp::stereo_t> outStereo((size_t)n);
        size_t produced = 0;
        if (cond.nrEnabled) {
            AmNR nr;
            nr.init(NULL, scene.rate);
            nr.setEnabled(true);
            std::vector<dsp::stereo_t> scratch((size_t)2048 + 8192);
            long pos = 0;
            while (pos < n) {
                int take = 2048;
                if (pos + take > n) { take = (int)(n - pos); }
                const int emitted = nr.process(take, afterBp.data() + pos, scratch.data());
                for (int i = 0; i < emitted && produced < (size_t)n; i++) {
                    outStereo[produced++] = scratch[(size_t)i];
                }
                pos += take;
            }
        } else {
            outStereo = afterBp;
            produced = (size_t)n;
        }
        const auto t1 = std::chrono::steady_clock::now();

        out.assign((size_t)n, 0.0f);
        const long have = (long)std::min(produced, (size_t)n);
        for (long i = 0; i < have; i++) {
            out[(size_t)i] = outStereo[(size_t)i].l;
        }

        RunResult r;
        r.noiseReductionDb = metrics::noiseReductionDb(scene.noisy, out, scene.txLabels,
                                                       scene.rate, 0.05);
        r.fidelityDb = metrics::inBandFidelityDb(scene.clean, out, scene.txLabels, delay);
        r.maxJump = metrics::maxSampleJump(out);
        r.delaySamples = delay;
        const double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        r.nsPerSample = ns / (double)n;
        return r;
    }

}  // namespace

int main(int argc, char** argv) {
    const char* exportDir = "/tmp";
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--export-dir") && i + 1 < argc) { exportDir = argv[++i]; }
    }

    const double snrPoints[] = { 20.0, 15.0, 10.0, 6.0, 0.0 };

    std::string manifestPath = std::string(exportDir) + "/phase2_nr_manifest.csv";
    std::FILE* manifest = std::fopen(manifestPath.c_str(), "w");
    if (manifest) {
        std::fprintf(manifest, "tag,ref,test,base1_name,base1_path,base2_name,base2_path\n");
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

            if (manifest) {
                char base[512];
                std::snprintf(base, sizeof(base), "%s/phase2_nr_%s_snr%.0f",
                              exportDir, cond.name, snr);

                // Align on the chain's reported latency before exporting:
                // correlation scoring must compare like against like.
                std::vector<float> aligned(std::max(0, (int)out.size() - r.delaySamples), 0.0f);
                aligned.resize(scene.clean.size(), 0.0f);
                for (long i = 0; i + r.delaySamples < (long)out.size(); i++) {
                    aligned[(size_t)i] = out[(size_t)(i + r.delaySamples)];
                }

                metrics::exportWavTriplet(base, scene.rate, scene.clean, scene.noisy, aligned);

                char refPath[600], procPath[640], offPath[640], bpPath[640];
                std::snprintf(refPath, sizeof(refPath), "%s_ref.wav", base);
                std::snprintf(procPath, sizeof(procPath), "%s_proc.wav", base);
                std::snprintf(offPath, sizeof(offPath), "%s/phase2_nr_off_snr%.0f_proc.wav",
                              exportDir, snr);
                std::snprintf(bpPath, sizeof(bpPath), "%s/phase2_nr_bp_snr%.0f_proc.wav",
                              exportDir, snr);
                if (cond.bpEnabled && !cond.nrEnabled) {
                    std::fprintf(manifest, "nr_snr%.0f_bp,%s,%s,noisy,%s,,\n",
                                 snr, refPath, procPath, offPath);
                }
                if (cond.nrEnabled) {
                    std::fprintf(manifest, "nr_snr%.0f_bp_nr,%s,%s,bp,%s,noisy,%s\n",
                                 snr, refPath, procPath, bpPath, offPath);
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
