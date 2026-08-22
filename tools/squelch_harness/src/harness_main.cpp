#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "signal_gen.h"
#include "scorer.h"
#include <dsp/noise_reduction/adaptive_squelch.h>
#include <dsp/noise_reduction/adaptive_squelch/gate.h>

using dsp::noise_reduction::AdaptiveSquelch;
namespace ad = dsp::noise_reduction::adaptive;

namespace {

    struct RunResult {
        scorer::Score score;
        double nsPerSample = 0.0;
    };

    // Feeds a scene through the block in uneven chunks, recording the gate
    // state for every output sample and optionally writing a frame trace.
    RunResult runOne(const sg::Scene& scene,
                     ad::EstimatorType type,
                     const char* csvPath) {
        AdaptiveSquelch sq;
        sq.init(NULL, scene.samplerate);
        sq.setEstimator(type);

        std::vector<char> gateOpen;
        gateOpen.reserve(scene.iq.size());
        std::vector<dsp::complex_t> scratch(65536);

        std::FILE* csv = NULL;
        if (csvPath) {
            csv = std::fopen(csvPath, "w");
            if (csv) {
                std::fprintf(csv, "t,framePowerDb,noiseFloorDb,snrDb,occupancy,state,reason\n");
            }
        }

        sg::Rng chunkRng(4242);
        long pos = 0;
        double lastTraceT = -1.0;
        const auto t0 = std::chrono::steady_clock::now();

        while (pos < (long)scene.iq.size()) {
            int take = 1 + (int)(chunkRng.uniform() * 8192.0f);
            if (pos + take > (long)scene.iq.size()) { take = (int)((long)scene.iq.size() - pos); }
            const int n = sq.process(take, scene.iq.data() + pos, scratch.data());

            const ad::Diagnostics d = sq.diagnostics();
            const bool open = (d.state == (int)ad::GATE_OPEN || d.state == (int)ad::GATE_HANG);
            for (int i = 0; i < n; i++) { gateOpen.push_back(open ? 1 : 0); }

            if (csv) {
                const double t = (double)pos / scene.samplerate;
                if (t - lastTraceT >= 0.010) {
                    lastTraceT = t;
                    std::fprintf(csv, "%.4f,%.2f,%.2f,%.2f,%.3f,%s,%s\n",
                                 t, d.framePowerDb, d.noiseFloorDb, d.snrDb, d.occupancy,
                                 ad::stateName((ad::GateState)d.state),
                                 ad::reasonName((ad::TransitionReason)d.lastTransitionReason));
                }
            }
            pos += take;
        }
        const auto t1 = std::chrono::steady_clock::now();
        if (csv) { std::fclose(csv); }

        RunResult r;
        r.score = scorer::score(gateOpen, scene.labels, scene.samplerate);
        const double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        r.nsPerSample = ns / (double)scene.iq.size();
        return r;
    }
}

int main(int argc, char** argv) {
    const char* csvPath = NULL;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--trace") && i + 1 < argc) { csvPath = argv[++i]; }
    }

    const struct { const char* name; ad::EstimatorType type; } estimators[] = {
        { "percentile", ad::ESTIMATOR_PERCENTILE },
        { "mcra",       ad::ESTIMATOR_MCRA       },
        { "gated_ema",  ad::ESTIMATOR_GATED_EMA  }
    };
    const float cnrPoints[] = { 30.0f, 20.0f, 15.0f, 10.0f, 6.0f, 3.0f };

    std::printf("%-12s %6s %12s %10s %10s %10s %7s %8s %10s\n",
                "estimator", "CNR", "falseOpen%", "miss%", "FEC ms", "OVER ms",
                "MSC", "chatter", "ns/sample");
    std::printf("--------------------------------------------------------"
                "----------------------------------------\n");

    for (auto& e : estimators) {
        for (float cnr : cnrPoints) {
            sg::SceneConfig cfg;
            cfg.durationSeconds = 120.0;
            cfg.cnrDb = cnr;
            cfg.seed = 1u;
            sg::Scene scene = sg::generate(cfg);

            // Only trace the first combination, or the files get unwieldy.
            const char* trace = (csvPath && e.type == ad::ESTIMATOR_PERCENTILE
                                 && cnr == cnrPoints[0]) ? csvPath : NULL;
            RunResult r = runOne(scene, e.type, trace);

            std::printf("%-12s %6.1f %12.3f %10.3f %10.1f %10.1f %7d %8d %10.1f\n",
                        e.name, cnr,
                        100.0 * r.score.falseOpenFrac,
                        100.0 * r.score.missFrac,
                        r.score.fecMs, r.score.overMs,
                        r.score.mscCount, r.score.chatter,
                        r.nsPerSample);
        }
        std::printf("\n");
    }
    return 0;
}
