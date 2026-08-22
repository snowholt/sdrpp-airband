#pragma once
#include <cmath>
#include <vector>
#include "signal_gen.h"

// Ground-truth timing metrics for a gate. The terminology follows the voice
// activity detection literature deliberately, so results are comparable with
// published work: FEC is front-end clipping, MSC is mid-speech clipping and
// OVER is overhang.
namespace scorer {

    struct Score {
        double falseOpenFrac = 0.0;   // fraction of idle time passed unmuted
        double missFrac = 0.0;        // fraction of labelled time muted
        double fecMs = 0.0;           // mean speech lost after a true onset
        double overMs = 0.0;          // mean noise passed after a true end
        int mscCount = 0;             // closures inside a transmission
        int chatter = 0;              // total gate transitions
    };

    // gateOpen is one entry per OUTPUT sample.
    //
    // No delay compensation is applied here, and that is deliberate. The block
    // emits the pre-rolled sample, so output index j holds input sample j; the
    // pre-roll shows up as the gate decision for output j having been taken one
    // pre-roll interval later in input time. Subtracting the delay again would
    // cancel exactly the benefit this metric exists to measure.
    inline Score score(const std::vector<char>& gateOpen,
                       const std::vector<sg::Label>& labels,
                       double samplerate) {
        Score s;
        const long n = (long)gateOpen.size();
        const double msPerSample = 1000.0 / samplerate;

        auto openAtInput = [&](long inputIdx) -> bool {
            if (inputIdx < 0 || inputIdx >= n) { return false; }
            return gateOpen[(size_t)inputIdx] != 0;
        };

        // Mark which input samples are inside a labelled transmission.
        std::vector<char> active((size_t)n, 0);
        for (auto& l : labels) {
            for (long i = l.start; i < l.end && i < (long)active.size(); i++) {
                active[(size_t)i] = 1;
            }
        }

        long idleTotal = 0, idleOpen = 0, activeTotal = 0, activeMuted = 0;
        for (long i = 0; i < (long)active.size(); i++) {
            const bool op = openAtInput(i);
            if (active[(size_t)i]) {
                activeTotal++;
                if (!op) { activeMuted++; }
            }
            else {
                idleTotal++;
                if (op) { idleOpen++; }
            }
        }
        s.falseOpenFrac = idleTotal ? (double)idleOpen / (double)idleTotal : 0.0;
        s.missFrac = activeTotal ? (double)activeMuted / (double)activeTotal : 0.0;

        double fecSum = 0.0, overSum = 0.0;
        for (auto& l : labels) {
            long fec = 0;
            for (long i = l.start; i < l.end; i++) {
                if (openAtInput(i)) { break; }
                fec++;
            }
            fecSum += (double)fec * msPerSample;

            long over = 0;
            for (long i = l.end; i < (long)active.size() && !active[(size_t)i]; i++) {
                if (!openAtInput(i)) { break; }
                over++;
            }
            overSum += (double)over * msPerSample;

            // Closures strictly inside the transmission, after it first opened.
            bool started = false, wasOpen = false;
            for (long i = l.start; i < l.end; i++) {
                const bool op = openAtInput(i);
                if (!started) {
                    if (op) { started = true; wasOpen = true; }
                    continue;
                }
                if (wasOpen && !op) { s.mscCount++; }
                wasOpen = op;
            }
        }
        if (!labels.empty()) {
            s.fecMs = fecSum / (double)labels.size();
            s.overMs = overSum / (double)labels.size();
        }

        for (long i = 1; i < n; i++) {
            if (gateOpen[(size_t)i] != gateOpen[(size_t)i - 1]) { s.chatter++; }
        }
        return s;
    }
}
