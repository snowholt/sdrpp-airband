#pragma once
#include <atomic>
#include <cstdint>

namespace dsp::noise_reduction::adaptive {

    // Everything a user or a test needs to understand why the gate did what it
    // did. Prior art across SDR software is consistently opaque about estimator
    // internals; exposing this is a deliberate difference and it is what makes
    // field tuning possible at all.
    struct Diagnostics {
        float framePowerDb = -140.0f;
        float noiseFloorDb = -140.0f;
        float snrDb = 0.0f;
        float occupancy = 0.0f;
        float carrierConfidence = 0.0f;   // Phase 1b; always 0 in Phase 1
        float carrierOffsetHz = 0.0f;     // Phase 1b; always 0 in Phase 1
        int state = 0;
        int lastTransitionReason = 0;
    };

    // Single-writer, multi-reader seqlock.
    //
    // The DSP thread writes and must never block on the GUI, so a mutex is the
    // wrong tool here: a GUI stall would stall audio. The writer brackets each
    // update with an odd then even sequence number; a reader that sees an odd
    // sequence, or a sequence that changed across its read, retries. The writer
    // is wait-free.
    class DiagnosticsPublisher {
    public:
        void publish(const Diagnostics& d) {
            const uint32_t s = _seq.load(std::memory_order_relaxed);
            _seq.store(s + 1, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
            _data = d;
            std::atomic_thread_fence(std::memory_order_release);
            _seq.store(s + 2, std::memory_order_relaxed);
        }

        Diagnostics read() const {
            Diagnostics out;
            for (;;) {
                const uint32_t before = _seq.load(std::memory_order_relaxed);
                // Odd means a write is in progress. Retry without reading.
                if (before & 1u) { continue; }
                std::atomic_thread_fence(std::memory_order_acquire);
                out = _data;
                std::atomic_thread_fence(std::memory_order_acquire);
                const uint32_t after = _seq.load(std::memory_order_relaxed);
                if (before == after) { return out; }
            }
        }

    private:
        std::atomic<uint32_t> _seq{ 0 };
        Diagnostics _data;
    };
}
