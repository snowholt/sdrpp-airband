#pragma once
#include "../processor.h"
#include "am_nr/stft.h"
#include "am_nr/noise_psd.h"
#include "am_nr/gain_rule.h"
#include <atomic>

namespace dsp::noise_reduction {

    namespace amnr {
        enum NrState {
            NR_OFF,
            NR_WARMUP,
            NR_ACTIVE,
            NR_BYPASS_UNRELIABLE
        };

        inline const char* nrStateName(int s) {
            switch (s) {
            case NR_OFF:               return "OFF";
            case NR_WARMUP:            return "WARMUP";
            case NR_ACTIVE:            return "ACTIVE";
            case NR_BYPASS_UNRELIABLE: return "BYPASS";
            }
            return "UNKNOWN";
        }
    }

    // Mild statistical noise reduction for the AM audio chain (design section
    // 7, step 2b): a capped log-MMSE spectral gain driven by a reliability-
    // gated per-bin noise PSD tracker, running inside a weighted overlap-add
    // STFT.
    //
    // Channel handling: the analysis power spectrum is the average of both
    // channels' powers, and one shared gain is applied to each channel's
    // spectrum. The AM demodulator delivers effectively mono content, and a
    // shared gain preserves whatever image exists by construction.
    //
    // Bypass semantics: whenever the tracker is not reliable - warm-up,
    // staleness or occupancy - the frame callback leaves the spectra exactly
    // untouched, so bypassed audio is bit-identical to a disabled block, not
    // merely similar. Enabling mid-stream clears all DSP state and warms up
    // again: a noise estimate from another moment is worse than none.
    class AmNR : public Processor<dsp::stereo_t, dsp::stereo_t> {
        using base_type = Processor<dsp::stereo_t, dsp::stereo_t>;

        class FrameProcessor : public amnr::StereoFrameListener {
        public:
            AmNR* owner = nullptr;
            void processFrame(fftwf_complex* l, fftwf_complex* r) override {
                owner->onFrame(l, r);
            }
        };

    public:
        AmNR() { _frameProc.owner = this; }

        void init(stream<dsp::stereo_t>* in, double samplerate) {
            _samplerate = samplerate;
            rebuild();
            base_type::init(in);
        }

        void setSamplerate(double samplerate) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            if (!_block_init || samplerate == _samplerate) { return; }
            base_type::tempStop();
            _samplerate = samplerate;
            rebuild();
            base_type::tempStart();
        }

        void setEnabled(bool enabled) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            if (!_block_init) {
                _enabled.store(enabled, std::memory_order_relaxed);
                publishState();
                return;
            }
            base_type::tempStop();
            _enabled.store(enabled, std::memory_order_relaxed);
            clearDspState();
            base_type::tempStart();
        }

        bool enabled() const { return _enabled.load(std::memory_order_relaxed); }

        int nrState() const { return _state.load(std::memory_order_relaxed); }

        void reset() {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            base_type::tempStop();
            clearDspState();
            base_type::tempStart();
        }

        int frameSize() const { return _engine.frameSize(); }
        int hop() const { return _engine.hop(); }
        double samplerate() const { return _samplerate; }

        // Availability latency of the underlying STFT batching.
        int latencySamples() const { return _engine.latencySamples(); }

        int process(int count, const dsp::stereo_t* in, dsp::stereo_t* out) {
            return _engine.process(count, in, out);
        }

        int run() {
            int count = base_type::_in->read();
            if (count < 0) { return -1; }
            int outCount = process(count, base_type::_in->readBuf, base_type::out.writeBuf);
            base_type::_in->flush();
            if (outCount) {
                if (!base_type::out.swap(outCount)) { return -1; }
            }
            return count;
        }

    private:
        friend class FrameProcessor;

        void rebuild() {
            _engine.init(&_frameProc, _samplerate);
            _tracker.init(_engine.bins(),
                          (double)_engine.hop() / _samplerate);
            _rule.init(_engine.bins());
            const size_t b = (size_t)_engine.bins();
            _magSq.assign(b, 0.0f);
            _psd.assign(b, 0.0f);
            _gains.assign(b, 1.0f);
            clearDspState();
        }

        void clearDspState() {
            _engine.reset();
            _tracker.reset();
            _rule.reset();
            _everActive = false;
            publishState();
        }

        void publishState() {
            int s;
            if (!_enabled.load(std::memory_order_relaxed)) {
                s = amnr::NR_OFF;
            } else if (_everActive) {
                s = _tracker.reliable() ? amnr::NR_ACTIVE
                                        : amnr::NR_BYPASS_UNRELIABLE;
            } else {
                s = amnr::NR_WARMUP;
            }
            _state.store(s, std::memory_order_relaxed);
        }

        void onFrame(fftwf_complex* lSpec, fftwf_complex* rSpec) {
            if (!_enabled.load(std::memory_order_relaxed)) {
                // Spectra stay untouched: bit-exact passthrough.
                _state.store(amnr::NR_OFF, std::memory_order_relaxed);
                return;
            }

            const int b = _engine.bins();
            for (int k = 0; k < b; k++) {
                const float lr = lSpec[k][0] * lSpec[k][0] + lSpec[k][1] * lSpec[k][1];
                const float rr = rSpec[k][0] * rSpec[k][0] + rSpec[k][1] * rSpec[k][1];
                _magSq[(size_t)k] = 0.5f * (lr + rr);
            }

            _tracker.process(_magSq.data(), _psd.data());

            if (_tracker.reliable()) {
                _everActive = true;
                // The rule consumes the tracker's smoothed periodogram, not
                // the raw frame powers: chi-square fluctuations of active
                // speech must not gate enhancement within syllables.
                _rule.process(_tracker.smoothedPower().data(), _psd.data(), _gains.data());
                for (int k = 0; k < b; k++) {
                    const float g = _gains[(size_t)k];
                    lSpec[k][0] *= g;
                    lSpec[k][1] *= g;
                    rSpec[k][0] *= g;
                    rSpec[k][1] *= g;
                }
                _state.store(amnr::NR_ACTIVE, std::memory_order_relaxed);
            } else {
                // Unreliable estimate: unity, bit-exact. Warm-up until the
                // tracker has ever been reliable, bypass afterwards.
                _state.store(_everActive ? amnr::NR_BYPASS_UNRELIABLE
                                         : amnr::NR_WARMUP,
                             std::memory_order_relaxed);
            }
        }

        FrameProcessor _frameProc;
        amnr::StftEngine _engine;
        amnr::NoisePsdTracker _tracker;
        amnr::GainRule _rule;

        std::vector<float> _magSq, _psd, _gains;

        double _samplerate = 48000.0;
        std::atomic<bool> _enabled{ false };
        std::atomic<int> _state{ amnr::NR_OFF };
        bool _everActive = false;
    };
}
