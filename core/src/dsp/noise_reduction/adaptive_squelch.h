#pragma once
#include "../processor.h"
#include "adaptive_squelch/frame_meter.h"
#include "adaptive_squelch/noise_estimator.h"
#include "adaptive_squelch/preroll.h"
#include "adaptive_squelch/gate.h"
#include "adaptive_squelch/diagnostics.h"

namespace dsp::noise_reduction {

    // Adaptive squelch. Implements the SQUELCH_MODE_SNR branch that upstream
    // left stubbed at radio_module.h:675.
    //
    // The stream reaching this block has already been filtered to the channel
    // bandwidth by RxVFO, so there is no adjacent-channel noise reference and
    // the floor must be estimated across time. Every time constant is derived
    // from the samplerate in seconds, because process() receives a variable
    // count (stream.h:70 returns whatever upstream wrote).
    class AdaptiveSquelch : public Processor<complex_t, complex_t> {
        using base_type = Processor<complex_t, complex_t>;
    public:
        AdaptiveSquelch() {}

        void init(stream<complex_t>* in, double samplerate) {
            _samplerate = samplerate;
            rebuild();
            base_type::init(in);
        }

        // ---- lifecycle and invalidation (U5) ----------------------------
        // Every one of these invalidates estimator state, the delay line and
        // the state machine. Stale state from a different samplerate, gain or
        // band is worse than no state at all.

        void setSamplerate(double samplerate) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            if (samplerate == _samplerate) { return; }
            base_type::tempStop();
            _samplerate = samplerate;
            rebuild();
            base_type::tempStart();
        }

        void reset() {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            base_type::tempStop();
            clearState();
            base_type::tempStart();
        }

        // ---- configuration ----------------------------------------------

        void setEstimator(adaptive::EstimatorType type) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            base_type::tempStop();
            _estimatorType = type;
            selectEstimator();
            clearState();
            base_type::tempStart();
        }

        void setOpenMargin(float dB) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            _openMarginDb = dB;
            _gate.setMargins(_openMarginDb, _openMarginDb - _hysteresisDb);
        }

        void setHysteresis(float dB) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            _hysteresisDb = (dB < 0.0f) ? 0.0f : dB;
            _gate.setMargins(_openMarginDb, _openMarginDb - _hysteresisDb);
        }

        void setQualify(double seconds) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            _gate.setQualify(seconds);
        }

        void setHang(double seconds) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            _gate.setHang(seconds);
        }

        void setRelease(double seconds) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            _gate.setRelease(seconds);
        }

        void setRamp(double seconds) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            _gate.setRamp(seconds);
        }

        // Zero disables pre-roll entirely and removes the added latency.
        void setPreRoll(double seconds) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            base_type::tempStop();
            _preRollSeconds = (seconds < 0.0) ? 0.0 : seconds;
            _preRoll.setLength(_samplerate, _preRollSeconds);
            base_type::tempStart();
        }

        double preRollSeconds() const { return _preRollSeconds; }

        adaptive::Diagnostics diagnostics() const { return _diag.read(); }

        // ---- processing ---------------------------------------------------

        // Returns the number of samples written to out, which is fewer than
        // count while the pre-roll line primes.
        int process(int count, const complex_t* in, complex_t* out) {
            int emitted = 0;
            float frameDb = 0.0f;

            for (int i = 0; i < count; i++) {
                // The gate decision for this frame is computed from the newest
                // samples but applied to samples one pre-roll interval older,
                // which is what stops the onset being clipped.
                if (_meter.feedSample(in[i], frameDb)) {
                    const float noiseDb = _estimator->update(frameDb, _gate.isOpen());
                    const float snrDb = frameDb - noiseDb;
                    _gate.updateFrame(snrDb);

                    adaptive::Diagnostics d;
                    d.framePowerDb = frameDb;
                    d.noiseFloorDb = noiseDb;
                    d.snrDb = snrDb;
                    d.occupancy = _estimator->occupancy();
                    d.state = (int)_gate.state();
                    d.lastTransitionReason = (int)_gate.reason();
                    _diag.publish(d);
                }

                complex_t delayed;
                if (_preRoll.push(in[i], delayed)) {
                    const float g = _gate.nextGain();
                    out[emitted].re = delayed.re * g;
                    out[emitted].im = delayed.im * g;
                    emitted++;
                }
            }
            return emitted;
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
        void rebuild() {
            _meter.init(_samplerate, FRAME_SECONDS);
            _preRoll.setLength(_samplerate, _preRollSeconds);
            _gate.init(_samplerate, FRAME_SECONDS);
            _gate.setMargins(_openMarginDb, _openMarginDb - _hysteresisDb);
            selectEstimator();
            clearState();
        }

        void selectEstimator() {
            switch (_estimatorType) {
            case adaptive::ESTIMATOR_MCRA:      _estimator = &_mcra; break;
            case adaptive::ESTIMATOR_GATED_EMA: _estimator = &_ema;  break;
            case adaptive::ESTIMATOR_PERCENTILE:
            default:                            _estimator = &_pct;  break;
            }
            _estimator->init(FRAME_SECONDS);
        }

        void clearState() {
            _meter.reset();
            _preRoll.reset();
            _gate.reset();
            _estimator->reset();
            _diag.publish(adaptive::Diagnostics{});
        }

        static constexpr double FRAME_SECONDS = 0.010;

        double _samplerate = 48000.0;
        double _preRollSeconds = 0.040;
        float _openMarginDb = 9.5f;
        float _hysteresisDb = 3.0f;

        adaptive::EstimatorType _estimatorType = adaptive::ESTIMATOR_PERCENTILE;
        adaptive::PercentileEstimator _pct;
        adaptive::McraEstimator _mcra;
        adaptive::GatedEmaEstimator _ema;
        adaptive::NoiseEstimator* _estimator = &_pct;

        adaptive::FrameMeter _meter;
        adaptive::PreRoll _preRoll;
        adaptive::Gate _gate;
        adaptive::DiagnosticsPublisher _diag;
    };
}
