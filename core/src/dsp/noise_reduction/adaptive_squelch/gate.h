#pragma once
#include <cmath>

namespace dsp::noise_reduction::adaptive {

    enum GateState {
        GATE_CLOSED,
        GATE_QUALIFY,
        GATE_OPEN,
        GATE_HANG
    };

    enum TransitionReason {
        REASON_RESET,
        REASON_SNR_ABOVE_OPEN,
        REASON_QUALIFY_ABANDONED,
        REASON_QUALIFY_MET,
        REASON_SNR_BELOW_CLOSE,
        REASON_HANG_EXPIRED
    };

    inline const char* stateName(GateState s) {
        switch (s) {
        case GATE_CLOSED:  return "CLOSED";
        case GATE_QUALIFY: return "QUALIFY";
        case GATE_OPEN:    return "OPEN";
        case GATE_HANG:    return "HANG";
        }
        return "UNKNOWN";
    }

    inline const char* reasonName(TransitionReason r) {
        switch (r) {
        case REASON_RESET:              return "reset";
        case REASON_SNR_ABOVE_OPEN:     return "snr_above_open";
        case REASON_QUALIFY_ABANDONED:  return "qualify_abandoned";
        case REASON_QUALIFY_MET:        return "qualify_met";
        case REASON_SNR_BELOW_CLOSE:    return "snr_below_close";
        case REASON_HANG_EXPIRED:       return "hang_expired";
        }
        return "unknown";
    }

    // The decision layer. Most of the perceived quality of a squelch lives here
    // rather than in the estimator, so it is shared unchanged by every
    // estimator; that is also what lets the harness compare estimators with
    // everything else held constant.
    //
    // Two thresholds, never one. openDb and closeDb are both dB above the
    // estimated noise floor and their difference is the hysteresis.
    class Gate {
    public:
        void init(double samplerate, double frameSeconds) {
            _samplerate = samplerate;
            _frameSeconds = frameSeconds;
            recomputeFrames();
            setRamp(_rampSeconds);
            reset();
        }

        void setMargins(float openDb, float closeDb) {
            _openDb = openDb;
            _closeDb = (closeDb > openDb) ? openDb : closeDb;
        }

        void setQualify(double seconds) { _qualifySeconds = seconds; recomputeFrames(); }
        void setHang(double seconds)    { _hangSeconds = seconds;    recomputeFrames(); }
        void setRelease(double seconds) { _releaseSeconds = seconds; recomputeFrames(); }

        void setRamp(double seconds) {
            _rampSeconds = seconds;
            _rampSamples = (int)std::lround(_samplerate * seconds);
            if (_rampSamples < 1) { _rampSamples = 1; }
        }

        void reset() {
            _state = GATE_CLOSED;
            _reason = REASON_RESET;
            _counter = 0;
            _target = 0.0f;
            _gain = 0.0f;
            _rampFrom = 0.0f;
            _rampLeft = 0;
        }

        GateState state() const { return _state; }
        TransitionReason reason() const { return _reason; }
        bool isOpen() const { return _state == GATE_OPEN || _state == GATE_HANG; }

        // Advances the state machine by exactly one frame.
        void updateFrame(float snrDb) {
            switch (_state) {
            case GATE_CLOSED:
                if (snrDb >= _openDb) {
                    // A qualify period of N frames must take exactly N frames,
                    // including N = 1, so the threshold is checked on entry
                    // rather than only on the following frame.
                    _counter = 1;
                    _reason = REASON_SNR_ABOVE_OPEN;
                    if (_counter >= _qualifyFrames) {
                        _state = GATE_OPEN;
                        _counter = 0;
                        _reason = REASON_QUALIFY_MET;
                        setTarget(1.0f);
                    }
                    else {
                        _state = GATE_QUALIFY;
                    }
                }
                break;

            case GATE_QUALIFY:
                if (snrDb < _openDb) {
                    _state = GATE_CLOSED;
                    _counter = 0;
                    _reason = REASON_QUALIFY_ABANDONED;
                }
                else {
                    _counter++;
                    if (_counter >= _qualifyFrames) {
                        _state = GATE_OPEN;
                        _counter = 0;
                        _reason = REASON_QUALIFY_MET;
                        setTarget(1.0f);
                    }
                }
                break;

            case GATE_OPEN:
                if (snrDb < _closeDb) {
                    _state = GATE_HANG;
                    _counter = 0;
                    _reason = REASON_SNR_BELOW_CLOSE;
                }
                break;

            case GATE_HANG:
                if (snrDb >= _openDb) {
                    _state = GATE_OPEN;
                    _counter = 0;
                    _reason = REASON_SNR_ABOVE_OPEN;
                }
                else {
                    _counter++;
                    // Hang and release are one countdown. Nothing distinguishes
                    // their behaviour, only their intent: hang protects word
                    // endings, release confirms the channel really went quiet.
                    if (_counter >= _hangFrames + _releaseFrames) {
                        _state = GATE_CLOSED;
                        _counter = 0;
                        _reason = REASON_HANG_EXPIRED;
                        setTarget(0.0f);
                    }
                }
                break;
            }
        }

        // Advances the ramp by exactly one sample and returns the gain to apply.
        // A raised cosine starts and ends with zero slope, which is what removes
        // the click that power_squelch.h's memcpy/memset switch produces.
        inline float nextGain() {
            if (_rampLeft > 0) {
                _rampLeft--;
                const float phase = 1.0f - ((float)_rampLeft / (float)_rampSamples);
                const float shaped = 0.5f - 0.5f * std::cos(3.14159265358979f * phase);
                _gain = _rampFrom + (_target - _rampFrom) * shaped;
            }
            else {
                _gain = _target;
            }
            return _gain;
        }

    private:
        void setTarget(float t) {
            if (t == _target) { return; }
            _rampFrom = _gain;
            _target = t;
            _rampLeft = _rampSamples;
        }

        int framesFor(double seconds) const {
            int n = (int)std::lround(seconds / _frameSeconds);
            return (n < 1) ? 1 : n;
        }

        void recomputeFrames() {
            _qualifyFrames = framesFor(_qualifySeconds);
            _hangFrames = framesFor(_hangSeconds);
            _releaseFrames = framesFor(_releaseSeconds);
        }

        double _samplerate = 48000.0;
        double _frameSeconds = 0.010;

        float _openDb = 9.5f;
        float _closeDb = 6.5f;
        double _qualifySeconds = 0.030;
        double _hangSeconds = 0.200;
        double _releaseSeconds = 0.150;
        double _rampSeconds = 0.005;

        int _qualifyFrames = 3;
        int _hangFrames = 20;
        int _releaseFrames = 15;
        int _rampSamples = 240;

        GateState _state = GATE_CLOSED;
        TransitionReason _reason = REASON_RESET;
        int _counter = 0;

        float _target = 0.0f;
        float _gain = 0.0f;
        float _rampFrom = 0.0f;
        int _rampLeft = 0;
    };
}
