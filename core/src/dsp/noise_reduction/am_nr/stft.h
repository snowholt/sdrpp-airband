#pragma once
#include <fftw3.h>
#include <cmath>
#include <vector>
#include "../../types.h"

namespace dsp::noise_reduction::amnr {

    // Implemented by the owner to modify each analysis frame's spectra in
    // place, between the forward and inverse transforms. Bin layout is the
    // r2c layout: bins 0..frameSize/2, DC at index 0.
    struct StereoFrameListener {
        virtual void processFrame(fftwf_complex* lSpectrum, fftwf_complex* rSpectrum) = 0;
        virtual ~StereoFrameListener() {}
    };

    // Streaming weighted overlap-add STFT core for the AM noise reduction
    // chain (design section 7, step 2b).
    //
    // Frames are 10 ms rounded up to a power of two at the current samplerate
    // (512 at 48 kHz), hop is half a frame, and analysis/synthesis windows are
    // the square-root Hann pair, whose overlap sum is exactly constant at 50%
    // overlap. FFTW plans follow the lifecycle established by fm_if.h:
    // allocated and planned outside processing, destroyed under the same lock
    // that stops the worker.
    //
    // Output accounting is honest about batching: a sample becomes final only
    // when the last overlapping frame covering it has been synthesized, which
    // happens once per hop. process() therefore emits whatever is final and
    // returns how much was written - fewer than count while priming or when
    // callers deliver sub-hop chunks. Over any stretch longer than one frame
    // the emitted total equals the consumed total minus the fixed algorithmic
    // latency of latencySamples(), and the concatenated output is bitwise
    // identical however the input is chunked.
    class StftEngine {
    public:
        static constexpr double TARGET_FRAME_SECONDS = 0.010;

        StftEngine() {}
        ~StftEngine() { destroyBuffers(); }

        void init(StereoFrameListener* listener, double samplerate) {
            destroyBuffers();
            _listener = listener;
            _samplerate = samplerate;
            initBuffers();
        }

        void setSamplerate(double samplerate) {
            if (!_buffersValid || samplerate == _samplerate) { return; }
            destroyBuffers();
            _samplerate = samplerate;
            initBuffers();
        }

        void reset() {
            if (!_buffersValid) { return; }
            std::fill(_inL.begin(), _inL.end(), 0.0f);
            std::fill(_inR.begin(), _inR.end(), 0.0f);
            std::fill(_accL.begin(), _accL.end(), 0.0f);
            std::fill(_accR.begin(), _accR.end(), 0.0f);
            _totalIn = 0;
            _totalOut = 0;
            _nextFrameStart = 0;
        }

        int frameSize() const { return _n; }
        int hop() const { return _hop; }
        int bins() const { return _n / 2 + 1; }
        double samplerate() const { return _samplerate; }

        // Availability horizon of the batching, not a signal displacement:
        // output index j holds input index j (the reconstruction is aligned),
        // but sample j is only final once the frame starting hop later than
        // it has been synthesized, which requires this much newer input.
        int latencySamples() const { return _n - _hop; }

        int process(int count, const dsp::stereo_t* in, dsp::stereo_t* out) {
            int emitted = 0;
            for (int k = 0; k < count; k++) {
                const long long a = _totalIn;
                const size_t slot = (size_t)(a % _n);
                _inL[slot] = in[k].l;
                _inR[slot] = in[k].r;
                _totalIn++;

                while (_totalIn >= _nextFrameStart + _n) {
                    runFrame();
                    _nextFrameStart += _hop;
                }

                // Drain whenever anything is final - NOT only on frame
                // triggers. Draining inside the trigger branch alone lets
                // callers delivering sub-hop chunks starve emission: the
                // finalized backlog grows, and the overlap-add ring
                // eventually overwrites its own pending tail. Per-sample
                // draining keeps the write frontier from ever lapping the
                // read frontier within the ring's two-frame span.
                while (_totalOut < _nextFrameStart && emitted < count) {
                    const size_t oslot = (size_t)(_totalOut % (2 * _n));
                    out[emitted].l = _accL[oslot];
                    out[emitted].r = _accR[oslot];
                    _accL[oslot] = 0.0f;
                    _accR[oslot] = 0.0f;
                    _totalOut++;
                    emitted++;
                }
            }
            return emitted;
        }

    private:
        void initBuffers() {
            // Smallest power of two covering TARGET_FRAME_SECONDS.
            int n = 64;
            while ((double)n < TARGET_FRAME_SECONDS * _samplerate) { n <<= 1; }
            _n = n;
            _hop = n / 2;

            _inL.assign((size_t)_n, 0.0f);
            _inR.assign((size_t)_n, 0.0f);
            _accL.assign((size_t)(2 * _n), 0.0f);
            _accR.assign((size_t)(2 * _n), 0.0f);

            _win.resize((size_t)_n);
            for (int k = 0; k < _n; k++) {
                // Periodic square-root Hann: w_a*w_s sums to exactly 1 at 50%
                // overlap, so the unity-gain roundtrip needs only the 1/N
                // unnormalized-inverse correction.
                _win[(size_t)k] = std::sqrt(0.5f * (1.0f - std::cos(6.283185307179586f * (float)k / (float)_n)));
            }

            _timeL = (float*)fftwf_malloc(sizeof(float) * (size_t)_n);
            _timeR = (float*)fftwf_malloc(sizeof(float) * (size_t)_n);
            _specL = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size_t)bins());
            _specR = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size_t)bins());
            _fwdL = fftwf_plan_dft_r2c_1d(_n, _timeL, _specL, FFTW_ESTIMATE);
            _fwdR = fftwf_plan_dft_r2c_1d(_n, _timeR, _specR, FFTW_ESTIMATE);
            _invL = fftwf_plan_dft_c2r_1d(_n, _specL, _timeL, FFTW_ESTIMATE);
            _invR = fftwf_plan_dft_c2r_1d(_n, _specR, _timeR, FFTW_ESTIMATE);

            _totalIn = 0;
            _totalOut = 0;
            _nextFrameStart = 0;
            _buffersValid = true;
        }

        void destroyBuffers() {
            if (!_buffersValid) { return; }
            fftwf_destroy_plan(_invR);
            fftwf_destroy_plan(_invL);
            fftwf_destroy_plan(_fwdR);
            fftwf_destroy_plan(_fwdL);
            fftwf_free(_specR);
            fftwf_free(_specL);
            fftwf_free(_timeR);
            fftwf_free(_timeL);
            _inL.clear();  _inL.shrink_to_fit();
            _inR.clear();  _inR.shrink_to_fit();
            _accL.clear(); _accL.shrink_to_fit();
            _accR.clear(); _accR.shrink_to_fit();
            _win.clear();  _win.shrink_to_fit();
            _buffersValid = false;
        }

        void runFrame() {
            const long long s = _nextFrameStart;
            for (int k = 0; k < _n; k++) {
                const size_t slot = (size_t)((s + k) % _n);
                _timeL[k] = _inL[slot] * _win[(size_t)k];
                _timeR[k] = _inR[slot] * _win[(size_t)k];
            }

            fftwf_execute(_fwdL);
            fftwf_execute(_fwdR);

            _listener->processFrame(_specL, _specR);

            fftwf_execute(_invL);
            fftwf_execute(_invR);

            // The c2r transform is unnormalized (yields N times the signal)
            // and the window-pair sum contributes another factor covered by
            // the same division, so a plain 1/N restores unity gain.
            const float scale = 1.0f / (float)_n;
            for (int k = 0; k < _n; k++) {
                const size_t oslot = (size_t)((s + k) % (2 * _n));
                _accL[oslot] += _timeL[k] * _win[(size_t)k] * scale;
                _accR[oslot] += _timeR[k] * _win[(size_t)k] * scale;
            }
        }

        StereoFrameListener* _listener = nullptr;
        double _samplerate = 48000.0;
        int _n = 512;
        int _hop = 256;
        bool _buffersValid = false;

        std::vector<float> _inL, _inR;          // last N input samples
        std::vector<float> _accL, _accR;        // overlap-add accumulator, 2N
        std::vector<float> _win;                // sqrt-Hann

        float* _timeL = nullptr;
        float* _timeR = nullptr;
        fftwf_complex* _specL = nullptr;
        fftwf_complex* _specR = nullptr;
        fftwf_plan _fwdL = nullptr, _fwdR = nullptr, _invL = nullptr, _invR = nullptr;

        long long _totalIn = 0;
        long long _totalOut = 0;
        long long _nextFrameStart = 0;
    };
}
