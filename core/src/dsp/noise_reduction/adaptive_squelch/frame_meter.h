#pragma once
#include <cmath>
#include <vector>
#include "../../types.h"

namespace dsp::noise_reduction::adaptive {

    // Converts a variable-length sample stream into fixed-duration power frames.
    //
    // process() receives whatever the upstream block wrote (stream.h:70 returns
    // dataSize), so nothing downstream of this class may ever see a raw sample
    // count. Frame boundaries follow the sample stream and a partial frame is
    // carried across calls, which makes the emitted frame sequence identical
    // however the input happens to be chunked.
    class FrameMeter {
    public:
        void init(double samplerate, double frameSeconds = 0.010) {
            _frameSeconds = frameSeconds;
            setSamplerate(samplerate);
        }

        void setSamplerate(double samplerate) {
            _samplerate = samplerate;
            _frameLen = (int)std::lround(_frameSeconds * samplerate);
            if (_frameLen < 1) { _frameLen = 1; }
            reset();
        }

        void reset() {
            _acc = 0.0;
            _filled = 0;
        }

        int frameLength() const { return _frameLen; }
        double frameSeconds() const { return _frameSeconds; }
        double samplerate() const { return _samplerate; }

        // Feeds one sample. Returns true, and writes the frame's dB value into
        // frameDb, on the sample that completes a frame. This is the entry
        // point the block uses, because the gate decision has to interleave
        // with the samples rather than be applied in a second pass.
        inline bool feedSample(const complex_t& s, float& frameDb) {
            _acc += (double)s.re * (double)s.re + (double)s.im * (double)s.im;
            _filled++;
            if (_filled < _frameLen) { return false; }
            frameDb = powerToDb(_acc / (double)_frameLen);
            _acc = 0.0;
            _filled = 0;
            return true;
        }

        // Appends one dB value to out for each frame completed by these samples.
        // Convenience wrapper for tests and the harness; allocates via
        // push_back, so it is never called from process().
        void feed(int count, const complex_t* in, std::vector<float>& out) {
            float db = 0.0f;
            for (int i = 0; i < count; i++) {
                if (feedSample(in[i], db)) { out.push_back(db); }
            }
        }

        // Floored at 1e-20 (-200 dB) so digital silence produces a finite value
        // rather than -inf, which would poison every downstream average.
        static float powerToDb(double meanPower) {
            if (!(meanPower > 1e-20)) { return -200.0f; }
            return (float)(10.0 * std::log10(meanPower));
        }

    private:
        double _samplerate = 0.0;
        double _frameSeconds = 0.010;
        int _frameLen = 1;
        double _acc = 0.0;
        int _filled = 0;
    };
}
