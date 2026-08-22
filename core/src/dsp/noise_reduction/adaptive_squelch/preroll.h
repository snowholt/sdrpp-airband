#pragma once
#include <cmath>
#include <vector>
#include "../../types.h"

namespace dsp::noise_reduction::adaptive {

    // Fixed look-behind so a detected onset can be emitted from history rather
    // than clipped. A gain ramp removes the click at a gate transition but it
    // cannot restore a consonant that was already cut, which is why this exists.
    //
    // The cost is latency on the audio path, so this is user-configurable and
    // user-disableable. Enabling the block mid-stream produces a one-time gap of
    // one pre-roll interval; the chain tolerates that because it makes no
    // sample-count-parity assumption between blocks (chain.h enableBlock), and
    // fm_if.h already introduces latency inside the same ifChain.
    class PreRoll {
    public:
        void init(double samplerate, double seconds) { setLength(samplerate, seconds); }

        void setLength(double samplerate, double seconds) {
            _delay = (int)std::lround(samplerate * seconds);
            if (_delay < 0) { _delay = 0; }
            _buf.assign((size_t)_delay + 1, complex_t{ 0.0f, 0.0f });
            reset();
        }

        void reset() {
            std::fill(_buf.begin(), _buf.end(), complex_t{ 0.0f, 0.0f });
            _head = 0;
            _primed = 0;
        }

        int delaySamples() const { return _delay; }

        // Pushes one sample and writes the sample from delaySamples() ago into
        // out. Returns false while the line is still priming, in which case out
        // holds no meaningful value and the caller must not emit it.
        inline bool push(const complex_t& in, complex_t& out) {
            if (_delay == 0) {
                out = in;
                return true;
            }
            _buf[(size_t)_head] = in;
            _head++;
            if (_head >= (int)_buf.size()) { _head = 0; }
            out = _buf[(size_t)_head];
            if (_primed <= _delay) {
                _primed++;
                return _primed > _delay;
            }
            return true;
        }

    private:
        std::vector<complex_t> _buf;
        int _delay = 0;
        int _head = 0;
        int _primed = 0;
    };
}
