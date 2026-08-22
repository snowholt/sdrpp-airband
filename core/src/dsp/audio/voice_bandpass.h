#pragma once
#include "../processor.h"
#include "../filter/fir.h"
#include "../taps/band_pass.h"

namespace dsp::audio {

    // AM voice bandpass for the audio chain (design section 7, step 2a).
    //
    // A linear-phase FIR bandpass defaulting to 300-2700 Hz, rebuilt from
    // samplerate-derived windowed-sinc taps whenever the rate or cutoffs
    // change. The lower cutoff follows SDRangel's documented readability gain
    // for weak air traffic audio; the upper default is measured against 2500
    // and 3000 Hz variants in the harness sweep rather than exposed in the UI.
    //
    // Deterministic, structurally incapable of producing musical noise, and
    // free of dynamic allocation inside process(). Latency is pure linear-
    // phase group delay of (taps-1)/2 samples and is reported through
    // delaySamples() so downstream scoring can compensate exactly.
    class VoiceBandpass : public filter::FIR<dsp::stereo_t, float> {
        using base_type = filter::FIR<dsp::stereo_t, float>;
    public:
        static constexpr double DEFAULT_LOW_HZ = 300.0;
        static constexpr double DEFAULT_HIGH_HZ = 2700.0;
        static constexpr double TRANS_WIDTH_HZ = 200.0;

        VoiceBandpass() {}

        void init(stream<dsp::stereo_t>* in, double samplerate) {
            _samplerate = samplerate;
            _lowHz = DEFAULT_LOW_HZ;
            _highHz = DEFAULT_HIGH_HZ;
            rebuildTaps();
            base_type::init(in, _taps);
        }

        void setSamplerate(double samplerate) {
            assert(base_type::_block_init);
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            base_type::tempStop();
            _samplerate = samplerate;
            rebuildTaps();
            base_type::setTaps(_taps);
            base_type::tempStart();
        }

        void setCutoffs(double lowHz, double highHz) {
            assert(base_type::_block_init);
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            base_type::tempStop();
            _lowHz = lowHz;
            _highHz = highHz;
            rebuildTaps();
            base_type::setTaps(_taps);
            base_type::tempStart();
        }

        int delaySamples() const { return (int)(_taps.size - 1) / 2; }
        int tapCount() const { return (int)_taps.size; }

        float tapsPeak() const {
            float peak = 0.0f;
            for (int i = 0; i < (int)_taps.size; i++) {
                float a = std::fabs(_taps.taps[i]);
                if (a > peak) { peak = a; }
            }
            return peak;
        }

        double samplerate() const { return _samplerate; }
        double lowCutoffHz() const { return _lowHz; }
        double highCutoffHz() const { return _highHz; }

    private:
        void rebuildTaps() {
            // Keep the design inside the Nyquist envelope and non-degenerate:
            // the upper edge never passes 90% of Nyquist, the lower edge stays
            // above DC, and both transition bands fit between the cutoffs.
            double nyq = _samplerate * 0.5;
            double low = _lowHz;
            double high = _highHz;
            if (!(low > 1.0)) { low = 1.0; }
            double highLimit = nyq * 0.9;
            if (high > highLimit) { high = highLimit; }
            if (!(high > low + 2.0 * TRANS_WIDTH_HZ)) {
                low = high - 2.0 * TRANS_WIDTH_HZ;
                if (!(low > 1.0)) { low = 1.0; }
            }
            _lowHz = low;
            _highHz = high;

            // Odd length: symmetric Type-I taps give an integer group delay,
            // which is what makes exact delay compensation possible.
            _taps = taps::bandPass<float>(_lowHz, _highHz, TRANS_WIDTH_HZ,
                                          _samplerate, true);
        }

        double _samplerate = 48000.0;
        double _lowHz = DEFAULT_LOW_HZ;
        double _highHz = DEFAULT_HIGH_HZ;
    };
}
