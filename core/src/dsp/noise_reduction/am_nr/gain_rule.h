#pragma once
#include <cmath>
#include <vector>

namespace dsp::noise_reduction::amnr {

    inline float expint1(float xf) {
        // Exponential integral E1(x) = integral_x..inf e^-t/t dt, x > 0.
        // Series below 1 (fast convergence), continued fraction above.
        double x = (double)xf;
        if (!(x > 1e-10)) { x = 1e-10; }
        const double EULER = 0.57721566490153286;
        if (x < 1.0) {
            double sum = 0.0;
            double term = 1.0;                 // (-x)^k / k!
            for (int k = 1; k <= 24; k++) {
                term *= -x / (double)k;
                const double contrib = term / (double)k;
                sum += contrib;
                if (std::fabs(contrib) < 1e-20) { break; }
            }
            return (float)(-EULER - std::log(x) - sum);
        }
        const int MAXIT = 120;
        const double FPMIN = 1e-300;
        double b = x + 1.0;
        double c = 1.0 / FPMIN;
        double d = 1.0 / b;
        double h = d;
        for (int i = 1; i <= MAXIT; i++) {
            const double an = -(double)i;
            b += 2.0;
            d = an * d + b;
            if (std::fabs(d) < FPMIN) { d = FPMIN; }
            c = b + an / c;
            if (std::fabs(c) < FPMIN) { c = FPMIN; }
            d = 1.0 / d;
            const double del = d * c;
            h *= del;
            if (std::fabs(del - 1.0) < 1e-12) { break; }
        }
        return (float)(std::exp(-x) * h);
    }

    // Capped log-MMSE amplitude gain rule for the AM noise reduction chain
    // (design section 7, step 2b).
    //
    // Ephraim-Malah (1985): per-bin gain G = xi/(1+xi) * exp(0.5*E1(v)) with
    // v = gamma*xi/(1+xi), where gamma is the a posteriori SNR and xi a priori
    // SNR estimated by decision-directed smoothing of the previous frame's
    // enhanced power. The design's hard attenuation cap is applied as an
    // absolute gain floor: no bin is ever attenuated more than CAP_DB,
    // which - together with the DD smoothing and the tracker's frequency
    // smoothing upstream - is the musical-noise defence. Spectral
    // subtraction without over-subtraction control and a floor is explicitly
    // rejected by the design; this estimator never over-subtracts at all.
    //
    // Every output is finite and inside [GAIN_FLOOR, 1] whatever the input:
    // hostile spectra are sanitized rather than trusted.
    class GainRule {
    public:
        static constexpr double DD_ALPHA = 0.98;
        static constexpr float GAIN_FLOOR = 0.35481f;   // 10^(-9/20)
        static constexpr float GAMMA_MAX = 1000.0f;
        static constexpr float V_MIN = 1e-8f;

        void init(int bins) {
            _bins = bins;
            _prevEnh.assign((size_t)_bins, 0.0f);
            reset();
        }

        void reset() {
            std::fill(_prevEnh.begin(), _prevEnh.end(), 0.0f);
        }

        // magSq: observed power spectrum; noisePsd: current noise estimate.
        // gainOut receives linear amplitude gains in [GAIN_FLOOR, 1].
        void process(const float* magSq, const float* noisePsd, float* gainOut) {
            for (int k = 0; k < _bins; k++) {
                float n = noisePsd[k];
                float y = magSq[k];
                if (!std::isfinite(y) || y < 0.0f) { y = 0.0f; }

                // No usable floor reference: pass through untouched.
                if (!(n > 0.0f) || !std::isfinite(n)) {
                    _prevEnh[(size_t)k] = y;
                    gainOut[k] = 1.0f;
                    continue;
                }

                float gamma = y / n;
                if (!(gamma > 0.0f)) { gamma = 0.0f; }
                if (gamma > GAMMA_MAX) { gamma = GAMMA_MAX; }

                // Decision-directed a priori SNR from the previous frame's
                // enhanced power plus the positive part of this frame's
                // a posteriori excess.
                float xi = (float)DD_ALPHA * (_prevEnh[(size_t)k] / n);
                const float excess = gamma - 1.0f;
                if (excess > 0.0f) { xi += (float)(1.0 - DD_ALPHA) * excess; }
                if (!(xi > 1e-6f)) { xi = 1e-6f; }

                float v = gamma * xi / (1.0f + xi);
                if (!(v > V_MIN)) { v = V_MIN; }

                float g = (xi / (1.0f + xi)) * std::exp(0.5f * expint1(v));
                if (!(g >= GAIN_FLOOR)) { g = GAIN_FLOOR; }   // also catches NaN
                if (g > 1.0f) { g = 1.0f; }

                gainOut[k] = g;
                _prevEnh[(size_t)k] = g * g * y;
            }
        }

    private:
        int _bins = 0;
        std::vector<float> _prevEnh;
    };
}
