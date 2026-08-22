#include "test_framework.h"
#include "signal_gen.h"

TEST(framework_self_check) {
    CHECK(1 + 1 == 2);
    CHECK_EQ(3, 3);
    CHECK_NEAR(1.0, 1.0005, 1e-3);
}

TEST(scene_is_deterministic) {
    sg::SceneConfig cfg;
    cfg.durationSeconds = 2.0;
    sg::Scene a = sg::generate(cfg);
    sg::Scene b = sg::generate(cfg);
    CHECK_EQ(a.iq.size(), b.iq.size());
    bool identical = true;
    for (size_t i = 0; i < a.iq.size(); i++) {
        if (a.iq[i].re != b.iq[i].re || a.iq[i].im != b.iq[i].im) { identical = false; break; }
    }
    CHECK(identical);
}

TEST(scene_labels_match_requested_duty) {
    sg::SceneConfig cfg;
    cfg.durationSeconds = 20.0;
    cfg.onSeconds = 3.0;
    cfg.offSeconds = 2.0;
    sg::Scene s = sg::generate(cfg);
    // 20 s at a 5 s period, starting with 2 s of silence, gives bursts at
    // 2-5, 7-10, 12-15 and 17-20 seconds.
    CHECK_EQ((int)s.labels.size(), 4);
    for (auto& l : s.labels) {
        CHECK_NEAR((double)(l.end - l.start) / s.samplerate, 3.0, 0.01);
    }
}

TEST(scene_cnr_is_approximately_as_requested) {
    sg::SceneConfig cfg;
    cfg.durationSeconds = 30.0;
    cfg.cnrDb = 20.0f;
    sg::Scene s = sg::generate(cfg);

    // Mean power inside the first label versus mean power in the leading silence.
    double onSum = 0.0, offSum = 0.0;
    long onN = 0, offN = 0;
    long onStart = s.labels[0].start, onEnd = s.labels[0].end;
    for (long i = 0; i < onStart; i++) {
        offSum += (double)s.iq[i].re * s.iq[i].re + (double)s.iq[i].im * s.iq[i].im;
        offN++;
    }
    for (long i = onStart; i < onEnd; i++) {
        onSum += (double)s.iq[i].re * s.iq[i].re + (double)s.iq[i].im * s.iq[i].im;
        onN++;
    }
    double onDb  = 10.0 * std::log10(onSum / (double)onN);
    double offDb = 10.0 * std::log10(offSum / (double)offN);
    // The syllabic envelope means the burst average sits several dB below its
    // peak, so a requested 20 dB peak CNR measures lower on average. This test
    // guards the scale, not an exact figure.
    double diff = onDb - offDb;
    CHECK(diff > 12.0);
    CHECK(diff < 21.0);
}

#include <dsp/noise_reduction/adaptive_squelch/frame_meter.h>
#include <dsp/noise_reduction/adaptive_squelch/noise_estimator.h>
#include <limits>

using dsp::noise_reduction::adaptive::FrameMeter;
using dsp::noise_reduction::adaptive::PercentileEstimator;

TEST(frame_meter_length_follows_samplerate) {
    FrameMeter m;
    m.init(48000.0, 0.010);
    CHECK_EQ(m.frameLength(), 480);
    m.setSamplerate(15000.0);
    CHECK_EQ(m.frameLength(), 150);
    m.setSamplerate(50000.0);
    CHECK_EQ(m.frameLength(), 500);
}

TEST(frame_meter_measures_known_power) {
    // A constant vector of magnitude 0.1 has mean power 0.01, i.e. -20 dB.
    std::vector<dsp::complex_t> in(1500, dsp::complex_t{ 0.1f, 0.0f });
    FrameMeter m;
    m.init(15000.0, 0.010);
    std::vector<float> frames;
    m.feed((int)in.size(), in.data(), frames);
    CHECK_EQ((int)frames.size(), 10);
    for (float f : frames) { CHECK_NEAR(f, -20.0, 0.001); }
}

TEST(frame_meter_output_is_independent_of_chunking) {
    sg::Rng rng(7);
    std::vector<dsp::complex_t> in(97531);
    for (auto& s : in) { s = dsp::complex_t{ rng.gaussian() * 0.01f, rng.gaussian() * 0.01f }; }

    FrameMeter a;
    a.init(15000.0, 0.010);
    std::vector<float> whole;
    a.feed((int)in.size(), in.data(), whole);

    // Same samples, fed in wildly uneven chunks including a single-sample call.
    FrameMeter b;
    b.init(15000.0, 0.010);
    std::vector<float> chunked;
    sg::Rng chunkRng(99);
    int pos = 0;
    while (pos < (int)in.size()) {
        int take = 1 + (int)(chunkRng.uniform() * 4096.0f);
        if (pos + take > (int)in.size()) { take = (int)in.size() - pos; }
        b.feed(take, in.data() + pos, chunked);
        pos += take;
    }

    CHECK_EQ(whole.size(), chunked.size());
    bool identical = whole.size() == chunked.size();
    for (size_t i = 0; identical && i < whole.size(); i++) {
        if (whole[i] != chunked[i]) { identical = false; }
    }
    CHECK(identical);
}

TEST(frame_meter_feed_sample_agrees_with_feed) {
    sg::Rng rng(21);
    std::vector<dsp::complex_t> in(4000);
    for (auto& s : in) { s = dsp::complex_t{ rng.gaussian() * 0.05f, rng.gaussian() * 0.05f }; }

    FrameMeter a;
    a.init(15000.0, 0.010);
    std::vector<float> bulk;
    a.feed((int)in.size(), in.data(), bulk);

    FrameMeter b;
    b.init(15000.0, 0.010);
    std::vector<float> single;
    float db = 0.0f;
    for (auto& s : in) {
        if (b.feedSample(s, db)) { single.push_back(db); }
    }

    CHECK_EQ(bulk.size(), single.size());
    bool identical = bulk.size() == single.size();
    for (size_t i = 0; identical && i < bulk.size(); i++) {
        if (bulk[i] != single[i]) { identical = false; }
    }
    CHECK(identical);
}

TEST(frame_meter_reset_clears_partial_frame) {
    std::vector<dsp::complex_t> in(100, dsp::complex_t{ 1.0f, 0.0f });
    FrameMeter m;
    m.init(15000.0, 0.010);   // 150-sample frames
    std::vector<float> frames;
    m.feed(100, in.data(), frames);      // 100 samples buffered, no frame emitted
    CHECK_EQ((int)frames.size(), 0);
    m.reset();
    m.feed(100, in.data(), frames);      // must not complete the earlier frame
    CHECK_EQ((int)frames.size(), 0);
}

TEST(percentile_bin_indices_are_clamped_at_both_ends) {
    // SDRPlusPlusBrown issue #73 is an out-of-bounds histogram index bug in
    // exactly this kind of code. Every input, however absurd, must land inside
    // the array.
    const float probes[] = {
        -1e30f, -1000.0f, -140.5f, -140.0f, -139.9f,
        -0.1f, 0.0f, 0.1f, 1000.0f, 1e30f,
        std::nanf(""), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()
    };
    for (float p : probes) {
        int b = PercentileEstimator::binOf(p);
        CHECK(b >= 0);
        CHECK(b < PercentileEstimator::BIN_COUNT);
    }
    CHECK_EQ(PercentileEstimator::BIN_COUNT, 560);
}

TEST(percentile_survives_out_of_range_sweep) {
    PercentileEstimator e;
    e.init(0.010);
    sg::Rng rng(11);
    for (int i = 0; i < 20000; i++) {
        // Deliberately sweeps far outside the configured dB range at both ends.
        float db = -400.0f + rng.uniform() * 800.0f;
        float n = e.update(db, false);
        CHECK(n >= PercentileEstimator::MIN_DB);
        CHECK(n <= PercentileEstimator::MAX_DB);
    }
}

TEST(percentile_finds_the_noise_floor_of_a_bimodal_stream) {
    // 70% of frames at -70 dB, 30% at -45 dB. The 20th percentile must report
    // the lower population.
    PercentileEstimator e;
    e.init(0.010);
    sg::Rng rng(3);
    float last = 0.0f;
    for (int i = 0; i < 5000; i++) {
        float db = (rng.uniform() < 0.7f) ? -70.0f : -45.0f;
        last = e.update(db, false);
    }
    CHECK_NEAR(last, -70.0, 0.5);
}

TEST(percentile_tracks_a_noise_floor_step_within_the_window) {
    PercentileEstimator e;
    e.init(0.010);
    e.setWindow(2.0);            // 200 frames
    for (int i = 0; i < 400; i++) { e.update(-80.0f, false); }
    CHECK_NEAR(e.update(-80.0f, false), -80.0, 0.5);
    // Step the floor up. After a full window the old population has expired.
    float n = 0.0f;
    for (int i = 0; i < 400; i++) { n = e.update(-60.0f, false); }
    CHECK_NEAR(n, -60.0, 0.5);
}

TEST(percentile_reports_occupancy) {
    PercentileEstimator e;
    e.init(0.010);
    sg::Rng rng(5);
    for (int i = 0; i < 5000; i++) {
        // 40% of frames well above the floor.
        e.update((rng.uniform() < 0.4f) ? -50.0f : -70.0f, false);
    }
    CHECK_NEAR(e.occupancy(), 0.40, 0.05);
}

TEST(percentile_reset_clears_history) {
    PercentileEstimator e;
    e.init(0.010);
    for (int i = 0; i < 2000; i++) { e.update(-90.0f, false); }
    e.reset();
    float n = 0.0f;
    for (int i = 0; i < 300; i++) { n = e.update(-40.0f, false); }
    CHECK_NEAR(n, -40.0, 0.5);
}

using dsp::noise_reduction::adaptive::McraEstimator;
using dsp::noise_reduction::adaptive::alphaFor;

TEST(alpha_for_matches_the_documented_defaults) {
    // Spec table: tau = 0.75 s at 10 ms frames gives alpha 0.9868;
    //             tau = 20 s at 10 ms frames gives alpha 0.99950.
    CHECK_NEAR(alphaFor(0.75, 0.010), 0.9868, 0.0002);
    CHECK_NEAR(alphaFor(20.0, 0.010), 0.99950, 0.00002);
}

TEST(mcra_converges_on_a_stationary_floor) {
    McraEstimator e;
    e.init(0.010);
    float n = 0.0f;
    for (int i = 0; i < 3000; i++) { n = e.update(-72.0f, false); }
    CHECK_NEAR(n, -72.0, 1.5);
}

TEST(mcra_does_not_freeze_during_a_long_transmission) {
    // The whole point of the soft update: the estimate slows during signal but
    // never stops. Feed 60 s of continuous signal after the floor steps up and
    // the estimate must still move most of the way.
    McraEstimator e;
    e.init(0.010);
    for (int i = 0; i < 3000; i++) { e.update(-90.0f, false); }
    float n = 0.0f;
    for (int i = 0; i < 6000; i++) { n = e.update(-60.0f, true); }
    CHECK(n > -75.0f);
}

TEST(mcra_rejects_a_single_impulse) {
    McraEstimator e;
    e.init(0.010);
    for (int i = 0; i < 3000; i++) { e.update(-80.0f, false); }
    float before = e.update(-80.0f, false);
    e.update(20.0f, false);              // one enormous impulse
    float after = e.update(-80.0f, false);
    CHECK_NEAR(after, before, 1.0);
}

TEST(mcra_reset_clears_state) {
    McraEstimator e;
    e.init(0.010);
    for (int i = 0; i < 3000; i++) { e.update(-100.0f, false); }
    e.reset();
    CHECK_NEAR(e.update(-40.0f, false), -40.0, 0.001);
}

using dsp::noise_reduction::adaptive::GatedEmaEstimator;

TEST(gated_ema_converges_while_the_gate_is_closed) {
    GatedEmaEstimator e;
    e.init(0.010);
    float n = 0.0f;
    for (int i = 0; i < 1000; i++) { n = e.update(-75.0f, false); }
    CHECK_NEAR(n, -75.0, 0.5);
}

TEST(gated_ema_goes_stale_while_the_gate_is_open) {
    // This is the documented staleness failure, asserted deliberately so the
    // behaviour is recorded rather than discovered later. The watchdog has not
    // fired yet at this point.
    GatedEmaEstimator e;
    e.init(0.010);
    for (int i = 0; i < 1000; i++) { e.update(-90.0f, false); }
    float frozen = 0.0f;
    for (int i = 0; i < 500; i++) { frozen = e.update(-50.0f, true); }
    CHECK_NEAR(frozen, -90.0, 0.5);
}

TEST(gated_ema_watchdog_recovers_from_a_long_open_gate) {
    // After WATCHDOG_SECONDS of continuous open the fast tracker is no longer
    // trustworthy and control passes to the shadow, which never freezes.
    GatedEmaEstimator e;
    e.init(0.010);
    for (int i = 0; i < 1000; i++) { e.update(-90.0f, false); }
    float n = 0.0f;
    const int frames = (int)(GatedEmaEstimator::WATCHDOG_SECONDS / 0.010) + 12000;
    for (int i = 0; i < frames; i++) { n = e.update(-50.0f, true); }
    CHECK(n > -85.0f);
}

TEST(gated_ema_follows_a_downward_step_while_closed) {
    GatedEmaEstimator e;
    e.init(0.010);
    for (int i = 0; i < 5000; i++) { e.update(-40.0f, false); }
    float n = 0.0f;
    for (int i = 0; i < 500; i++) { n = e.update(-95.0f, false); }
    CHECK(n < -85.0f);
}

TEST(gated_ema_reset_clears_state) {
    GatedEmaEstimator e;
    e.init(0.010);
    for (int i = 0; i < 2000; i++) { e.update(-100.0f, false); }
    e.reset();
    CHECK_NEAR(e.update(-30.0f, false), -30.0, 0.001);
}

#include <dsp/noise_reduction/adaptive_squelch/preroll.h>

using dsp::noise_reduction::adaptive::PreRoll;

TEST(preroll_length_follows_samplerate) {
    PreRoll p;
    p.init(48000.0, 0.040);
    CHECK_EQ(p.delaySamples(), 1920);
    p.setLength(15000.0, 0.040);
    CHECK_EQ(p.delaySamples(), 600);
}

TEST(preroll_delays_by_exactly_the_requested_samples) {
    PreRoll p;
    p.init(1000.0, 0.005);        // 5 samples
    CHECK_EQ(p.delaySamples(), 5);

    dsp::complex_t out;
    // The first 5 pushes prime the line and produce nothing usable.
    for (int i = 0; i < 5; i++) {
        CHECK(!p.push(dsp::complex_t{ (float)i, 0.0f }, out));
    }
    // From here every push returns the sample from 5 pushes ago.
    for (int i = 5; i < 50; i++) {
        CHECK(p.push(dsp::complex_t{ (float)i, 0.0f }, out));
        CHECK_NEAR(out.re, (double)(i - 5), 1e-6);
    }
}

TEST(preroll_zero_length_is_a_passthrough) {
    PreRoll p;
    p.init(48000.0, 0.0);
    CHECK_EQ(p.delaySamples(), 0);
    dsp::complex_t out;
    CHECK(p.push(dsp::complex_t{ 3.0f, 4.0f }, out));
    CHECK_NEAR(out.re, 3.0, 1e-6);
    CHECK_NEAR(out.im, 4.0, 1e-6);
}

TEST(preroll_reset_re_primes) {
    PreRoll p;
    p.init(1000.0, 0.005);
    dsp::complex_t out;
    for (int i = 0; i < 20; i++) { p.push(dsp::complex_t{ (float)i, 0.0f }, out); }
    p.reset();
    for (int i = 0; i < 5; i++) {
        CHECK(!p.push(dsp::complex_t{ 99.0f, 0.0f }, out));
    }
    CHECK(p.push(dsp::complex_t{ 99.0f, 0.0f }, out));
    CHECK_NEAR(out.re, 99.0, 1e-6);
}

#include <dsp/noise_reduction/adaptive_squelch/gate.h>

using dsp::noise_reduction::adaptive::Gate;
using dsp::noise_reduction::adaptive::GATE_CLOSED;
using dsp::noise_reduction::adaptive::GATE_QUALIFY;
using dsp::noise_reduction::adaptive::GATE_OPEN;
using dsp::noise_reduction::adaptive::GATE_HANG;

// Advances the gate by n frames at a fixed SNR, running the per-sample ramp
// for a whole frame each time so gain and state stay in step.
static void runFrames(Gate& g, float snrDb, int frames, int samplesPerFrame) {
    for (int f = 0; f < frames; f++) {
        g.updateFrame(snrDb);
        for (int s = 0; s < samplesPerFrame; s++) { g.nextGain(); }
    }
}

TEST(gate_starts_closed_and_silent) {
    Gate g;
    g.init(15000.0, 0.010);
    CHECK(g.state() == GATE_CLOSED);
    CHECK(!g.isOpen());
    CHECK_NEAR(g.nextGain(), 0.0, 1e-6);
}

TEST(gate_requires_the_qualify_period_before_opening) {
    Gate g;
    g.init(15000.0, 0.010);
    g.setMargins(9.5f, 6.5f);
    g.setQualify(0.030);            // 3 frames

    g.updateFrame(12.0f);
    CHECK(g.state() == GATE_QUALIFY);
    g.updateFrame(12.0f);
    CHECK(g.state() == GATE_QUALIFY);
    g.updateFrame(12.0f);
    CHECK(g.state() == GATE_OPEN);
}

TEST(gate_rejects_a_single_frame_impulse) {
    Gate g;
    g.init(15000.0, 0.010);
    g.setMargins(9.5f, 6.5f);
    g.setQualify(0.030);

    g.updateFrame(30.0f);           // one loud frame
    CHECK(g.state() == GATE_QUALIFY);
    g.updateFrame(0.0f);            // and it is gone
    CHECK(g.state() == GATE_CLOSED);
    CHECK(!g.isOpen());
}

TEST(gate_hysteresis_holds_open_between_the_two_thresholds) {
    Gate g;
    g.init(15000.0, 0.010);
    g.setMargins(9.5f, 6.5f);
    g.setQualify(0.030);
    runFrames(g, 12.0f, 5, 150);
    CHECK(g.state() == GATE_OPEN);

    // 8 dB is below the open threshold but above the close threshold.
    runFrames(g, 8.0f, 50, 150);
    CHECK(g.state() == GATE_OPEN);

    runFrames(g, 5.0f, 1, 150);
    CHECK(g.state() == GATE_HANG);
}

TEST(gate_hang_holds_then_closes) {
    Gate g;
    g.init(15000.0, 0.010);
    g.setMargins(9.5f, 6.5f);
    g.setQualify(0.030);
    g.setHang(0.200);
    g.setRelease(0.150);
    runFrames(g, 12.0f, 5, 150);
    CHECK(g.state() == GATE_OPEN);

    runFrames(g, 0.0f, 20, 150);    // 200 ms: still holding
    CHECK(g.isOpen());
    runFrames(g, 0.0f, 20, 150);    // 350 ms total: closed
    CHECK(g.state() == GATE_CLOSED);
    CHECK(!g.isOpen());
}

TEST(gate_hang_reopens_without_re_qualifying) {
    Gate g;
    g.init(15000.0, 0.010);
    g.setMargins(9.5f, 6.5f);
    g.setQualify(0.030);
    g.setHang(0.200);
    runFrames(g, 12.0f, 5, 150);
    runFrames(g, 0.0f, 5, 150);
    CHECK(g.state() == GATE_HANG);
    g.updateFrame(12.0f);
    CHECK(g.state() == GATE_OPEN);
}

TEST(gate_ramp_is_monotonic_and_reaches_both_endpoints) {
    Gate g;
    g.init(1000.0, 0.010);          // 10 samples per frame
    g.setMargins(9.5f, 6.5f);
    g.setQualify(0.010);            // 1 frame
    g.setRamp(0.020);               // 20 samples

    g.updateFrame(12.0f);
    CHECK(g.state() == GATE_OPEN);

    float prev = -1.0f;
    bool monotonic = true;
    for (int i = 0; i < 20; i++) {
        float gain = g.nextGain();
        if (gain < prev - 1e-6f) { monotonic = false; }
        prev = gain;
    }
    CHECK(monotonic);
    CHECK_NEAR(prev, 1.0, 1e-4);
    CHECK_NEAR(g.nextGain(), 1.0, 1e-6);
}

TEST(gate_ramp_has_no_discontinuity_at_the_start_of_a_transition) {
    // A raised cosine starts with zero slope, which is the property that
    // removes the click. The first ramp step must be far smaller than a
    // uniform step of the same length would be.
    Gate g;
    g.init(1000.0, 0.010);
    g.setMargins(9.5f, 6.5f);
    g.setQualify(0.010);
    g.setRamp(0.100);               // 100 samples
    g.updateFrame(12.0f);

    float first = g.nextGain();
    CHECK(first < (1.0f / 100.0f));
}

TEST(gate_reset_returns_to_closed_and_silent) {
    Gate g;
    g.init(15000.0, 0.010);
    g.setMargins(9.5f, 6.5f);
    g.setQualify(0.010);
    runFrames(g, 20.0f, 20, 150);
    CHECK(g.isOpen());
    g.reset();
    CHECK(g.state() == GATE_CLOSED);
    CHECK_NEAR(g.nextGain(), 0.0, 1e-6);
}

int main() {
    return tf::runAll();
}
