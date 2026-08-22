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

#include <dsp/noise_reduction/adaptive_squelch/diagnostics.h>
#include <thread>
#include <atomic>

using dsp::noise_reduction::adaptive::Diagnostics;
using dsp::noise_reduction::adaptive::DiagnosticsPublisher;

TEST(diagnostics_round_trips) {
    DiagnosticsPublisher p;
    Diagnostics d;
    d.framePowerDb = -61.5f;
    d.noiseFloorDb = -72.25f;
    d.snrDb = 10.75f;
    d.occupancy = 0.31f;
    d.state = (int)GATE_OPEN;
    d.lastTransitionReason = 3;
    p.publish(d);

    Diagnostics r = p.read();
    CHECK_NEAR(r.framePowerDb, -61.5, 1e-6);
    CHECK_NEAR(r.noiseFloorDb, -72.25, 1e-6);
    CHECK_NEAR(r.snrDb, 10.75, 1e-6);
    CHECK_NEAR(r.occupancy, 0.31, 1e-6);
    CHECK_EQ(r.state, (int)GATE_OPEN);
    CHECK_EQ(r.lastTransitionReason, 3);
}

TEST(diagnostics_reader_never_observes_a_torn_snapshot) {
    // The writer publishes internally consistent snapshots where snrDb is always
    // exactly framePowerDb minus noiseFloorDb. A torn read would break that.
    DiagnosticsPublisher p;
    std::atomic<bool> stop{ false };
    std::atomic<int> torn{ 0 };

    std::thread writer([&]() {
        for (int i = 0; i < 200000 && !stop.load(); i++) {
            Diagnostics d;
            d.framePowerDb = (float)(-100 + (i % 50));
            d.noiseFloorDb = (float)(-120 + (i % 50));
            d.snrDb = d.framePowerDb - d.noiseFloorDb;
            d.state = i % 4;
            p.publish(d);
        }
        stop.store(true);
    });

    while (!stop.load()) {
        Diagnostics r = p.read();
        if (std::fabs((r.framePowerDb - r.noiseFloorDb) - r.snrDb) > 1e-3f) {
            torn.fetch_add(1);
        }
    }
    writer.join();
    CHECK_EQ(torn.load(), 0);
}

#include <dsp/noise_reduction/adaptive_squelch.h>
#include <dsp/noise_reduction/power_squelch.h>

using dsp::noise_reduction::AdaptiveSquelch;
using dsp::noise_reduction::adaptive::ESTIMATOR_PERCENTILE;
using dsp::noise_reduction::adaptive::ESTIMATOR_MCRA;
using dsp::noise_reduction::adaptive::ESTIMATOR_GATED_EMA;

// Runs a whole scene through the block in one call and returns the output.
static std::vector<dsp::complex_t> runScene(AdaptiveSquelch& sq, const sg::Scene& scene) {
    std::vector<dsp::complex_t> out(scene.iq.size());
    int n = sq.process((int)scene.iq.size(), scene.iq.data(), out.data());
    out.resize((size_t)n);
    return out;
}

TEST(block_mutes_a_noise_only_scene) {
    sg::SceneConfig cfg;
    cfg.durationSeconds = 20.0;
    cfg.cnrDb = -100.0f;               // effectively no transmission at all
    sg::Scene scene = sg::generate(cfg);

    AdaptiveSquelch sq;
    sq.init(NULL, cfg.samplerate);
    sq.setEstimator(ESTIMATOR_PERCENTILE);
    std::vector<dsp::complex_t> out = runScene(sq, scene);

    // Ignore the first 12 s while the 10 s window fills.
    const long skip = (long)(12.0 * cfg.samplerate);
    double energy = 0.0;
    for (size_t i = (size_t)skip; i < out.size(); i++) {
        energy += (double)out[i].re * out[i].re + (double)out[i].im * out[i].im;
    }
    CHECK_NEAR(energy, 0.0, 1e-9);
}

TEST(block_passes_a_strong_transmission) {
    sg::SceneConfig cfg;
    cfg.durationSeconds = 40.0;
    cfg.cnrDb = 25.0f;
    sg::Scene scene = sg::generate(cfg);

    AdaptiveSquelch sq;
    sq.init(NULL, cfg.samplerate);
    sq.setEstimator(ESTIMATOR_PERCENTILE);
    std::vector<dsp::complex_t> out = runScene(sq, scene);

    // Take a label well after the window has filled and check that most of its
    // energy survived. Output index j holds input sample j: the block emits the
    // pre-rolled sample, so no realignment is needed.
    const sg::Label& lbl = scene.labels.back();
    double inE = 0.0, outE = 0.0;
    for (long i = lbl.start; i < lbl.end; i++) {
        inE += (double)scene.iq[(size_t)i].re * scene.iq[(size_t)i].re
             + (double)scene.iq[(size_t)i].im * scene.iq[(size_t)i].im;
        if (i < (long)out.size()) {
            outE += (double)out[(size_t)i].re * out[(size_t)i].re
                  + (double)out[(size_t)i].im * out[(size_t)i].im;
        }
    }
    CHECK(outE > 0.7 * inE);
}

TEST(block_output_is_independent_of_chunking) {
    // The acceptance criterion for the variable-count constraint, at block level.
    sg::SceneConfig cfg;
    cfg.durationSeconds = 25.0;
    cfg.cnrDb = 15.0f;
    sg::Scene scene = sg::generate(cfg);

    AdaptiveSquelch a;
    a.init(NULL, cfg.samplerate);
    std::vector<dsp::complex_t> whole(scene.iq.size());
    int nWhole = a.process((int)scene.iq.size(), scene.iq.data(), whole.data());
    whole.resize((size_t)nWhole);

    AdaptiveSquelch b;
    b.init(NULL, cfg.samplerate);
    std::vector<dsp::complex_t> chunked;
    std::vector<dsp::complex_t> scratch(65536);
    sg::Rng chunkRng(1234);
    int pos = 0;
    while (pos < (int)scene.iq.size()) {
        int take = 1 + (int)(chunkRng.uniform() * 8192.0f);
        if (pos + take > (int)scene.iq.size()) { take = (int)scene.iq.size() - pos; }
        int n = b.process(take, scene.iq.data() + pos, scratch.data());
        chunked.insert(chunked.end(), scratch.begin(), scratch.begin() + n);
        pos += take;
    }

    CHECK_EQ(whole.size(), chunked.size());
    bool identical = whole.size() == chunked.size();
    for (size_t i = 0; identical && i < whole.size(); i++) {
        if (whole[i].re != chunked[i].re || whole[i].im != chunked[i].im) { identical = false; }
    }
    CHECK(identical);
}

TEST(block_timing_scales_across_samplerates) {
    // The same scene in wall-clock seconds at three samplerates must produce
    // gate transitions at the same times, not at the same sample indices.
    const double rates[] = { 15000.0, 24000.0, 50000.0 };
    double openTimes[3] = { 0.0, 0.0, 0.0 };

    for (int r = 0; r < 3; r++) {
        sg::SceneConfig cfg;
        cfg.samplerate = rates[r];
        cfg.durationSeconds = 30.0;
        cfg.cnrDb = 20.0f;
        sg::Scene scene = sg::generate(cfg);

        AdaptiveSquelch sq;
        sq.init(NULL, cfg.samplerate);
        std::vector<dsp::complex_t> out(scene.iq.size());

        // Feed in 10 ms slices and record when the gate first opens after 15 s.
        const int slice = (int)std::lround(0.010 * cfg.samplerate);
        for (int pos = 0; pos + slice <= (int)scene.iq.size(); pos += slice) {
            sq.process(slice, scene.iq.data() + pos, out.data());
            double t = (double)pos / cfg.samplerate;
            if (t > 15.0 && openTimes[r] == 0.0 && sq.diagnostics().state != 0) {
                openTimes[r] = t;
            }
        }
        CHECK(openTimes[r] > 15.0);
    }
    CHECK_NEAR(openTimes[0], openTimes[1], 0.15);
    CHECK_NEAR(openTimes[1], openTimes[2], 0.15);
}

TEST(block_all_three_estimators_run) {
    const dsp::noise_reduction::adaptive::EstimatorType types[] = {
        ESTIMATOR_PERCENTILE, ESTIMATOR_MCRA, ESTIMATOR_GATED_EMA
    };
    sg::SceneConfig cfg;
    cfg.durationSeconds = 25.0;
    sg::Scene scene = sg::generate(cfg);

    for (auto t : types) {
        AdaptiveSquelch sq;
        sq.init(NULL, cfg.samplerate);
        sq.setEstimator(t);
        std::vector<dsp::complex_t> out = runScene(sq, scene);
        CHECK(out.size() > 0);
        dsp::noise_reduction::adaptive::Diagnostics d = sq.diagnostics();
        CHECK(d.noiseFloorDb > -140.0f);
        CHECK(d.noiseFloorDb < 0.0f);
    }
}

TEST(block_invalidation_paths_clear_state) {
    sg::SceneConfig cfg;
    cfg.durationSeconds = 25.0;
    cfg.cnrDb = 25.0f;
    sg::Scene scene = sg::generate(cfg);

    AdaptiveSquelch sq;
    sq.init(NULL, cfg.samplerate);
    runScene(sq, scene);
    CHECK(sq.diagnostics().noiseFloorDb > -140.0f);

    sq.reset();
    CHECK_EQ(sq.diagnostics().state, 0);

    // A samplerate change must also invalidate, not just resize.
    runScene(sq, scene);
    sq.setSamplerate(48000.0);
    CHECK_EQ(sq.diagnostics().state, 0);
}

TEST(block_switching_artifact_is_smaller_than_power_squelch) {
    // Acceptance criterion 6: the ramp must measurably beat the abrupt switch.
    // Measured as the largest single-sample jump in output magnitude.
    sg::SceneConfig cfg;
    cfg.durationSeconds = 40.0;
    cfg.cnrDb = 25.0f;
    sg::Scene scene = sg::generate(cfg);

    AdaptiveSquelch sq;
    sq.init(NULL, cfg.samplerate);
    std::vector<dsp::complex_t> adaptive = runScene(sq, scene);

    // power_squelch.h thresholds on 10*log10(mean|x|) over one process() block,
    // not on mean power, so both its level and its block size have to be
    // realistic or it never switches at all and the comparison is vacuous.
    auto meanMagDb = [&](long a, long b) {
        double sum = 0.0;
        for (long i = a; i < b; i++) {
            sum += std::sqrt((double)scene.iq[(size_t)i].re * scene.iq[(size_t)i].re
                           + (double)scene.iq[(size_t)i].im * scene.iq[(size_t)i].im);
        }
        return 10.0 * std::log10(sum / (double)(b - a));
    };
    const double idleDb = meanMagDb(0, scene.labels[0].start);
    const double activeDb = meanMagDb(scene.labels[0].start, scene.labels[0].end);

    dsp::noise_reduction::PowerSquelch ps;
    ps.init(NULL, 0.5 * (idleDb + activeDb));
    std::vector<dsp::complex_t> naive(scene.iq.size());
    const int blockLen = (int)std::lround(0.010 * cfg.samplerate);
    long pos = 0;
    while (pos + blockLen <= (long)scene.iq.size()) {
        ps.process(blockLen, scene.iq.data() + pos, naive.data() + pos);
        pos += blockLen;
    }
    naive.resize((size_t)pos);

    auto maxJump = [](const std::vector<dsp::complex_t>& v) {
        double worst = 0.0;
        for (size_t i = 1; i < v.size(); i++) {
            double a = std::sqrt((double)v[i - 1].re * v[i - 1].re + (double)v[i - 1].im * v[i - 1].im);
            double b = std::sqrt((double)v[i].re * v[i].re + (double)v[i].im * v[i].im);
            double d = std::fabs(b - a);
            if (d > worst) { worst = d; }
        }
        return worst;
    };
    CHECK(maxJump(adaptive) < maxJump(naive));
}

#include "scorer.h"

TEST(scorer_reports_a_perfect_gate_as_perfect) {
    const double sr = 1000.0;
    std::vector<char> open(10000, 0);
    std::vector<sg::Label> labels{ { 2000, 5000 } };
    for (long i = 2000; i < 5000; i++) { open[(size_t)i] = 1; }

    scorer::Score s = scorer::score(open, labels, sr);
    CHECK_NEAR(s.falseOpenFrac, 0.0, 1e-9);
    CHECK_NEAR(s.missFrac, 0.0, 1e-9);
    CHECK_NEAR(s.fecMs, 0.0, 1e-6);
    CHECK_NEAR(s.overMs, 0.0, 1e-6);
    CHECK_EQ(s.mscCount, 0);
}

TEST(scorer_measures_front_end_clipping_and_overhang) {
    const double sr = 1000.0;
    std::vector<char> open(10000, 0);
    std::vector<sg::Label> labels{ { 2000, 5000 } };
    // Opens 100 samples late, closes 250 samples late.
    for (long i = 2100; i < 5250; i++) { open[(size_t)i] = 1; }

    scorer::Score s = scorer::score(open, labels, sr);
    CHECK_NEAR(s.fecMs, 100.0, 1e-6);
    CHECK_NEAR(s.overMs, 250.0, 1e-6);
}

TEST(scorer_counts_mid_speech_clipping) {
    const double sr = 1000.0;
    std::vector<char> open(10000, 0);
    std::vector<sg::Label> labels{ { 2000, 5000 } };
    for (long i = 2000; i < 5000; i++) { open[(size_t)i] = 1; }
    for (long i = 3000; i < 3100; i++) { open[(size_t)i] = 0; }
    for (long i = 4000; i < 4050; i++) { open[(size_t)i] = 0; }

    scorer::Score s = scorer::score(open, labels, sr);
    CHECK_EQ(s.mscCount, 2);
}

TEST(preroll_measurably_reduces_front_end_clipping) {
    // Acceptance criterion 7. Scored through the same path the sweep uses, so
    // the number means the same thing here as it does in the comparison table.
    sg::SceneConfig cfg;
    cfg.durationSeconds = 60.0;
    cfg.cnrDb = 15.0f;
    sg::Scene scene = sg::generate(cfg);

    auto measureFec = [&](double preRollSeconds) {
        AdaptiveSquelch sq;
        sq.init(NULL, cfg.samplerate);
        sq.setPreRoll(preRollSeconds);

        std::vector<char> open;
        open.reserve(scene.iq.size());
        std::vector<dsp::complex_t> scratch(65536);
        long pos = 0;
        while (pos < (long)scene.iq.size()) {
            int take = 4096;
            if (pos + take > (long)scene.iq.size()) { take = (int)((long)scene.iq.size() - pos); }
            int n = sq.process(take, scene.iq.data() + pos, scratch.data());
            int st = sq.diagnostics().state;
            char isOpen = (st == (int)dsp::noise_reduction::adaptive::GATE_OPEN ||
                           st == (int)dsp::noise_reduction::adaptive::GATE_HANG) ? 1 : 0;
            for (int i = 0; i < n; i++) { open.push_back(isOpen); }
            pos += take;
        }
        return scorer::score(open, scene.labels, cfg.samplerate).fecMs;
    };

    double withPreRoll = measureFec(0.040);
    double without = measureFec(0.0);
    CHECK(withPreRoll < without);
}

TEST(scorer_measures_false_open_time) {
    const double sr = 1000.0;
    std::vector<char> open(10000, 0);
    std::vector<sg::Label> labels{ { 2000, 5000 } };
    for (long i = 2000; i < 5000; i++) { open[(size_t)i] = 1; }
    for (long i = 7000; i < 7700; i++) { open[(size_t)i] = 1; }   // 700 of 7000 idle samples

    scorer::Score s = scorer::score(open, labels, sr);
    CHECK_NEAR(s.falseOpenFrac, 0.10, 1e-6);
}

#include "wav_io.h"

TEST(wav_pcm16_round_trip_is_byte_exact) {
    // Known vector spanning most of the PCM16 range. Kept inside +/-0.95:
    // this test measures quantization error, not clipping behaviour, which
    // has its own check below. Worst-case round-trip error through
    // round(v*32767)/32768 is 1.5/32768 near full scale.
    std::vector<float> in(1000);
    sg::Rng rng(5);
    for (auto& v : in) {
        do { v = rng.gaussian() * 0.4f; } while (v < -0.95f || v > 0.95f);
    }
    in[0] = 0.0f; in[1] = 0.9999f; in[2] = -0.9999f;

    const char* path = "/tmp/squelch_wav_test_pcm16.wav";
    CHECK(ag::writePcm16(path, in, 48000));

    ag::WavData out = ag::read(path);
    CHECK(out.ok);
    CHECK_EQ((int)out.samples.size(), (int)in.size());
    CHECK_EQ(out.rate, 48000);
    for (size_t i = 0; i < in.size(); i++) {
        CHECK_NEAR(out.samples[i], in[i], 1.6f / 32768.0f);
    }
    std::remove(path);
}

TEST(wav_pcm16_clips_out_of_range_inputs) {
    // Out-of-range values are clamped, never wrapped or left unguarded.
    std::vector<float> in{ 0.0f, 2.0f, -2.0f };
    const char* path = "/tmp/squelch_wav_test_clip.wav";
    CHECK(ag::writePcm16(path, in, 8000));

    ag::WavData out = ag::read(path);
    CHECK(out.ok);
    CHECK_EQ((int)out.samples.size(), 3);
    CHECK_NEAR(out.samples[0], 0.0, 1e-9);
    CHECK_NEAR(out.samples[1], 32767.0 / 32768.0, 1e-9);
    CHECK_NEAR(out.samples[2], -32767.0 / 32768.0, 1e-9);
    std::remove(path);
}

TEST(wav_float_round_trip_is_exact) {
    std::vector<float> in(777);
    sg::Rng rng(9);
    for (auto& v : in) { v = rng.gaussian() * 0.7f; }

    const char* path = "/tmp/squelch_wav_test_f32.wav";
    CHECK(ag::writeFloat32(path, in, 44100));

    ag::WavData out = ag::read(path);
    CHECK(out.ok);
    CHECK_EQ((int)out.samples.size(), (int)in.size());
    CHECK_EQ(out.rate, 44100);
    bool exact = true;
    for (size_t i = 0; exact && i < in.size(); i++) {
        if (out.samples[i] != in[i]) { exact = false; }
    }
    CHECK(exact);
    std::remove(path);
}

TEST(wav_read_rejects_truncated_file) {
    const char* path = "/tmp/squelch_wav_test_trunc.wav";
    std::vector<float> in(500, 0.25f);
    CHECK(ag::writePcm16(path, in, 16000));

    // Keep only the RIFF header plus a few bytes: the data chunk claims far
    // more payload than the file holds.
    {
        std::FILE* f = std::fopen(path, "rb");
        CHECK(f != NULL);
        std::vector<char> head(60);
        size_t n = std::fread(head.data(), 1, head.size(), f);
        std::fclose(f);
        head.resize(n);

        std::FILE* g = std::fopen(path, "wb");
        std::fwrite(head.data(), 1, head.size(), g);
        std::fclose(g);
    }

    ag::WavData out = ag::read(path);
    CHECK(!out.ok);
    CHECK(!out.err.empty());
    std::remove(path);
}

#include "audio_gen.h"

TEST(audio_scene_is_deterministic) {
    ag::AudioSceneConfig cfg;
    cfg.durationSeconds = 3.0;
    ag::AudioScene a = ag::generate(cfg);
    ag::AudioScene b = ag::generate(cfg);
    CHECK_EQ((int)a.clean.size(), (int)b.clean.size());
    bool identical = a.clean.size() == b.clean.size();
    for (size_t i = 0; identical && i < a.clean.size(); i++) {
        if (a.clean[i] != b.clean[i] || a.noisy[i] != b.noisy[i]) { identical = false; }
    }
    CHECK(identical);
}

TEST(audio_clean_is_silent_outside_transmissions) {
    // The clean signal must be exactly zero in the gaps: the ground truth for
    // "noise only" depends on it.
    ag::AudioSceneConfig cfg;
    cfg.durationSeconds = 20.0;
    ag::AudioScene s = ag::generate(cfg);
    std::vector<char> active((size_t)s.clean.size(), 0);
    for (auto& l : s.txLabels) {
        for (long i = l.start; i < l.end && i < (long)active.size(); i++) { active[(size_t)i] = 1; }
    }
    for (size_t i = 0; i < s.clean.size(); i++) {
        if (!active[i]) { CHECK_EQ(s.clean[i], 0.0f); }
    }
}

TEST(audio_measured_active_snr_matches_request) {
    // With speech only inside word labels and noise everywhere, an estimate
    // of active SNR is P(word regions) - P(gap regions) over P(gap regions).
    ag::AudioSceneConfig cfg;
    cfg.durationSeconds = 30.0;
    cfg.snrDb = 12.0f;
    ag::AudioScene s = ag::generate(cfg);

    double wordSum = 0.0, gapSum = 0.0;
    long wordN = 0, gapN = 0;
    std::vector<char> word((size_t)s.noisy.size(), 0);
    for (auto& l : s.wordLabels) {
        for (long i = l.start; i < l.end && i < (long)word.size(); i++) { word[(size_t)i] = 1; }
    }
    for (size_t i = 0; i < s.noisy.size(); i++) {
        double p = (double)s.noisy[i] * s.noisy[i];
        if (word[i]) { wordSum += p; wordN++; }
        else { gapSum += p; gapN++; }
    }
    CHECK(gapSum > 0.0);
    double snrDb = 10.0 * std::log10((wordSum - gapSum) / gapSum);
    CHECK_NEAR(snrDb, 12.0, 1.5);
}

TEST(audio_word_labels_nest_inside_transmissions) {
    ag::AudioSceneConfig cfg;
    cfg.durationSeconds = 25.0;
    cfg.wordsPerTx = 4;
    ag::AudioScene s = ag::generate(cfg);

    CHECK_EQ((int)s.wordLabels.size(), (int)s.txLabels.size() * 4);
    for (auto& w : s.wordLabels) {
        bool inside = false;
        for (auto& t : s.txLabels) {
            if (w.start >= t.start && w.end <= t.end) { inside = true; break; }
        }
        CHECK(inside);
        // Words are at least 50 ms so they carry something to transcribe.
        CHECK((w.end - w.start) >= (long)(0.05 * s.rate));
    }
}

#include "audio_metrics.h"

TEST(metrics_scores_a_known_noise_reduction_exactly) {
    // Out is exactly half the amplitude of In everywhere outside labels:
    // a -6.02 dB reduction in the noise-only regions.
    const long n = 48000;
    std::vector<float> in((size_t)n), out((size_t)n);
    sg::Rng rng(13);
    for (long i = 0; i < n; i++) {
        in[(size_t)i] = rng.gaussian() * 0.05f;
        out[(size_t)i] = 0.5f * in[(size_t)i];
    }
    std::vector<sg::Label> labels{ { 16000, 32000 } };   // one "transmission"

    double db = metrics::noiseReductionDb(in, out, labels, 48000.0, 0.05);
    CHECK_NEAR(db, -6.0206, 0.25);
}

TEST(metrics_fidelity_compensates_reported_delay) {
    const long n = 48000;
    const int delay = 456;
    std::vector<float> clean((size_t)n, 0.0f);
    sg::Rng rng(17);
    // Speech-ish energy inside the transmission only.
    for (long i = 16000; i < 32000; i++) { clean[(size_t)i] = rng.gaussian() * 0.1f; }
    std::vector<float> out((size_t)n, 0.0f);
    for (long i = 0; i + delay < n; i++) { out[(size_t)(i + delay)] = 0.891f * clean[(size_t)i]; }

    std::vector<sg::Label> labels{ { 16000, 32000 } };
    double db = metrics::inBandFidelityDb(clean, out, labels, delay);
    CHECK_NEAR(db, -1.0, 0.15);
}

TEST(metrics_jump_metric_catches_a_planted_click) {
    const long n = 48000;
    std::vector<float> v((size_t)n);
    for (long i = 0; i < n; i++) { v[(size_t)i] = 0.01f * std::sin(0.01 * i); }
    float base = metrics::maxSampleJump(v);
    CHECK(base < 0.001f);

    v[24000] += 0.5f;
    float planted = metrics::maxSampleJump(v);
    CHECK(planted > 0.4f);
}

TEST(export_triplet_round_trips_through_wav_io) {
    const double rate = 48000.0;
    std::vector<float> v(480, 0.125f);
    ag::AudioScene dummy;
    dummy.rate = rate;

    bool ok = metrics::exportWavTriplet("/tmp/squelch_tri", rate, v, v, v);
    CHECK(ok);

    ag::WavData r = ag::read("/tmp/squelch_tri_ref.wav");
    ag::WavData d = ag::read("/tmp/squelch_tri_deg.wav");
    ag::WavData p = ag::read("/tmp/squelch_tri_proc.wav");
    CHECK(r.ok && d.ok && p.ok);
    CHECK_EQ((int)r.samples.size(), 480);
    std::remove("/tmp/squelch_tri_ref.wav");
    std::remove("/tmp/squelch_tri_deg.wav");
    std::remove("/tmp/squelch_tri_proc.wav");
}

#include <dsp/audio/voice_bandpass.h>

using dsp::audio::VoiceBandpass;

// Feeds a steady tone through the bandpass in one call, discards the filter
// transient plus its group delay, and returns the gain over the known input
// amplitude.
static double bpToneGainDb(VoiceBandpass& bp, double fs, double freqHz, double amp = 0.5) {
    const long n = (long)(fs * 1.2);
    std::vector<dsp::stereo_t> in((size_t)n);
    for (long i = 0; i < n; i++) {
        float v = amp * std::sin(6.283185307179586 * freqHz * (double)i / fs);
        in[(size_t)i] = dsp::stereo_t{ v, v };
    }
    std::vector<dsp::stereo_t> out((size_t)n);
    bp.process((int)n, in.data(), out.data());

    const long skip = bp.delaySamples() + (long)bp.tapCount() + 64;
    double sum = 0.0;
    long cnt = 0;
    for (long i = skip; i + 8 < n; i++) {
        sum += (double)out[(size_t)i].l * out[(size_t)i].l;
        cnt++;
    }
    double outRms = std::sqrt(sum / (double)cnt);
    double inRms = amp / std::sqrt(2.0);
    return 20.0 * std::log10(outRms / inRms);
}

TEST(bandpass_passband_is_flat_within_1_db) {
    const double probes[] = { 500.0, 900.0, 1500.0, 2100.0 };
    for (double f : probes) {
        VoiceBandpass bp;
        bp.init(NULL, 48000.0);
        double g = bpToneGainDb(bp, 48000.0, f);
        CHECK_NEAR(g, 0.0, 1.0);
    }
}

TEST(bandpass_rejects_the_stopbands_by_40_db) {
    const double rejects[] = { 150.0, 3100.0, 3400.0 };
    for (double f : rejects) {
        VoiceBandpass bp;
        bp.init(NULL, 48000.0);
        CHECK(bpToneGainDb(bp, 48000.0, f) <= -40.0);
    }
}

TEST(bandpass_impulse_response_is_centered_on_the_reported_delay) {
    VoiceBandpass bp;
    bp.init(NULL, 48000.0);
    // Linear-phase windowed sinc: odd tap count, integer group delay.
    CHECK_EQ(bp.tapCount() % 2, 1);
    CHECK_EQ(bp.delaySamples(), (bp.tapCount() - 1) / 2);

    const int n = bp.delaySamples() * 4 + 256;
    std::vector<dsp::stereo_t> in((size_t)n, dsp::stereo_t{ 0.0f, 0.0f });
    in[(size_t)bp.delaySamples()] = dsp::stereo_t{ 1.0f, 1.0f };
    std::vector<dsp::stereo_t> out((size_t)n);
    bp.process(n, in.data(), out.data());

    // The magnitude peak of a modulated sinc need not sit on the group-delay
    // sample - the carrier can cross zero there. The exact invariant of a
    // symmetric Type-I response is its energy centroid.
    double num = 0.0, den = 0.0;
    float peakMag = 0.0f;
    for (int i = 0; i < n; i++) {
        double m = (double)out[(size_t)i].l * out[(size_t)i].l;
        num += (double)i * m;
        den += m;
        float a = std::fabs(out[(size_t)i].l);
        if (a > peakMag) { peakMag = a; }
    }
    CHECK(den > 0.0);
    CHECK_NEAR(num / den, 2.0 * bp.delaySamples(), 0.01);   // input delay + group delay
    CHECK_NEAR(peakMag, bp.tapsPeak(), 1e-3);
}

TEST(bandpass_output_is_independent_of_chunking) {
    sg::Rng rng(31);
    const long n = 200000;
    std::vector<dsp::stereo_t> in((size_t)n);
    for (auto& s : in) { s = dsp::stereo_t{ rng.gaussian() * 0.05f, rng.gaussian() * 0.05f }; }

    VoiceBandpass whole;
    whole.init(NULL, 48000.0);
    std::vector<dsp::stereo_t> outW((size_t)n);
    whole.process((int)n, in.data(), outW.data());

    VoiceBandpass chunked;
    chunked.init(NULL, 48000.0);
    std::vector<dsp::stereo_t> outC((size_t)n);
    long pos = 0;
    while (pos < n) {
        int take = 1 + (int)(rng.uniform() * 8192.0f);
        if (pos + take > n) { take = (int)(n - pos); }
        chunked.process(take, in.data() + pos, outC.data() + pos);
        pos += take;
    }

    bool identical = true;
    for (long i = 0; identical && i < n; i++) {
        if (outW[(size_t)i].l != outC[(size_t)i].l || outW[(size_t)i].r != outC[(size_t)i].r) {
            identical = false;
        }
    }
    CHECK(identical);
}

TEST(bandpass_response_follows_samplerate_in_hz_domain) {
    const double rates[] = { 16000.0, 32000.0, 48000.0 };
    double firstGain = 0.0;
    for (int r = 0; r < 3; r++) {
        VoiceBandpass bp;
        bp.init(NULL, rates[r]);
        double g = bpToneGainDb(bp, rates[r], 1200.0);
        if (r == 0) { firstGain = g; }
        CHECK_NEAR(g, firstGain, 1.0);
    }
}

TEST(bandpass_reset_clears_history) {
    VoiceBandpass bp;
    bp.init(NULL, 48000.0);

    const long n = 4800;
    std::vector<dsp::stereo_t> loud((size_t)n);
    for (long i = 0; i < n; i++) {
        float v = 0.9f * std::sin(6.283185307179586 * 1000.0 * (double)i / 48000.0);
        loud[(size_t)i] = dsp::stereo_t{ v, v };
    }
    std::vector<dsp::stereo_t> out((size_t)n);
    bp.process((int)n, loud.data(), out.data());
    bp.reset();

    std::vector<dsp::stereo_t> silence((size_t)n, dsp::stereo_t{ 0.0f, 0.0f });
    std::vector<dsp::stereo_t> after((size_t)n);
    bp.process((int)n, silence.data(), after.data());
    for (long i = 0; i < n; i++) {
        CHECK(std::fabs(after[(size_t)i].l) < 1e-9);
    }
}

int main() {
    return tf::runAll();
}
