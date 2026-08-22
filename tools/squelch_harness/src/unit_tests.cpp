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

int main() {
    return tf::runAll();
}
