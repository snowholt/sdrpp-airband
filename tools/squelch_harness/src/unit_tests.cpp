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

using dsp::noise_reduction::adaptive::FrameMeter;

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

int main() {
    return tf::runAll();
}
