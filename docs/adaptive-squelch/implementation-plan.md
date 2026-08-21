# Adaptive Squelch Implementation Plan — Phase 0 and Phase 1

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline replay harness and an `AdaptiveSquelch` DSP block that estimates the noise floor over time and gates the audio with hysteresis, hang, pre-roll and a ramped mute, wired to SDR++'s existing but unimplemented `SQUELCH_MODE_SNR`.

**Architecture:** Six small, independently testable units — frame meter, noise floor estimator (three interchangeable implementations), pre-roll delay line, decision state machine, lifecycle/invalidation, diagnostics — composed by one `Processor<complex_t, complex_t>` block. A standalone harness binary drives the same headers with no streams, no threads and no GUI, so every estimator can be scored against ground truth with everything else held constant.

**Tech Stack:** C++17, header-only DSP in the style of `core/src/dsp/`, VOLK for vector math, CMake 3.13, a ~50-line hand-rolled test framework (the repository has no test infrastructure and no test library is installed).

**Spec:** `docs/adaptive-squelch/design.md` — read it before starting. This plan implements §4 (Phase 0), §5 (Phase 1) and §9 (acceptance criteria). Phase 1b (AM carrier detection) and Phase 2 (AM audio chain) are **not** in this plan.

## Global Constraints

Copied verbatim from the spec. Every task's requirements implicitly include this section.

- **C++17.** The project builds with `-std=c++17` (`CMakeLists.txt:88-99`).
- **No FFT anywhere in the squelch path.** Phase 2 may use FFT; Phase 1 may not.
- **No dynamic allocation inside `process()`.** All buffers are sized in `init` / `setSamplerate`.
- **All time constants are in seconds, derived from samplerate.** Never a fixed number of samples and never "N `process()` calls" — `process()` receives a variable `count` (`stream.h:70` returns `dataSize`).
- **Frame boundaries follow the sample stream, not the block boundary.** The emitted frame sequence must be identical however the input is chunked.
- **Histogram bin indices are hard-clamped at both ends.** SDRPlusPlusBrown issue #73 is a reported out-of-bounds histogram-index bug in exactly this kind of code.
- **Every persistent state has an invalidation rule.** Samplerate change, bandwidth change, retune, gain change, source start/stop and demodulator change all invalidate estimator state, histogram contents, delay line and state machine.
- **Estimator E (gated EMA) must never be the default** and its source must state the staleness and self-lock reasons.
- **No emoji, no casual tone, no personal voice** in any file, comment, commit message or console output. Everything in this plan lands in a public repository.
- **Default parameters** (spec §8): frame 10 ms, histogram −140…0 dBFS at 0.25 dB (560 bins), window 10 s, percentile 20th, open margin 9.5 dB, close margin 6.5 dB, qualify 30 ms, pre-roll 40 ms, hang 200 ms, release 150 ms, ramp 5 ms raised cosine, occupancy warning 80%.

## File Structure

| Path | Responsibility |
|---|---|
| `core/src/dsp/noise_reduction/adaptive_squelch/frame_meter.h` | U1 — variable `count` in, fixed-duration dB frames out |
| `core/src/dsp/noise_reduction/adaptive_squelch/noise_estimator.h` | U2 — estimator interface plus P, M and E implementations |
| `core/src/dsp/noise_reduction/adaptive_squelch/preroll.h` | U3 — look-behind delay line |
| `core/src/dsp/noise_reduction/adaptive_squelch/gate.h` | U4 — decision state machine and raised-cosine ramp |
| `core/src/dsp/noise_reduction/adaptive_squelch/diagnostics.h` | U6 — snapshot struct and lock-free publisher |
| `core/src/dsp/noise_reduction/adaptive_squelch.h` | The `Processor` block composing all units, plus U5 lifecycle |
| `tools/squelch_harness/CMakeLists.txt` | Standalone harness target, off by default |
| `tools/squelch_harness/src/test_framework.h` | Minimal test registry and assertions |
| `tools/squelch_harness/src/signal_gen.h` | Deterministic scene generator with ground-truth labels |
| `tools/squelch_harness/src/scorer.h` | Ground-truth timing metrics |
| `tools/squelch_harness/src/unit_tests.cpp` | Unit tests for every unit above |
| `tools/squelch_harness/src/harness_main.cpp` | Sweep runner, CSV trace, comparison table |
| `decoder_modules/radio/src/radio_module.h` | Mode plumbing, UI, config, invalidation hooks |

`core/CMakeLists.txt` globs `src/*.cpp` and `src/*.c` only, so adding header-only files under `core/src/dsp/` requires **no** CMake change.

---

### Task 1: Branch, harness skeleton and test framework

**Files:**
- Create: `tools/squelch_harness/CMakeLists.txt`
- Create: `tools/squelch_harness/src/test_framework.h`
- Create: `tools/squelch_harness/src/unit_tests.cpp`
- Modify: `CMakeLists.txt` (add the option and the conditional subdirectory)

**Interfaces:**
- Consumes: nothing.
- Produces: `TEST(name)`, `CHECK(cond)`, `CHECK_EQ(a, b)`, `CHECK_NEAR(a, b, tol)`, `tf::runAll() -> int`. Every later task's tests use exactly these four.

- [ ] **Step 1: Create the feature branch**

`main` is the default branch and is where finished work lands. Never commit this work directly to
`main`; it goes on a feature branch and is merged back through a pull request.

```bash
cd "$(git rev-parse --show-toplevel)"
git checkout -b feature/adaptive-squelch main
git status
```

- [ ] **Step 2: Write the test framework**

Create `tools/squelch_harness/src/test_framework.h`:

```cpp
#pragma once
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <functional>

// Minimal test registry. The repository has no test infrastructure and no test
// library is installed, so this file is the substrate the rest of the harness
// is built on. It is deliberately small enough to read in one sitting.
namespace tf {
    struct Case {
        const char* name;
        std::function<void()> fn;
    };

    inline std::vector<Case>& registry() {
        static std::vector<Case> r;
        return r;
    }

    inline int& failures() {
        static int f = 0;
        return f;
    }

    struct Reg {
        Reg(const char* n, std::function<void()> f) { registry().push_back({ n, f }); }
    };

    inline void fail(const char* file, int line, const std::string& msg) {
        std::fprintf(stderr, "    FAIL %s:%d  %s\n", file, line, msg.c_str());
        failures()++;
    }

    inline int runAll() {
        int failedCases = 0;
        for (auto& c : registry()) {
            int before = failures();
            std::printf("[ RUN  ] %s\n", c.name);
            c.fn();
            if (failures() > before) {
                std::printf("[ FAIL ] %s\n", c.name);
                failedCases++;
            }
            else {
                std::printf("[  OK  ] %s\n", c.name);
            }
        }
        std::printf("\n%d test(s), %d failed, %d assertion failure(s)\n",
                    (int)registry().size(), failedCases, failures());
        return failedCases ? 1 : 0;
    }
}

#define TEST(name)                                  \
    static void name();                             \
    static tf::Reg tf_reg_##name(#name, name);      \
    static void name()

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) { tf::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); } \
    } while (0)

#define CHECK_EQ(a, b)                                                  \
    do {                                                                \
        auto tf_a = (a); auto tf_b = (b);                               \
        if (!(tf_a == tf_b)) {                                          \
            tf::fail(__FILE__, __LINE__, "CHECK_EQ(" #a ", " #b ") got " \
                     + std::to_string(tf_a) + " vs " + std::to_string(tf_b)); \
        }                                                               \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                           \
    do {                                                                \
        double tf_a = (a), tf_b = (b), tf_t = (tol);                    \
        if (!(std::fabs(tf_a - tf_b) <= tf_t)) {                        \
            tf::fail(__FILE__, __LINE__, "CHECK_NEAR(" #a ", " #b ") got " \
                     + std::to_string(tf_a) + " vs " + std::to_string(tf_b)); \
        }                                                               \
    } while (0)
```

- [ ] **Step 3: Write the failing test**

Create `tools/squelch_harness/src/unit_tests.cpp`:

```cpp
#include "test_framework.h"

TEST(framework_self_check) {
    CHECK(1 + 1 == 2);
    CHECK_EQ(3, 3);
    CHECK_NEAR(1.0, 1.0005, 1e-3);
}

int main() {
    return tf::runAll();
}
```

- [ ] **Step 4: Write the harness CMakeLists**

Create `tools/squelch_harness/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.13)
project(squelch_harness)

find_package(PkgConfig REQUIRED)
pkg_check_modules(VOLK REQUIRED volk)

add_executable(squelch_unit_tests src/unit_tests.cpp)
target_compile_features(squelch_unit_tests PRIVATE cxx_std_17)
target_include_directories(squelch_unit_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/core/src"
    ${VOLK_INCLUDE_DIRS})
target_link_directories(squelch_unit_tests PRIVATE ${VOLK_LIBRARY_DIRS})
target_link_libraries(squelch_unit_tests PRIVATE ${VOLK_LIBRARIES})
```

`harness_main.cpp` is added to this file in Task 11; do not add it yet.

- [ ] **Step 5: Wire the option into the top-level build**

In `CMakeLists.txt`, add the option next to the other `option(...)` lines near the top of the file:

```cmake
option(OPT_BUILD_SQUELCH_HARNESS "Build the offline adaptive-squelch test harness (developer tool)" OFF)
```

Then add this near the end of the file, alongside the other `add_subdirectory` calls:

```cmake
if (OPT_BUILD_SQUELCH_HARNESS)
    add_subdirectory("tools/squelch_harness")
endif (OPT_BUILD_SQUELCH_HARNESS)
```

- [ ] **Step 6: Configure and run the test to verify it passes**

```bash
cd "$(git rev-parse --show-toplevel)"
cmake -S . -B build-harness -DOPT_BUILD_SQUELCH_HARNESS=ON
cmake --build build-harness --target squelch_unit_tests -j
./build-harness/tools/squelch_harness/squelch_unit_tests
```

Expected: `[  OK  ] framework_self_check` and `1 test(s), 0 failed`.

- [ ] **Step 7: Verify the harness is genuinely off by default**

```bash
cmake -S . -B build-default 2>&1 | tail -5
test ! -d build-default/tools/squelch_harness && echo "correctly not configured"
```

Expected: `correctly not configured`.

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt tools/squelch_harness
git commit -m "test: add offline squelch harness skeleton and test framework

The repository has no test infrastructure. This adds a standalone harness
target, off by default, that drives the header-only DSP blocks directly
with no streams, threads or GUI."
```

---

### Task 2: Deterministic scene generator with ground truth

**Files:**
- Create: `tools/squelch_harness/src/signal_gen.h`
- Modify: `tools/squelch_harness/src/unit_tests.cpp`

**Interfaces:**
- Consumes: `TEST`, `CHECK`, `CHECK_EQ`, `CHECK_NEAR` from Task 1.
- Produces: `sg::SceneConfig`, `sg::Label { long start; long end; }`, `sg::Scene { std::vector<dsp::complex_t> iq; std::vector<Label> labels; double samplerate; }`, `sg::Scene sg::generate(const SceneConfig&)`, and `sg::Rng` (a mulberry32 PRNG with `float uniform()` and `float gaussian()`).

Determinism matters more than realism here. Every run of the sweep in Task 11 must produce byte-identical input, or estimator comparisons are meaningless.

- [ ] **Step 1: Write the failing test**

Append to `tools/squelch_harness/src/unit_tests.cpp`, above `int main()`:

```cpp
#include "signal_gen.h"

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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cmake --build build-harness --target squelch_unit_tests -j
```

Expected: compile failure, `signal_gen.h: No such file or directory`.

- [ ] **Step 3: Write the implementation**

Create `tools/squelch_harness/src/signal_gen.h`:

```cpp
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include <dsp/types.h>

// M_PI is not guaranteed by the standard and is absent under MSVC without
// _USE_MATH_DEFINES, so it is spelled out here rather than assumed.
#define SG_TWO_PI 6.283185307179586

// Deterministic synthetic scenes with ground-truth transmission labels.
// Realism is not the goal; repeatability is. Every sweep run must see
// byte-identical input or estimator comparisons mean nothing.
namespace sg {

    // mulberry32. Small, fast, and identical on every platform and compiler,
    // which std::mt19937 distributions are not.
    class Rng {
    public:
        explicit Rng(uint32_t seed) : _s(seed) {}

        uint32_t next() {
            _s += 0x6D2B79F5u;
            uint32_t t = _s;
            t = (t ^ (t >> 15)) * (t | 1u);
            t ^= t + (t ^ (t >> 7)) * (t | 61u);
            return t ^ (t >> 14);
        }

        float uniform() { return (float)(next() >> 8) / 16777216.0f; }

        // Box-Muller. One value per call; the spare is cached.
        float gaussian() {
            if (_hasSpare) { _hasSpare = false; return _spare; }
            float u1 = uniform();
            if (u1 < 1e-7f) { u1 = 1e-7f; }
            float u2 = uniform();
            float mag = std::sqrt(-2.0f * std::log(u1));
            _spare = mag * std::sin(6.2831853f * u2);
            _hasSpare = true;
            return mag * std::cos(6.2831853f * u2);
        }

    private:
        uint32_t _s;
        bool _hasSpare = false;
        float _spare = 0.0f;
    };

    struct Label {
        long start;   // first sample of the transmission, inclusive
        long end;     // one past the last sample
    };

    struct SceneConfig {
        double samplerate = 15000.0;
        double durationSeconds = 60.0;
        float noiseFloorDbfs = -70.0f;   // mean power of the noise-only stream
        float cnrDb = 10.0f;             // peak transmission power above the floor
        double onSeconds = 3.0;
        double offSeconds = 2.0;
        double syllabicHz = 3.5;         // envelope modulation inside a burst
        uint32_t seed = 1u;
    };

    struct Scene {
        std::vector<dsp::complex_t> iq;
        std::vector<Label> labels;
        double samplerate = 0.0;
    };

    inline Scene generate(const SceneConfig& cfg) {
        Scene s;
        s.samplerate = cfg.samplerate;
        const long n = (long)std::llround(cfg.durationSeconds * cfg.samplerate);
        s.iq.resize((size_t)n);

        // Noise amplitude per component. Total mean power of re + im is
        // 2 * sigma^2, so sigma = sqrt(P / 2).
        const double noisePower = std::pow(10.0, cfg.noiseFloorDbfs / 10.0);
        const float sigma = (float)std::sqrt(noisePower / 2.0);
        const double sigPeakPower = noisePower * std::pow(10.0, cfg.cnrDb / 10.0);
        const float sigPeakAmp = (float)std::sqrt(sigPeakPower);

        const long onLen = (long)std::llround(cfg.onSeconds * cfg.samplerate);
        const long offLen = (long)std::llround(cfg.offSeconds * cfg.samplerate);
        const long period = onLen + offLen;

        // Start with silence so an estimator has something to converge on.
        for (long start = offLen; start + onLen <= n; start += period) {
            s.labels.push_back({ start, start + onLen });
        }

        Rng rng(cfg.seed);
        size_t labelIdx = 0;
        for (long i = 0; i < n; i++) {
            float re = sigma * rng.gaussian();
            float im = sigma * rng.gaussian();

            while (labelIdx < s.labels.size() && i >= s.labels[labelIdx].end) { labelIdx++; }
            if (labelIdx < s.labels.size() && i >= s.labels[labelIdx].start) {
                const long into = i - s.labels[labelIdx].start;
                const double t = (double)into / cfg.samplerate;
                // Syllabic envelope: never quite zero, so the burst is one
                // transmission rather than a train of short ones.
                const double env = 0.25 + 0.75 * 0.5 * (1.0 - std::cos(SG_TWO_PI * cfg.syllabicHz * t));
                const float amp = sigPeakAmp * (float)env;
                const double phase = SG_TWO_PI * 1200.0 * t;
                re += amp * (float)std::cos(phase);
                im += amp * (float)std::sin(phase);
            }
            s.iq[(size_t)i] = dsp::complex_t{ re, im };
        }
        return s;
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cmake --build build-harness --target squelch_unit_tests -j && ./build-harness/tools/squelch_harness/squelch_unit_tests
```

Expected: all four tests OK.

- [ ] **Step 5: Commit**

```bash
git add tools/squelch_harness/src/signal_gen.h tools/squelch_harness/src/unit_tests.cpp
git commit -m "test: add deterministic scene generator with ground-truth labels"
```

---

### Task 3: U1 frame meter

**Files:**
- Create: `core/src/dsp/noise_reduction/adaptive_squelch/frame_meter.h`
- Modify: `tools/squelch_harness/src/unit_tests.cpp`

**Interfaces:**
- Consumes: `sg::Rng` from Task 2, the test macros from Task 1.
- Produces: `dsp::noise_reduction::adaptive::FrameMeter` with `void init(double samplerate, double frameSeconds)`, `void setSamplerate(double)`, `void reset()`, `int frameLength() const`, `double frameSeconds() const`, `inline bool feedSample(const dsp::complex_t& s, float& frameDb)`, `void feed(int count, const dsp::complex_t* in, std::vector<float>& out)`, and `static float powerToDb(double meanPower)`.

The chunking-invariance test below is the direct test of the variable-`count` constraint and is the single most important test in this plan. Write it first.

- [ ] **Step 1: Write the failing test**

Append to `unit_tests.cpp`, above `int main()`:

```cpp
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cmake --build build-harness --target squelch_unit_tests -j
```

Expected: compile failure, `frame_meter.h: No such file or directory`.

- [ ] **Step 3: Write the implementation**

Create `core/src/dsp/noise_reduction/adaptive_squelch/frame_meter.h`:

```cpp
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cmake --build build-harness --target squelch_unit_tests -j && ./build-harness/tools/squelch_harness/squelch_unit_tests
```

Expected: all tests OK, including `frame_meter_output_is_independent_of_chunking`.

- [ ] **Step 5: Commit**

```bash
git add core/src/dsp/noise_reduction/adaptive_squelch/frame_meter.h tools/squelch_harness/src/unit_tests.cpp
git commit -m "feat(dsp): add frame meter for samplerate-derived power framing

Carries a partial frame across process() calls so the emitted frame
sequence does not depend on how the input stream is chunked."
```

---

### Task 4: U2 estimator interface and P (rolling dB histogram percentile)

**Files:**
- Create: `core/src/dsp/noise_reduction/adaptive_squelch/noise_estimator.h`
- Modify: `tools/squelch_harness/src/unit_tests.cpp`

**Interfaces:**
- Consumes: `FrameMeter::powerToDb` conceptually, `sg::Rng`, the test macros.
- Produces: `dsp::noise_reduction::adaptive::EstimatorType { ESTIMATOR_PERCENTILE, ESTIMATOR_MCRA, ESTIMATOR_GATED_EMA }`; abstract `NoiseEstimator` with `void init(double frameSeconds)`, `void reset()`, `float update(float framePowerDb, bool gateOpen)`, `float occupancy() const`; and `PercentileEstimator` adding `void setWindow(double seconds)`, `void setPercentile(float)`, `static int binOf(float db)`, and the constants `MIN_DB`, `MAX_DB`, `BIN_WIDTH`, `BIN_COUNT`.

Tasks 5 and 6 add `McraEstimator` and `GatedEmaEstimator` to this same file. All three implement the interface defined here.

- [ ] **Step 1: Write the failing test**

Append to `unit_tests.cpp`, above `int main()`:

```cpp
#include <dsp/noise_reduction/adaptive_squelch/noise_estimator.h>

using dsp::noise_reduction::adaptive::PercentileEstimator;

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
```

Add `#include <limits>` to the top of `unit_tests.cpp` if it is not already there.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cmake --build build-harness --target squelch_unit_tests -j
```

Expected: compile failure, `noise_estimator.h: No such file or directory`.

- [ ] **Step 3: Write the implementation**

Create `core/src/dsp/noise_reduction/adaptive_squelch/noise_estimator.h`:

```cpp
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dsp::noise_reduction::adaptive {

    enum EstimatorType {
        ESTIMATOR_PERCENTILE,
        ESTIMATOR_MCRA,
        ESTIMATOR_GATED_EMA
    };

    // Estimates the noise floor of a channel over time.
    //
    // The stream reaching this code has already been frequency-shifted,
    // resampled and low-pass filtered to the channel bandwidth
    // (radio_module.h:88, rx_vfo.h:24), so everything outside the channel is
    // stopband and there is no adjacent-channel noise reference. The floor must
    // therefore be estimated across time, not across frequency.
    class NoiseEstimator {
    public:
        virtual ~NoiseEstimator() {}

        // frameSeconds is the fixed frame duration produced by FrameMeter.
        // Every time constant in an implementation is derived from it.
        virtual void init(double frameSeconds) = 0;
        virtual void reset() = 0;

        // Consumes one frame and returns the current noise floor estimate in dB.
        virtual float update(float framePowerDb, bool gateOpen) = 0;

        // Fraction of the recent window sitting clearly above the estimate.
        // Above roughly 0.8 the estimate is no longer trustworthy: there is not
        // enough noise-only time in the window. RTLSDR-Airband documents this
        // exact failure for continuous ATIS/AWOS channels.
        virtual float occupancy() const { return 0.0f; }

        virtual const char* name() const = 0;
    };

    // ---------------------------------------------------------------------
    // P - rolling dB histogram percentile. The default.
    //
    // Deterministic memory, exact window expiration, no FFT, robust to
    // impulses, and no distribution-specific bias table to get wrong. The
    // percentile encodes an occupancy assumption: a q-quantile stays
    // noise-representative only while at least q of the window is noise-only,
    // so the 20th percentile tolerates roughly 80% channel occupancy.
    // ---------------------------------------------------------------------
    class PercentileEstimator : public NoiseEstimator {
    public:
        static constexpr float MIN_DB = -140.0f;
        static constexpr float MAX_DB = 0.0f;
        static constexpr float BIN_WIDTH = 0.25f;
        static constexpr int BIN_COUNT = (int)((MAX_DB - MIN_DB) / BIN_WIDTH);   // 560
        static constexpr float OCCUPANCY_MARGIN_DB = 6.0f;

        void init(double frameSeconds) override {
            _frameSeconds = frameSeconds;
            rebuild();
        }

        void setWindow(double seconds) {
            _windowSeconds = seconds;
            rebuild();
        }

        void setPercentile(float p) {
            _percentile = std::clamp(p, 0.01f, 0.99f);
        }

        void reset() override {
            std::fill(_bins.begin(), _bins.end(), 0);
            std::fill(_ring.begin(), _ring.end(), (uint16_t)0);
            _head = 0;
            _filled = 0;
            _lastEstimate = MIN_DB;
        }

        // Maps a dB value onto a bin index, hard-clamped at both ends.
        // The leading !(db > MIN_DB) also catches NaN, which would otherwise
        // fall through every comparison and produce an out-of-range index.
        static int binOf(float db) {
            if (!(db > MIN_DB)) { return 0; }
            if (db >= MAX_DB) { return BIN_COUNT - 1; }
            int idx = (int)((db - MIN_DB) / BIN_WIDTH);
            if (idx < 0) { idx = 0; }
            if (idx >= BIN_COUNT) { idx = BIN_COUNT - 1; }
            return idx;
        }

        float update(float framePowerDb, bool /*gateOpen*/) override {
            const int idx = binOf(framePowerDb);

            if (_filled == _windowFrames) {
                const uint16_t old = _ring[_head];
                if (_bins[old] > 0) { _bins[old]--; }
            }
            else {
                _filled++;
            }
            _ring[_head] = (uint16_t)idx;
            _head++;
            if (_head >= _windowFrames) { _head = 0; }
            _bins[idx]++;

            int target = (int)std::lround((double)_percentile * (double)_filled);
            if (target < 1) { target = 1; }

            int cum = 0;
            for (int b = 0; b < BIN_COUNT; b++) {
                cum += _bins[b];
                if (cum >= target) {
                    _lastEstimate = MIN_DB + ((float)b + 0.5f) * BIN_WIDTH;
                    break;
                }
            }
            return _lastEstimate;
        }

        float occupancy() const override {
            if (_filled == 0) { return 0.0f; }
            const int thr = binOf(_lastEstimate + OCCUPANCY_MARGIN_DB);
            int above = 0;
            for (int b = thr; b < BIN_COUNT; b++) { above += _bins[b]; }
            return (float)above / (float)_filled;
        }

        const char* name() const override { return "percentile"; }

    private:
        void rebuild() {
            _windowFrames = (int)std::lround(_windowSeconds / _frameSeconds);
            if (_windowFrames < 1) { _windowFrames = 1; }
            _ring.assign((size_t)_windowFrames, (uint16_t)0);
            _bins.assign((size_t)BIN_COUNT, 0);
            reset();
        }

        double _frameSeconds = 0.010;
        double _windowSeconds = 10.0;
        float _percentile = 0.20f;
        int _windowFrames = 1000;
        std::vector<uint16_t> _ring;
        std::vector<int32_t> _bins;
        int _head = 0;
        int _filled = 0;
        float _lastEstimate = MIN_DB;
    };
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cmake --build build-harness --target squelch_unit_tests -j && ./build-harness/tools/squelch_harness/squelch_unit_tests
```

Expected: all tests OK.

- [ ] **Step 5: Commit**

```bash
git add core/src/dsp/noise_reduction/adaptive_squelch/noise_estimator.h tools/squelch_harness/src/unit_tests.cpp
git commit -m "feat(dsp): add noise estimator interface and histogram percentile

Bin indices are hard-clamped at both ends, including NaN, following the
out-of-bounds histogram bug reported as SDRPlusPlusBrown issue #73."
```

---

### Task 5: U2 estimator M (MCRA-inspired soft recursive tracker)

**Files:**
- Modify: `core/src/dsp/noise_reduction/adaptive_squelch/noise_estimator.h`
- Modify: `tools/squelch_harness/src/unit_tests.cpp`

**Interfaces:**
- Consumes: `NoiseEstimator` from Task 4.
- Produces: `McraEstimator` implementing `NoiseEstimator`, plus the free helper `static float alphaFor(double tauSeconds, double frameSeconds)` used by Task 6.

- [ ] **Step 1: Write the failing test**

Append to `unit_tests.cpp`, above `int main()`:

```cpp
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
```

- [ ] **Step 2: Run the test to verify it fails**

Expected: compile failure, `McraEstimator` is not a member of the namespace.

- [ ] **Step 3: Write the implementation**

Add to `noise_estimator.h`, inside the namespace, after `PercentileEstimator`:

```cpp
    // One-pole smoothing coefficient for a given time constant.
    inline float alphaFor(double tauSeconds, double frameSeconds) {
        if (tauSeconds <= 0.0) { return 0.0f; }
        return (float)std::exp(-frameSeconds / tauSeconds);
    }

    // ---------------------------------------------------------------------
    // M - minimum-guided soft recursive tracker.
    //
    // Follows the MCRA idea: the excess of smoothed power over a slowly
    // tracked local minimum drives a presence probability, and that
    // probability throttles - but never fully stops - the noise update. That
    // is what makes it immune to the freeze-forever failure of a hard-gated
    // estimator.
    //
    // Deliberately named "inspired". The published MCRA and IMCRA results are
    // per-frequency-bin STFT results (Cohen and Berdugo 2002; Cohen 2003) and
    // do not transfer unchanged to a scalar broadband power envelope. Nothing
    // here claims their bias figures.
    // ---------------------------------------------------------------------
    class McraEstimator : public NoiseEstimator {
    public:
        static constexpr float PRESENCE_DB = 4.0f;        // excess over minimum that reads as signal
        static constexpr float IMPULSE_CLIP_DB = 20.0f;   // one sample may not move the tracker further
        static constexpr float MAX_ABOVE_MIN_DB = 3.0f;   // watchdog ceiling above the tracked minimum

        void init(double frameSeconds) override {
            _frameSeconds = frameSeconds;
            _aS = alphaFor(0.10, frameSeconds);      // power smoothing
            _aP = alphaFor(0.20, frameSeconds);      // presence smoothing
            _aFast = alphaFor(0.50, frameSeconds);   // fastest noise update, presence 0
            _aSlow = alphaFor(30.0, frameSeconds);   // slowest leak, presence 1
            _minWindowFrames = (int)std::lround(1.0 / frameSeconds);
            if (_minWindowFrames < 1) { _minWindowFrames = 1; }
            reset();
        }

        void reset() override {
            _started = false;
            _s = _min = _runMin = _noise = 0.0f;
            _p = 0.0f;
            _minCount = 0;
        }

        float update(float framePowerDb, bool /*gateOpen*/) override {
            if (!_started) {
                _started = true;
                _s = _min = _runMin = _noise = framePowerDb;
                _p = 0.0f;
                _minCount = 0;
                return _noise;
            }

            // Robust clipping. A single ignition pulse must not be able to drag
            // the tracker up; it is bounded to IMPULSE_CLIP_DB above the estimate.
            float clipped = framePowerDb;
            const float ceiling = _noise + IMPULSE_CLIP_DB;
            if (clipped > ceiling) { clipped = ceiling; }

            _s = _aS * _s + (1.0f - _aS) * clipped;

            // Two-half rolling minimum, so the minimum can rise again once the
            // frames that produced it have expired.
            if (_s < _runMin) { _runMin = _s; }
            _minCount++;
            if (_minCount >= _minWindowFrames) {
                _min = _runMin;
                _runMin = _s;
                _minCount = 0;
            }
            if (_s < _min) { _min = _s; }

            // Presence indicator, then temporal smoothing into a probability.
            const float raw = ((_s - _min) > PRESENCE_DB) ? 1.0f : 0.0f;
            _p = _aP * _p + (1.0f - _aP) * raw;

            // Probability-controlled smoothing. alpha rises toward _aSlow with
            // presence, so the update slows during signal without ever freezing.
            const float aD = _aFast + (_aSlow - _aFast) * _p;
            _noise = aD * _noise + (1.0f - aD) * clipped;

            // Watchdog. The estimate may never walk more than MAX_ABOVE_MIN_DB
            // above the tracked minimum, which bounds the runaway case where a
            // long transmission slowly drags the floor up into the signal.
            const float cap = _min + MAX_ABOVE_MIN_DB;
            if (_noise > cap) { _noise = cap; }
            return _noise;
        }

        float occupancy() const override {
            // A cheap proxy: the smoothed presence probability.
            return _p;
        }

        const char* name() const override { return "mcra"; }

    private:
        double _frameSeconds = 0.010;
        float _aS = 0.0f, _aP = 0.0f, _aFast = 0.0f, _aSlow = 0.0f;
        int _minWindowFrames = 100;
        int _minCount = 0;
        bool _started = false;
        float _s = 0.0f, _min = 0.0f, _runMin = 0.0f, _p = 0.0f, _noise = 0.0f;
    };
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cmake --build build-harness --target squelch_unit_tests -j && ./build-harness/tools/squelch_harness/squelch_unit_tests
```

Expected: all tests OK.

- [ ] **Step 5: Commit**

```bash
git add core/src/dsp/noise_reduction/adaptive_squelch/noise_estimator.h tools/squelch_harness/src/unit_tests.cpp
git commit -m "feat(dsp): add MCRA-inspired soft recursive noise tracker"
```

---

### Task 6: U2 estimator E (gated EMA with shadow tracker and watchdog)

**Files:**
- Modify: `core/src/dsp/noise_reduction/adaptive_squelch/noise_estimator.h`
- Modify: `tools/squelch_harness/src/unit_tests.cpp`

**Interfaces:**
- Consumes: `NoiseEstimator`, `alphaFor` from Tasks 4 and 5.
- Produces: `GatedEmaEstimator` implementing `NoiseEstimator`.

This estimator exists to be measured against, not to be shipped as a default. Its source comment must say so and say why.

- [ ] **Step 1: Write the failing test**

Append to `unit_tests.cpp`, above `int main()`:

```cpp
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
```

- [ ] **Step 2: Run the test to verify it fails**

Expected: compile failure, `GatedEmaEstimator` is not a member of the namespace.

- [ ] **Step 3: Write the implementation**

Add to `noise_estimator.h`, inside the namespace, after `McraEstimator`:

```cpp
    // ---------------------------------------------------------------------
    // E - gate-conditioned EMA with a slow shadow tracker and a watchdog.
    //
    // RETAINED AS A MEASURED BASELINE. NOT SUITABLE AS A DEFAULT.
    //
    // Two structural failures, both documented in the literature and both
    // observed in shipped software:
    //
    //   Staleness  - while the gate is open the fast tracker receives no
    //                observations at all. A long transmission, or a
    //                continuously transmitting channel such as ATIS or AWOS,
    //                starves it indefinitely. RTLSDR-Airband documents exactly
    //                this failure for its automatic squelch.
    //   Self-lock  - a false open freezes the very estimate that caused it, so
    //                the error can never be corrected by new observations.
    //
    // The shadow tracker and the watchdog bound both failures. They do not
    // remove them. Use PercentileEstimator or McraEstimator by default.
    // ---------------------------------------------------------------------
    class GatedEmaEstimator : public NoiseEstimator {
    public:
        static constexpr double FAST_TAU_SECONDS = 0.75;
        static constexpr double SHADOW_TAU_SECONDS = 20.0;
        static constexpr double WATCHDOG_SECONDS = 30.0;

        void init(double frameSeconds) override {
            _frameSeconds = frameSeconds;
            _aFast = alphaFor(FAST_TAU_SECONDS, frameSeconds);
            _aShadow = alphaFor(SHADOW_TAU_SECONDS, frameSeconds);
            _watchdogFrames = (int)std::lround(WATCHDOG_SECONDS / frameSeconds);
            if (_watchdogFrames < 1) { _watchdogFrames = 1; }
            reset();
        }

        void reset() override {
            _started = false;
            _fast = 0.0f;
            _shadow = 0.0f;
            _openFrames = 0;
        }

        float update(float framePowerDb, bool gateOpen) override {
            if (!_started) {
                _started = true;
                _fast = _shadow = framePowerDb;
                _openFrames = 0;
                return _fast;
            }

            // The shadow never freezes, whatever the gate is doing. It follows
            // downward at the fast rate and upward only slowly, so an open gate
            // cannot drag it up quickly but a genuinely quieter band still
            // reaches it in well under a second.
            if (framePowerDb < _shadow) {
                _shadow = _aFast * _shadow + (1.0f - _aFast) * framePowerDb;
            }
            else {
                _shadow = _aShadow * _shadow + (1.0f - _aShadow) * framePowerDb;
            }

            if (!gateOpen) {
                _fast = _aFast * _fast + (1.0f - _aFast) * framePowerDb;
                _openFrames = 0;
            }
            else {
                _openFrames++;
                if (_openFrames > _watchdogFrames) {
                    // The gate has been open too long for the frozen value to be
                    // trusted. Hand control to the shadow rather than keep it.
                    _fast = _shadow;
                }
            }
            return _fast;
        }

        const char* name() const override { return "gated_ema"; }

    private:
        double _frameSeconds = 0.010;
        float _aFast = 0.0f, _aShadow = 0.0f;
        int _watchdogFrames = 3000;
        int _openFrames = 0;
        bool _started = false;
        float _fast = 0.0f, _shadow = 0.0f;
    };
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cmake --build build-harness --target squelch_unit_tests -j && ./build-harness/tools/squelch_harness/squelch_unit_tests
```

Expected: all tests OK.

- [ ] **Step 5: Commit**

```bash
git add core/src/dsp/noise_reduction/adaptive_squelch/noise_estimator.h tools/squelch_harness/src/unit_tests.cpp
git commit -m "feat(dsp): add gated EMA baseline estimator with shadow and watchdog

Retained as a measured baseline only. Documented as unsuitable as a
default because of staleness and self-reinforcing lock-up."
```

---

### Task 7: U3 pre-roll delay line

**Files:**
- Create: `core/src/dsp/noise_reduction/adaptive_squelch/preroll.h`
- Modify: `tools/squelch_harness/src/unit_tests.cpp`

**Interfaces:**
- Consumes: nothing from earlier tasks except the test macros.
- Produces: `dsp::noise_reduction::adaptive::PreRoll` with `void init(double samplerate, double seconds)`, `void setLength(double samplerate, double seconds)`, `void reset()`, `int delaySamples() const`, and `inline bool push(const dsp::complex_t& in, dsp::complex_t& out)` which returns false while the line is still priming.

- [ ] **Step 1: Write the failing test**

Append to `unit_tests.cpp`, above `int main()`:

```cpp
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
```

- [ ] **Step 2: Run the test to verify it fails**

Expected: compile failure, `preroll.h: No such file or directory`.

- [ ] **Step 3: Write the implementation**

Create `core/src/dsp/noise_reduction/adaptive_squelch/preroll.h`:

```cpp
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cmake --build build-harness --target squelch_unit_tests -j && ./build-harness/tools/squelch_harness/squelch_unit_tests
```

- [ ] **Step 5: Commit**

```bash
git add core/src/dsp/noise_reduction/adaptive_squelch/preroll.h tools/squelch_harness/src/unit_tests.cpp
git commit -m "feat(dsp): add pre-roll delay line to prevent onset clipping"
```

---

### Task 8: U4 decision state machine and raised-cosine ramp

**Files:**
- Create: `core/src/dsp/noise_reduction/adaptive_squelch/gate.h`
- Modify: `tools/squelch_harness/src/unit_tests.cpp`

**Interfaces:**
- Consumes: the test macros.
- Produces: `dsp::noise_reduction::adaptive::GateState { GATE_CLOSED, GATE_QUALIFY, GATE_OPEN, GATE_HANG }`; `TransitionReason { REASON_RESET, REASON_SNR_ABOVE_OPEN, REASON_QUALIFY_ABANDONED, REASON_QUALIFY_MET, REASON_SNR_BELOW_CLOSE, REASON_HANG_EXPIRED }`; `const char* reasonName(TransitionReason)`; `const char* stateName(GateState)`; and `Gate` with `void init(double samplerate, double frameSeconds)`, `void setMargins(float openDb, float closeDb)`, `void setQualify(double)`, `void setHang(double)`, `void setRelease(double)`, `void setRamp(double)`, `void reset()`, `void updateFrame(float snrDb)`, `inline float nextGain()`, `GateState state() const`, `TransitionReason reason() const`, `bool isOpen() const`.

The state machine advances once per frame; the ramp advances once per sample.

- [ ] **Step 1: Write the failing test**

Append to `unit_tests.cpp`, above `int main()`:

```cpp
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
```

- [ ] **Step 2: Run the test to verify it fails**

Expected: compile failure, `gate.h: No such file or directory`.

- [ ] **Step 3: Write the implementation**

Create `core/src/dsp/noise_reduction/adaptive_squelch/gate.h`:

```cpp
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cmake --build build-harness --target squelch_unit_tests -j && ./build-harness/tools/squelch_harness/squelch_unit_tests
```

- [ ] **Step 5: Commit**

```bash
git add core/src/dsp/noise_reduction/adaptive_squelch/gate.h tools/squelch_harness/src/unit_tests.cpp
git commit -m "feat(dsp): add squelch decision state machine with hysteresis and hang

Two thresholds with a raised-cosine gain ramp, replacing the abrupt
memcpy/memset switch that produces a click in power_squelch.h."
```

---

### Task 9: U6 diagnostics snapshot and lock-free publisher

**Files:**
- Create: `core/src/dsp/noise_reduction/adaptive_squelch/diagnostics.h`
- Modify: `tools/squelch_harness/src/unit_tests.cpp`

**Interfaces:**
- Consumes: `GateState`, `TransitionReason` from Task 8.
- Produces: `dsp::noise_reduction::adaptive::Diagnostics` (a trivially copyable struct) and `DiagnosticsPublisher` with `void publish(const Diagnostics&)` and `Diagnostics read() const`.

One struct serves two clients: the GUI readout and the harness CSV. Serving both from one source is what makes the observability cheap enough to always leave on.

- [ ] **Step 1: Write the failing test**

Append to `unit_tests.cpp`, above `int main()`:

```cpp
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
```

Add `find_package(Threads REQUIRED)` and `Threads::Threads` to the harness link libraries in `tools/squelch_harness/CMakeLists.txt`:

```cmake
find_package(Threads REQUIRED)
target_link_libraries(squelch_unit_tests PRIVATE ${VOLK_LIBRARIES} Threads::Threads)
```

- [ ] **Step 2: Run the test to verify it fails**

Expected: compile failure, `diagnostics.h: No such file or directory`.

- [ ] **Step 3: Write the implementation**

Create `core/src/dsp/noise_reduction/adaptive_squelch/diagnostics.h`:

```cpp
#pragma once
#include <atomic>
#include <cstdint>

namespace dsp::noise_reduction::adaptive {

    // Everything a user or a test needs to understand why the gate did what it
    // did. Prior art across SDR software is consistently opaque about estimator
    // internals; exposing this is a deliberate difference and it is what makes
    // field tuning possible at all.
    struct Diagnostics {
        float framePowerDb = -140.0f;
        float noiseFloorDb = -140.0f;
        float snrDb = 0.0f;
        float occupancy = 0.0f;
        float carrierConfidence = 0.0f;   // Phase 1b; always 0 in Phase 1
        float carrierOffsetHz = 0.0f;     // Phase 1b; always 0 in Phase 1
        int state = 0;
        int lastTransitionReason = 0;
    };

    // Single-writer, multi-reader seqlock.
    //
    // The DSP thread writes and must never block on the GUI, so a mutex is the
    // wrong tool here: a GUI stall would stall audio. The writer brackets each
    // update with an odd then even sequence number; a reader that sees an odd
    // sequence, or a sequence that changed across its read, retries. The writer
    // is wait-free.
    class DiagnosticsPublisher {
    public:
        void publish(const Diagnostics& d) {
            const uint32_t s = _seq.load(std::memory_order_relaxed);
            _seq.store(s + 1, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
            _data = d;
            std::atomic_thread_fence(std::memory_order_release);
            _seq.store(s + 2, std::memory_order_relaxed);
        }

        Diagnostics read() const {
            Diagnostics out;
            for (;;) {
                const uint32_t before = _seq.load(std::memory_order_relaxed);
                // Odd means a write is in progress. Retry without reading.
                if (before & 1u) { continue; }
                std::atomic_thread_fence(std::memory_order_acquire);
                out = _data;
                std::atomic_thread_fence(std::memory_order_acquire);
                const uint32_t after = _seq.load(std::memory_order_relaxed);
                if (before == after) { return out; }
            }
        }

    private:
        std::atomic<uint32_t> _seq{ 0 };
        Diagnostics _data;
    };
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cmake --build build-harness --target squelch_unit_tests -j && ./build-harness/tools/squelch_harness/squelch_unit_tests
```

The torn-read test runs for a few seconds. That is expected.

- [ ] **Step 5: Commit**

```bash
git add core/src/dsp/noise_reduction/adaptive_squelch/diagnostics.h tools/squelch_harness/src/unit_tests.cpp tools/squelch_harness/CMakeLists.txt
git commit -m "feat(dsp): add squelch diagnostics snapshot with seqlock publisher"
```

---

### Task 10: The `AdaptiveSquelch` block and U5 lifecycle

**Files:**
- Create: `core/src/dsp/noise_reduction/adaptive_squelch.h`
- Modify: `tools/squelch_harness/src/unit_tests.cpp`

**Interfaces:**
- Consumes: `FrameMeter`, `NoiseEstimator` and its three implementations, `PreRoll`, `Gate`, `Diagnostics`, `DiagnosticsPublisher`.
- Produces: `dsp::noise_reduction::AdaptiveSquelch : Processor<complex_t, complex_t>` with `void init(stream<complex_t>* in, double samplerate)`, `void setSamplerate(double)`, `void setEstimator(adaptive::EstimatorType)`, `void setOpenMargin(float dB)`, `void setHysteresis(float dB)`, `void setQualify(double s)`, `void setHang(double s)`, `void setRelease(double s)`, `void setPreRoll(double s)`, `void setRamp(double s)`, `void reset()`, `adaptive::Diagnostics diagnostics() const`, `int process(int count, const complex_t* in, complex_t* out)` returning the emitted count, and `int run()`.

`process()` returns fewer samples than it consumes while the pre-roll line primes, so `run()` uses the multirate pattern already established in `processor.h` (`OVERRIDE_MULTIRATE_PROC_RUN`) rather than the default.

- [ ] **Step 1: Write the failing test**

Append to `unit_tests.cpp`, above `int main()`:

```cpp
#include <dsp/noise_reduction/adaptive_squelch.h>

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
```

Add `#include <dsp/noise_reduction/power_squelch.h>` to the top of `unit_tests.cpp`.

- [ ] **Step 2: Run the test to verify it fails**

Expected: compile failure, `adaptive_squelch.h: No such file or directory`.

- [ ] **Step 3: Write the implementation**

Create `core/src/dsp/noise_reduction/adaptive_squelch.h`:

```cpp
#pragma once
#include "../processor.h"
#include "adaptive_squelch/frame_meter.h"
#include "adaptive_squelch/noise_estimator.h"
#include "adaptive_squelch/preroll.h"
#include "adaptive_squelch/gate.h"
#include "adaptive_squelch/diagnostics.h"

namespace dsp::noise_reduction {

    // Adaptive squelch. Implements the SQUELCH_MODE_SNR branch that upstream
    // left stubbed at radio_module.h:675.
    //
    // The stream reaching this block has already been filtered to the channel
    // bandwidth by RxVFO, so there is no adjacent-channel noise reference and
    // the floor must be estimated across time. Every time constant is derived
    // from the samplerate in seconds, because process() receives a variable
    // count (stream.h:70 returns whatever upstream wrote).
    class AdaptiveSquelch : public Processor<complex_t, complex_t> {
        using base_type = Processor<complex_t, complex_t>;
    public:
        AdaptiveSquelch() {}

        void init(stream<complex_t>* in, double samplerate) {
            _samplerate = samplerate;
            rebuild();
            base_type::init(in);
        }

        // ---- lifecycle and invalidation (U5) ----------------------------
        // Every one of these invalidates estimator state, the delay line and
        // the state machine. Stale state from a different samplerate, gain or
        // band is worse than no state at all.

        void setSamplerate(double samplerate) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            if (samplerate == _samplerate) { return; }
            base_type::tempStop();
            _samplerate = samplerate;
            rebuild();
            base_type::tempStart();
        }

        void reset() {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            base_type::tempStop();
            clearState();
            base_type::tempStart();
        }

        // ---- configuration ----------------------------------------------

        void setEstimator(adaptive::EstimatorType type) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            base_type::tempStop();
            _estimatorType = type;
            selectEstimator();
            clearState();
            base_type::tempStart();
        }

        void setOpenMargin(float dB) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            _openMarginDb = dB;
            _gate.setMargins(_openMarginDb, _openMarginDb - _hysteresisDb);
        }

        void setHysteresis(float dB) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            _hysteresisDb = (dB < 0.0f) ? 0.0f : dB;
            _gate.setMargins(_openMarginDb, _openMarginDb - _hysteresisDb);
        }

        void setQualify(double seconds) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            _gate.setQualify(seconds);
        }

        void setHang(double seconds) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            _gate.setHang(seconds);
        }

        void setRelease(double seconds) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            _gate.setRelease(seconds);
        }

        void setRamp(double seconds) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            _gate.setRamp(seconds);
        }

        // Zero disables pre-roll entirely and removes the added latency.
        void setPreRoll(double seconds) {
            std::lock_guard<std::recursive_mutex> lck(base_type::ctrlMtx);
            base_type::tempStop();
            _preRollSeconds = (seconds < 0.0) ? 0.0 : seconds;
            _preRoll.setLength(_samplerate, _preRollSeconds);
            base_type::tempStart();
        }

        double preRollSeconds() const { return _preRollSeconds; }

        adaptive::Diagnostics diagnostics() const { return _diag.read(); }

        // ---- processing ---------------------------------------------------

        // Returns the number of samples written to out, which is fewer than
        // count while the pre-roll line primes.
        int process(int count, const complex_t* in, complex_t* out) {
            int emitted = 0;
            float frameDb = 0.0f;

            for (int i = 0; i < count; i++) {
                // The gate decision for this frame is computed from the newest
                // samples but applied to samples one pre-roll interval older,
                // which is what stops the onset being clipped.
                if (_meter.feedSample(in[i], frameDb)) {
                    const float noiseDb = _estimator->update(frameDb, _gate.isOpen());
                    const float snrDb = frameDb - noiseDb;
                    _gate.updateFrame(snrDb);

                    adaptive::Diagnostics d;
                    d.framePowerDb = frameDb;
                    d.noiseFloorDb = noiseDb;
                    d.snrDb = snrDb;
                    d.occupancy = _estimator->occupancy();
                    d.state = (int)_gate.state();
                    d.lastTransitionReason = (int)_gate.reason();
                    _diag.publish(d);
                }

                complex_t delayed;
                if (_preRoll.push(in[i], delayed)) {
                    const float g = _gate.nextGain();
                    out[emitted].re = delayed.re * g;
                    out[emitted].im = delayed.im * g;
                    emitted++;
                }
            }
            return emitted;
        }

        int run() {
            int count = base_type::_in->read();
            if (count < 0) { return -1; }
            int outCount = process(count, base_type::_in->readBuf, base_type::out.writeBuf);
            base_type::_in->flush();
            if (outCount) {
                if (!base_type::out.swap(outCount)) { return -1; }
            }
            return count;
        }

    private:
        void rebuild() {
            _meter.init(_samplerate, FRAME_SECONDS);
            _preRoll.setLength(_samplerate, _preRollSeconds);
            _gate.init(_samplerate, FRAME_SECONDS);
            _gate.setMargins(_openMarginDb, _openMarginDb - _hysteresisDb);
            selectEstimator();
            clearState();
        }

        void selectEstimator() {
            switch (_estimatorType) {
            case adaptive::ESTIMATOR_MCRA:      _estimator = &_mcra; break;
            case adaptive::ESTIMATOR_GATED_EMA: _estimator = &_ema;  break;
            case adaptive::ESTIMATOR_PERCENTILE:
            default:                            _estimator = &_pct;  break;
            }
            _estimator->init(FRAME_SECONDS);
        }

        void clearState() {
            _meter.reset();
            _preRoll.reset();
            _gate.reset();
            _estimator->reset();
            _diag.publish(adaptive::Diagnostics{});
        }

        static constexpr double FRAME_SECONDS = 0.010;

        double _samplerate = 48000.0;
        double _preRollSeconds = 0.040;
        float _openMarginDb = 9.5f;
        float _hysteresisDb = 3.0f;

        adaptive::EstimatorType _estimatorType = adaptive::ESTIMATOR_PERCENTILE;
        adaptive::PercentileEstimator _pct;
        adaptive::McraEstimator _mcra;
        adaptive::GatedEmaEstimator _ema;
        adaptive::NoiseEstimator* _estimator = &_pct;

        adaptive::FrameMeter _meter;
        adaptive::PreRoll _preRoll;
        adaptive::Gate _gate;
        adaptive::DiagnosticsPublisher _diag;
    };
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cmake --build build-harness --target squelch_unit_tests -j && ./build-harness/tools/squelch_harness/squelch_unit_tests
```

If `block_passes_a_strong_transmission` or `block_timing_scales_across_samplerates` fails, do **not** loosen the assertion. Dump the diagnostics CSV for the failing scene with Task 11's harness and find out why the gate behaved as it did.

- [ ] **Step 5: Commit**

```bash
git add core/src/dsp/noise_reduction/adaptive_squelch.h tools/squelch_harness/src/unit_tests.cpp
git commit -m "feat(dsp): add AdaptiveSquelch block composing meter, estimator and gate

Implements the SQUELCH_MODE_SNR branch left stubbed upstream. Emits fewer
samples than it consumes while the pre-roll line primes, so run() uses the
multirate pattern."
```

---

### Task 11: Scorer, CSV trace and estimator sweep

**Files:**
- Create: `tools/squelch_harness/src/scorer.h`
- Create: `tools/squelch_harness/src/harness_main.cpp`
- Modify: `tools/squelch_harness/CMakeLists.txt`
- Modify: `tools/squelch_harness/src/unit_tests.cpp`

**Interfaces:**
- Consumes: `sg::Scene`, `sg::Label`, `AdaptiveSquelch`.
- Produces: `scorer::Score { double falseOpenFrac; double missFrac; double fecMs; double overMs; int mscCount; int chatter; }` and `scorer::Score scorer::score(const std::vector<char>& gateOpenPerSample, const std::vector<sg::Label>& labels, double samplerate)`.

This task produces the numbers that acceptance criteria 3, 4, 6 and 7 are judged on.

- [ ] **Step 1: Write the failing test**

Append to `unit_tests.cpp`, above `int main()`:

`scorer.h` must be included after `adaptive_squelch.h` in `unit_tests.cpp`, because
the pre-roll test below uses both.

```cpp
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
```

- [ ] **Step 2: Run the test to verify it fails**

Expected: compile failure, `scorer.h: No such file or directory`.

- [ ] **Step 3: Write the scorer**

Create `tools/squelch_harness/src/scorer.h`:

```cpp
#pragma once
#include <cmath>
#include <vector>
#include "signal_gen.h"

// Ground-truth timing metrics for a gate. The terminology follows the voice
// activity detection literature deliberately, so results are comparable with
// published work: FEC is front-end clipping, MSC is mid-speech clipping and
// OVER is overhang.
namespace scorer {

    struct Score {
        double falseOpenFrac = 0.0;   // fraction of idle time passed unmuted
        double missFrac = 0.0;        // fraction of labelled time muted
        double fecMs = 0.0;           // mean speech lost after a true onset
        double overMs = 0.0;          // mean noise passed after a true end
        int mscCount = 0;             // closures inside a transmission
        int chatter = 0;              // total gate transitions
    };

    // gateOpen is one entry per OUTPUT sample.
    //
    // No delay compensation is applied here, and that is deliberate. The block
    // emits the pre-rolled sample, so output index j holds input sample j; the
    // pre-roll shows up as the gate decision for output j having been taken one
    // pre-roll interval later in input time. Subtracting the delay again would
    // cancel exactly the benefit this metric exists to measure.
    inline Score score(const std::vector<char>& gateOpen,
                       const std::vector<sg::Label>& labels,
                       double samplerate) {
        Score s;
        const long n = (long)gateOpen.size();
        const double msPerSample = 1000.0 / samplerate;

        auto openAtInput = [&](long inputIdx) -> bool {
            if (inputIdx < 0 || inputIdx >= n) { return false; }
            return gateOpen[(size_t)inputIdx] != 0;
        };

        // Mark which input samples are inside a labelled transmission.
        std::vector<char> active((size_t)n, 0);
        for (auto& l : labels) {
            for (long i = l.start; i < l.end && i < (long)active.size(); i++) {
                active[(size_t)i] = 1;
            }
        }

        long idleTotal = 0, idleOpen = 0, activeTotal = 0, activeMuted = 0;
        for (long i = 0; i < (long)active.size(); i++) {
            const bool op = openAtInput(i);
            if (active[(size_t)i]) {
                activeTotal++;
                if (!op) { activeMuted++; }
            }
            else {
                idleTotal++;
                if (op) { idleOpen++; }
            }
        }
        s.falseOpenFrac = idleTotal ? (double)idleOpen / (double)idleTotal : 0.0;
        s.missFrac = activeTotal ? (double)activeMuted / (double)activeTotal : 0.0;

        double fecSum = 0.0, overSum = 0.0;
        for (auto& l : labels) {
            long fec = 0;
            for (long i = l.start; i < l.end; i++) {
                if (openAtInput(i)) { break; }
                fec++;
            }
            fecSum += (double)fec * msPerSample;

            long over = 0;
            for (long i = l.end; i < (long)active.size() && !active[(size_t)i]; i++) {
                if (!openAtInput(i)) { break; }
                over++;
            }
            overSum += (double)over * msPerSample;

            // Closures strictly inside the transmission, after it first opened.
            bool started = false, wasOpen = false;
            for (long i = l.start; i < l.end; i++) {
                const bool op = openAtInput(i);
                if (!started) {
                    if (op) { started = true; wasOpen = true; }
                    continue;
                }
                if (wasOpen && !op) { s.mscCount++; }
                wasOpen = op;
            }
        }
        if (!labels.empty()) {
            s.fecMs = fecSum / (double)labels.size();
            s.overMs = overSum / (double)labels.size();
        }

        for (long i = 1; i < n; i++) {
            if (gateOpen[(size_t)i] != gateOpen[(size_t)i - 1]) { s.chatter++; }
        }
        return s;
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cmake --build build-harness --target squelch_unit_tests -j && ./build-harness/tools/squelch_harness/squelch_unit_tests
```

- [ ] **Step 5: Write the sweep runner**

Create `tools/squelch_harness/src/harness_main.cpp`:

```cpp
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "signal_gen.h"
#include "scorer.h"
#include <dsp/noise_reduction/adaptive_squelch.h>
#include <dsp/noise_reduction/adaptive_squelch/gate.h>

using dsp::noise_reduction::AdaptiveSquelch;
namespace ad = dsp::noise_reduction::adaptive;

namespace {

    struct RunResult {
        scorer::Score score;
        double nsPerSample = 0.0;
    };

    // Feeds a scene through the block in uneven chunks, recording the gate
    // state for every output sample and optionally writing a frame trace.
    RunResult runOne(const sg::Scene& scene,
                     ad::EstimatorType type,
                     const char* csvPath) {
        AdaptiveSquelch sq;
        sq.init(NULL, scene.samplerate);
        sq.setEstimator(type);

        std::vector<char> gateOpen;
        gateOpen.reserve(scene.iq.size());
        std::vector<dsp::complex_t> scratch(65536);

        std::FILE* csv = NULL;
        if (csvPath) {
            csv = std::fopen(csvPath, "w");
            if (csv) {
                std::fprintf(csv, "t,framePowerDb,noiseFloorDb,snrDb,occupancy,state,reason\n");
            }
        }

        sg::Rng chunkRng(4242);
        long pos = 0;
        double lastTraceT = -1.0;
        const auto t0 = std::chrono::steady_clock::now();

        while (pos < (long)scene.iq.size()) {
            int take = 1 + (int)(chunkRng.uniform() * 8192.0f);
            if (pos + take > (long)scene.iq.size()) { take = (int)((long)scene.iq.size() - pos); }
            const int n = sq.process(take, scene.iq.data() + pos, scratch.data());

            const ad::Diagnostics d = sq.diagnostics();
            const bool open = (d.state == (int)ad::GATE_OPEN || d.state == (int)ad::GATE_HANG);
            for (int i = 0; i < n; i++) { gateOpen.push_back(open ? 1 : 0); }

            if (csv) {
                const double t = (double)pos / scene.samplerate;
                if (t - lastTraceT >= 0.010) {
                    lastTraceT = t;
                    std::fprintf(csv, "%.4f,%.2f,%.2f,%.2f,%.3f,%s,%s\n",
                                 t, d.framePowerDb, d.noiseFloorDb, d.snrDb, d.occupancy,
                                 ad::stateName((ad::GateState)d.state),
                                 ad::reasonName((ad::TransitionReason)d.lastTransitionReason));
                }
            }
            pos += take;
        }
        const auto t1 = std::chrono::steady_clock::now();
        if (csv) { std::fclose(csv); }

        RunResult r;
        r.score = scorer::score(gateOpen, scene.labels, scene.samplerate);
        const double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        r.nsPerSample = ns / (double)scene.iq.size();
        return r;
    }
}

int main(int argc, char** argv) {
    const char* csvPath = NULL;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--trace") && i + 1 < argc) { csvPath = argv[++i]; }
    }

    const struct { const char* name; ad::EstimatorType type; } estimators[] = {
        { "percentile", ad::ESTIMATOR_PERCENTILE },
        { "mcra",       ad::ESTIMATOR_MCRA       },
        { "gated_ema",  ad::ESTIMATOR_GATED_EMA  }
    };
    const float cnrPoints[] = { 30.0f, 20.0f, 15.0f, 10.0f, 6.0f, 3.0f };

    std::printf("%-12s %6s %12s %10s %10s %10s %7s %8s %10s\n",
                "estimator", "CNR", "falseOpen%", "miss%", "FEC ms", "OVER ms",
                "MSC", "chatter", "ns/sample");
    std::printf("--------------------------------------------------------"
                "----------------------------------------\n");

    for (auto& e : estimators) {
        for (float cnr : cnrPoints) {
            sg::SceneConfig cfg;
            cfg.durationSeconds = 120.0;
            cfg.cnrDb = cnr;
            cfg.seed = 1u;
            sg::Scene scene = sg::generate(cfg);

            // Only trace the first combination, or the files get unwieldy.
            const char* trace = (csvPath && e.type == ad::ESTIMATOR_PERCENTILE
                                 && cnr == cnrPoints[0]) ? csvPath : NULL;
            RunResult r = runOne(scene, e.type, trace);

            std::printf("%-12s %6.1f %12.3f %10.3f %10.1f %10.1f %7d %8d %10.1f\n",
                        e.name, cnr,
                        100.0 * r.score.falseOpenFrac,
                        100.0 * r.score.missFrac,
                        r.score.fecMs, r.score.overMs,
                        r.score.mscCount, r.score.chatter,
                        r.nsPerSample);
        }
        std::printf("\n");
    }
    return 0;
}
```

- [ ] **Step 6: Add the sweep binary to the harness build**

Append to `tools/squelch_harness/CMakeLists.txt`:

```cmake
add_executable(squelch_sweep src/harness_main.cpp)
target_compile_features(squelch_sweep PRIVATE cxx_std_17)
target_include_directories(squelch_sweep PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/core/src"
    ${VOLK_INCLUDE_DIRS})
target_link_directories(squelch_sweep PRIVATE ${VOLK_LIBRARY_DIRS})
target_link_libraries(squelch_sweep PRIVATE ${VOLK_LIBRARIES} Threads::Threads)
```

- [ ] **Step 7: Run the sweep and record the result**

```bash
cmake --build build-harness --target squelch_sweep -j
./build-harness/tools/squelch_harness/squelch_sweep --trace /tmp/squelch-trace.csv | tee docs/adaptive-squelch/sweep-baseline.txt
head -20 /tmp/squelch-trace.csv
```

**Acceptance criterion 4 is checked here.** Somewhere in the table, `percentile` must dominate
`gated_ema`: either a lower `falseOpen%` at a comparable `miss%`, **or** a usable result at a CNR
where `gated_ema` misses everything. Compare against the reference run in the appendix below — it
shows the second form. If neither appears, stop and report the table rather than adjusting
parameters until it passes; the design assumption would be wrong and the spec would need revisiting.

- [ ] **Step 8: Check for warnings and for allocation inside `process()`**

Acceptance criterion 10. Rebuild the harness with the project's warning level plus
the extra warnings, and confirm the new headers are clean:

```bash
cd "$(git rev-parse --show-toplevel)"
g++ -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter -fsyntax-only \
    -Icore/src -Itools/squelch_harness/src \
    tools/squelch_harness/src/unit_tests.cpp 2>&1 | grep -v "^In file" | head -40
```

Expected: no output from the new `adaptive_squelch` headers.

Then confirm `process()` performs no heap traffic. Run the sweep under a
malloc-counting interposer:

```bash
cat > /tmp/allocwatch.c <<'EOF'
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
static long count = 0;
void* malloc(size_t n) {
    static void* (*real)(size_t) = 0;
    if (!real) { real = dlsym(RTLD_NEXT, "malloc"); }
    count++;
    return real(n);
}
__attribute__((destructor)) static void report(void) {
    fprintf(stderr, "total mallocs: %ld\n", count);
}
EOF
gcc -shared -fPIC -o /tmp/allocwatch.so /tmp/allocwatch.c -ldl
LD_PRELOAD=/tmp/allocwatch.so ./build-harness/tools/squelch_harness/squelch_sweep > /dev/null
```

The absolute count is not the point — scene generation and the output vectors
allocate. Run it twice with `cfg.durationSeconds` doubled in `harness_main.cpp`
and confirm the count does **not** scale with the number of `process()` calls.
If it does, something in the block is allocating per call. Restore the duration
afterwards.

- [ ] **Step 9: Commit**

```bash
git add tools/squelch_harness/src/scorer.h tools/squelch_harness/src/harness_main.cpp \
        tools/squelch_harness/CMakeLists.txt tools/squelch_harness/src/unit_tests.cpp
git commit -m "test: add ground-truth scorer and estimator sweep runner"
```

---

### Task 12: Wire `SQUELCH_MODE_SNR` into the radio module

**Files:**
- Modify: `decoder_modules/radio/src/radio_module.h`

**Interfaces:**
- Consumes: `dsp::noise_reduction::AdaptiveSquelch` and `dsp::noise_reduction::adaptive::*` from Task 10.
- Produces: a working `SNR` entry in the squelch mode combo, persisted per demodulator.

This task has no unit test — the file is a GUI module that cannot be built without the whole application. It is verified by building SDR++ and using it. Every algorithmic behaviour it depends on was already tested in Tasks 3 through 11.

- [ ] **Step 1: Add the include**

At the top of `radio_module.h`, next to the existing squelch includes (currently lines 11-12):

```cpp
#include <dsp/noise_reduction/power_squelch.h>
#include <dsp/noise_reduction/adaptive_squelch.h>
#include <dsp/noise_reduction/ctcss_squelch.h>
```

- [ ] **Step 2: Enable the mode and add the member variables**

Uncomment the SNR mode definition (currently line 55):

```cpp
        squelchModes.define("snr", "SNR", SQUELCH_MODE_SNR);
```

In the private member section, next to `squelchLevel`, add:

```cpp
    dsp::noise_reduction::AdaptiveSquelch adaptiveSquelch;

    const float MIN_SQUELCH_MARGIN = 0.0f;
    const float MAX_SQUELCH_MARGIN = 30.0f;

    // Defaults from the design spec, section 8. The open margin is the one
    // value with a cited source: RTLSDR-Airband opens at about 10 dB above its
    // estimated noise floor and its documented default SNR threshold is
    // 9.54 dB. Everything else is a starting point to be tuned on recordings.
    float squelchMargin = 9.5f;
    float squelchHysteresis = 3.0f;
    float squelchHangMs = 200.0f;
    float squelchPreRollMs = 40.0f;
    int squelchEstimatorId = 0;
    OptionList<std::string, int> squelchEstimators;
```

- [ ] **Step 3: Populate the estimator list and register the block**

In the constructor, next to the other `squelchModes.define(...)` calls:

```cpp
        squelchEstimators.define("percentile", "Percentile (recommended)", (int)dsp::noise_reduction::adaptive::ESTIMATOR_PERCENTILE);
        squelchEstimators.define("mcra", "Adaptive tracker", (int)dsp::noise_reduction::adaptive::ESTIMATOR_MCRA);
        squelchEstimators.define("gated_ema", "Gated average (basic)", (int)dsp::noise_reduction::adaptive::ESTIMATOR_GATED_EMA);
```

In the IF chain setup, next to `powerSquelch.init(NULL, MIN_SQUELCH);` (currently line 92):

```cpp
        powerSquelch.init(NULL, MIN_SQUELCH);
        adaptiveSquelch.init(NULL, 48000.0);
```

And next to `ifChain.addBlock(&powerSquelch, false);` (currently line 95):

```cpp
        ifChain.addBlock(&powerSquelch, false);
        ifChain.addBlock(&adaptiveSquelch, false);
```

Order in `links` is signal order (`chain.h` `blockBefore` / `blockAfter`), so this places the block
between the power squelch and `fmnr`. `powerSquelch` and `adaptiveSquelch` are never both enabled,
because `setSquelchMode` disables every squelch block before enabling one. `fmnr` **can** be enabled
at the same time on NFM, and it also buffers samples, so Step 11 check 9 exercises that pairing
specifically.

- [ ] **Step 4: Implement the mode switch**

In `setSquelchMode`, add the new block to the "Disable all squelch blocks" section:

```cpp
        // Disable all squelch blocks
        ifChain.disableBlock(&powerSquelch, [=](dsp::stream<dsp::complex_t>* out){ selectedDemod->setInput(out); });
        ifChain.disableBlock(&adaptiveSquelch, [=](dsp::stream<dsp::complex_t>* out){ selectedDemod->setInput(out); });
        afChain.disableBlock(&ctcss, [=](dsp::stream<dsp::stereo_t>* out){ stream.setInput(out); });
```

Replace the `SQUELCH_MODE_SNR` stub:

```cpp
        case SQUELCH_MODE_SNR:
            // Enable the adaptive squelch block. Its state is cleared on entry
            // because a floor estimated in a previous mode, band or gain
            // setting is not valid here.
            adaptiveSquelch.reset();
            ifChain.enableBlock(&adaptiveSquelch, [=](dsp::stream<dsp::complex_t>* out){ selectedDemod->setInput(out); });
            break;
```

- [ ] **Step 5: Add the setters**

Next to `setSquelchLevel`, add:

```cpp
    void setSquelchMargin(float margin) {
        squelchMargin = std::clamp<float>(margin, MIN_SQUELCH_MARGIN, MAX_SQUELCH_MARGIN);
        adaptiveSquelch.setOpenMargin(squelchMargin);
        if (!selectedDemod) { return; }
        config.acquire();
        config.conf[name][selectedDemod->getName()]["squelchMargin"] = squelchMargin;
        config.release(true);
    }

    void setSquelchHysteresis(float dB) {
        squelchHysteresis = std::clamp<float>(dB, 0.0f, 12.0f);
        adaptiveSquelch.setHysteresis(squelchHysteresis);
        if (!selectedDemod) { return; }
        config.acquire();
        config.conf[name][selectedDemod->getName()]["squelchHysteresis"] = squelchHysteresis;
        config.release(true);
    }

    void setSquelchHang(float ms) {
        squelchHangMs = std::clamp<float>(ms, 0.0f, 2000.0f);
        adaptiveSquelch.setHang((double)squelchHangMs / 1000.0);
        if (!selectedDemod) { return; }
        config.acquire();
        config.conf[name][selectedDemod->getName()]["squelchHang"] = squelchHangMs;
        config.release(true);
    }

    void setSquelchPreRoll(float ms) {
        squelchPreRollMs = std::clamp<float>(ms, 0.0f, 250.0f);
        adaptiveSquelch.setPreRoll((double)squelchPreRollMs / 1000.0);
        if (!selectedDemod) { return; }
        config.acquire();
        config.conf[name][selectedDemod->getName()]["squelchPreRoll"] = squelchPreRollMs;
        config.release(true);
    }

    void setSquelchEstimator(int estimatorId) {
        squelchEstimatorId = estimatorId;
        adaptiveSquelch.setEstimator((dsp::noise_reduction::adaptive::EstimatorType)squelchEstimators[estimatorId]);
        if (!selectedDemod) { return; }
        config.acquire();
        config.conf[name][selectedDemod->getName()]["squelchEstimator"] = squelchEstimators.key(estimatorId);
        config.release(true);
    }
```

- [ ] **Step 6: Wire the invalidation hooks (U5)**

This is the step most likely to be skipped and most likely to cause an unexplainable field failure. Every one of these is required.

In `selectDemod`, immediately after `vfo->setSampleRate(ifSamplerate, bandwidth);`:

```cpp
        // The noise floor estimate is only meaningful for one samplerate and
        // one channel bandwidth. Both just changed.
        adaptiveSquelch.setSamplerate(ifSamplerate);
```

In `setBandwidth`, after `selectedDemod->setBandwidth(bandwidth);`:

```cpp
        // A bandwidth change alters the noise power reaching the block, so the
        // stored floor is stale by exactly 10*log10(newBw/oldBw).
        adaptiveSquelch.reset();
```

In `setAudioSampleRate`, inside the `if (!postProcEnabled && vfo)` branch, after `vfo->setSampleRate(selectedDemod->getIFSampleRate(), bandwidth);`:

```cpp
            adaptiveSquelch.setSamplerate(selectedDemod->getIFSampleRate());
```

- [ ] **Step 7: Load and apply the persisted configuration**

In the defaults block of `selectDemod` (next to `squelchLevel = MIN_SQUELCH;`):

```cpp
        squelchMargin = 9.5f;
        squelchHysteresis = 3.0f;
        squelchHangMs = 200.0f;
        squelchPreRollMs = 40.0f;
        squelchEstimatorId = squelchEstimators.keyId("percentile");
```

In the config-loading block (next to the existing `squelchLevel` load):

```cpp
        if (config.conf[name][selectedDemod->getName()].contains("squelchMargin")) {
            squelchMargin = config.conf[name][selectedDemod->getName()]["squelchMargin"];
        }
        if (config.conf[name][selectedDemod->getName()].contains("squelchHysteresis")) {
            squelchHysteresis = config.conf[name][selectedDemod->getName()]["squelchHysteresis"];
        }
        if (config.conf[name][selectedDemod->getName()].contains("squelchHang")) {
            squelchHangMs = config.conf[name][selectedDemod->getName()]["squelchHang"];
        }
        if (config.conf[name][selectedDemod->getName()].contains("squelchPreRoll")) {
            squelchPreRollMs = config.conf[name][selectedDemod->getName()]["squelchPreRoll"];
        }
        if (config.conf[name][selectedDemod->getName()].contains("squelchEstimator")) {
            std::string est = config.conf[name][selectedDemod->getName()]["squelchEstimator"];
            if (squelchEstimators.keyExists(est)) {
                squelchEstimatorId = squelchEstimators.keyId(est);
            }
        }
```

In the "Configure squelch" block, **before** `setSquelchMode(...)` so the block is configured before it is enabled:

```cpp
        // Configure squelch
        setSquelchEstimator(squelchEstimatorId);
        setSquelchMargin(squelchMargin);
        setSquelchHysteresis(squelchHysteresis);
        setSquelchHang(squelchHangMs);
        setSquelchPreRoll(squelchPreRollMs);
        setSquelchMode(squelchAllowed ? squelchModes[squelchModeId] : SQUELCH_MODE_OFF);
        setSquelchLevel(squelchLevel);
        setCTCSSTone(ctcssTones[ctcssToneId]);
```

- [ ] **Step 8: Add the controls**

In `menuHandler`, in the squelch mode switch, next to `case SQUELCH_MODE_POWER:`:

```cpp
            case SQUELCH_MODE_SNR:
                ImGui::LeftLabel("Squelch Margin");
                ImGui::FillWidth();
                if (ImGui::SliderFloat(("##_radio_sqelch_margin_" + _this->name).c_str(), &_this->squelchMargin, _this->MIN_SQUELCH_MARGIN, _this->MAX_SQUELCH_MARGIN, "%.1f dB above noise")) {
                    _this->setSquelchMargin(_this->squelchMargin);
                }
                ImGui::LeftLabel("Hysteresis");
                ImGui::FillWidth();
                if (ImGui::SliderFloat(("##_radio_sqelch_hyst_" + _this->name).c_str(), &_this->squelchHysteresis, 0.0f, 12.0f, "%.1f dB")) {
                    _this->setSquelchHysteresis(_this->squelchHysteresis);
                }
                ImGui::LeftLabel("Hang");
                ImGui::FillWidth();
                if (ImGui::SliderFloat(("##_radio_sqelch_hang_" + _this->name).c_str(), &_this->squelchHangMs, 0.0f, 1000.0f, "%.0f ms")) {
                    _this->setSquelchHang(_this->squelchHangMs);
                }
                ImGui::LeftLabel("Pre-roll");
                ImGui::FillWidth();
                if (ImGui::SliderFloat(("##_radio_sqelch_preroll_" + _this->name).c_str(), &_this->squelchPreRollMs, 0.0f, 200.0f, "%.0f ms")) {
                    _this->setSquelchPreRoll(_this->squelchPreRollMs);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Delays the audio by this amount so the start of a\n"
                                      "transmission is not cut off. Set to 0 for the lowest\n"
                                      "latency, at the cost of clipped word beginnings.");
                }
                ImGui::LeftLabel("Noise Estimator");
                ImGui::FillWidth();
                if (ImGui::Combo(("##_radio_sqelch_est_" + _this->name).c_str(), &_this->squelchEstimatorId, _this->squelchEstimators.txt)) {
                    _this->setSquelchEstimator(_this->squelchEstimatorId);
                }
                break;
```

- [ ] **Step 9: Add the diagnostics readout**

In the "Display the squelch diagnostics" switch, add:

```cpp
        case SQUELCH_MODE_SNR:
            {
                auto d = _this->adaptiveSquelch.diagnostics();
                ImGui::Text("Noise floor: %.1f dB", d.noiseFloorDb);
                ImGui::Text("SNR: %.1f dB", d.snrDb);
                ImGui::TextUnformatted("Gate:");
                ImGui::SameLine();
                if (d.state == (int)dsp::noise_reduction::adaptive::GATE_OPEN ||
                    d.state == (int)dsp::noise_reduction::adaptive::GATE_HANG) {
                    ImGui::TextColored(ImVec4(0, 1, 0, 1), "%s",
                        dsp::noise_reduction::adaptive::stateName((dsp::noise_reduction::adaptive::GateState)d.state));
                }
                else {
                    ImGui::TextUnformatted(
                        dsp::noise_reduction::adaptive::stateName((dsp::noise_reduction::adaptive::GateState)d.state));
                }
                if (d.occupancy > 0.8f) {
                    ImGui::TextColored(ImVec4(1, 0.6f, 0, 1), "Channel busy %.0f%%", d.occupancy * 100.0f);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("This channel is transmitting almost continuously, so there\n"
                                          "is not enough quiet time to measure the noise floor.\n"
                                          "Use the Power squelch mode with a fixed level instead.");
                    }
                }
            }
            break;
```

- [ ] **Step 10: Build the full application**

```bash
cd "$(git rev-parse --show-toplevel)"
cmake -S . -B build-app -DCMAKE_BUILD_TYPE=Release
cmake --build build-app -j
```

Expected: a clean build with no new warnings. If dependencies are missing, disable the source modules that need them rather than changing anything in this plan.

- [ ] **Step 11: Verify in the application**

```bash
./build-app/sdrpp
```

Check, in order:

1. Select an AM or NFM demodulator. **SNR** now appears in the Squelch Mode list.
2. Selecting it shows the margin, hysteresis, hang, pre-roll and estimator controls, and the diagnostics readout below.
3. With no signal, the noise floor reading settles within about ten seconds and the gate reads CLOSED.
4. On a transmission, the gate reads OPEN and the SNR reading rises.
5. Changing the bandwidth slider makes the noise floor re-converge rather than jump to a wrong value and stay there.
6. Switching demodulator and back preserves every setting.
7. Setting pre-roll to 0 and back changes the audio latency audibly and does not crash.
8. Switching between Off, Power, SNR and CTCSS in every order does not crash.
9. **On NFM, enable FM IF Noise Reduction and SNR squelch together**, then toggle pre-roll between
   0 and 40 ms several times. Both blocks introduce latency and both sit in `ifChain`, and this is
   the one interaction the offline tests cannot reach. Audio must stay continuous with no lasting
   dropout. If it glitches, move `ifChain.addBlock(&adaptiveSquelch, false);` to **after**
   `ifChain.addBlock(&fmnr, false);` so the squelch is last in the IF chain, and retest.

- [ ] **Step 12: Commit**

```bash
git add decoder_modules/radio/src/radio_module.h
git commit -m "feat(radio): implement the SNR squelch mode

Enables the SQUELCH_MODE_SNR branch left stubbed upstream, with per-demod
persistence, a diagnostics readout and invalidation on samplerate,
bandwidth and demodulator changes."
```

---

## Definition of done

Phase 1 is complete when every box above is checked and:

- [ ] `squelch_unit_tests` passes with zero failures.
- [ ] `squelch_sweep` output is saved to `docs/adaptive-squelch/sweep-baseline.txt` and acceptance criterion 4 is met, or the failure is reported rather than tuned away.
- [ ] The full application builds and every check in Task 12 Step 11 passes.
- [ ] The branch `feature/adaptive-squelch` is ready for review.

Not in this plan and not to be started without a new design pass: AM carrier detection (spec §6) and the AM audio chain (spec §7).

---

## Appendix: reference sweep results

Produced by compiling this plan's own code exactly as written and running Task 11's sweep. Use it as
a sanity check: an implementation that lands here is behaving as designed.

**These numbers come from synthetic AWGN plus a tone with a syllabic envelope, not from real
airband IQ.** They demonstrate that the harness discriminates between estimators. They say nothing
about whether the algorithm is right for aviation voice — that is what Phase 0b and real captures
are for.

```
estimator       CNR   falseOpen%      miss%     FEC ms    OVER ms     MSC  chatter  ns/sample
------------------------------------------------------------------------------------------------
percentile     30.0       16.660      0.544        0.0      157.3       6       47       11.0
percentile     20.0       15.748      0.545        0.0      157.3       6       47        8.0
percentile     15.0       13.334      0.768        6.7      157.3       6       47        7.8
percentile     10.0       12.336      0.969       12.7      157.3       6       47        7.7
percentile      6.0        0.000    100.000     3000.0        0.0       0        0        7.6
percentile      3.0        0.000    100.000     3000.0        0.0       0        0        7.3

mcra           30.0       16.660      0.544        0.0      157.3       6       47        6.4
mcra           20.0        7.488      4.876        1.4        5.7      21       47        6.6
mcra           15.0        5.469     20.407        6.7        0.0      24       48        6.1
mcra           10.0        0.000    100.000     3000.0        0.0       0        0        6.0
mcra            6.0        0.000    100.000     3000.0        0.0       0        0        6.1
mcra            3.0        0.000    100.000     3000.0        0.0       0        0        6.1

gated_ema      30.0       16.660      0.544        0.0      157.3       6       47        6.1
gated_ema      20.0       15.748      0.545        0.0      157.3       6       47       12.0
gated_ema      15.0       13.334      0.768        6.7      157.3       6       47        8.3
gated_ema      10.0        0.000    100.000     3000.0        0.0       0        0        6.2
gated_ema       6.0        0.000    100.000     3000.0        0.0       0        0        6.1
gated_ema       3.0        0.000    100.000     3000.0        0.0       0        0        6.0
```

How to read it:

- **`percentile` is the most sensitive estimator**, and that is the headline. It still works at 10 dB
  C/N with a 0.97% miss rate where both other estimators miss everything. This is what satisfies
  acceptance criterion 4.
- **`mcra` clips mid-transmission.** At 20 dB it shows 21 mid-speech closures against `percentile`'s
  6, and by 15 dB it is missing a fifth of every transmission. Its rolling minimum follows the
  signal envelope upward during a burst, raising the floor under the very signal it is gating.
  Worth investigating in Phase 1b; `PRESENCE_DB` and `MAX_ABOVE_MIN_DB` are the parameters to look at
  first. Do not treat this as a bug to fix before shipping — `percentile` is the default.
- **The 16% `falseOpen%` at high C/N is the hang time, not a defect — and this limits the metric.**
  The scene has 2 s gaps and the gate holds for 350 ms after each burst, which is 17.5% of the gap.
  All three estimators report an identical 16.660% at 30 dB, which is the column saturating on gate
  timing rather than measuring anything about the estimator. **Whenever hang is a large fraction of
  the idle gap, `falseOpen%` measures timing, not discrimination — rank the estimators on `MSC` and
  on where the C/N cliff falls instead.** `OVER ms` reports the same hang effect directly, and if
  hang is retuned both columns move together.
- **The cliff below 10 dB is the syllabic envelope.** It dips to 0.25 amplitude, which is −12 dB, so
  a nominal 10 dB C/N spends part of each burst near 0 dB. That is realistic and it is why the
  metrics must always be read stratified by C/N.
- **CPU is 6-12 ns per sample** for all three, dominated by measurement noise rather than by the
  estimator. At a 15 kHz IF this is roughly 0.01% of one core. Cost is not a consideration here.
