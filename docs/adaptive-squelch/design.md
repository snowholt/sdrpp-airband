# Adaptive squelch and AM voice noise reduction — design specification

**Date:** 2026-08-21
**Repository:** `snowholt/sdrpp-airband`, derived from `AlexandreRouma/SDRPlusPlus` at commit `8c9f5ee`
**Status:** design approved for planning; no implementation started
**Inputs:** a source-reading spike, two independent literature research reports, and direct
inspection of the codebase.

> The two research reports behind sections 2 and 8 are third-party research-tool output and are kept
> outside this repository rather than redistributed. Every claim they support is cited in section 12
> against its primary source, so this document stands on its own.

---

## 1. Summary

Two features, built and validated in sequence:

| Project | Deliverable | Depends on |
|---|---|---|
| Phase 0 | Offline replay harness with ground-truth scoring | nothing |
| Phase 1 | `AdaptiveSquelch` DSP block wired to `SQUELCH_MODE_SNR` | Phase 0 |
| Phase 0b | Impairment matrix added to the harness | Phase 1 |
| Phase 1b | AM carrier acquisition and tracking, fused into the gate | Phase 0b, Phase 1 |
| Phase 2 | AM audio chain: voice bandpass, then statistical noise reduction | Phase 1 |

The ordering is not a preference. Both research reports independently arrive at the same build
order, and both state that the carrier detector must not be attempted before the energy gate has
been measured.

## 2. Decisions the research changed

Recording these explicitly, because each one reverses a position held before the research.

**2.1 Minimum statistics is out of the shipping scope.**
The original plan named Martin's minimum statistics as one of three estimators. Both reports
independently establish that Martin's bias-compensation terms are derived per frequency bin of a
smoothed periodogram, and that no primary source validates them on a single broadband time-power
envelope. The bias factor depends on the equivalent degrees of freedom, which change with the
channel filter's correlation, frame averaging and modulation. Shipping a scalar port under the name
"Martin minimum statistics" would overstate the evidence.

*Replaced by:* an MCRA-inspired soft recursive tracker, which uses the same minimum-guided idea
(local power over a running minimum drives a presence probability, which drives the update rate)
without inheriting bias constants that do not cover the scalar case.

*Retained as:* a documented research branch. If it is ever built, it requires Monte Carlo bias
calibration through the actual channel filter, decimator and frame-power code, and it must beat the
histogram on both error and adaptation speed before it ships.

**2.2 Rolling percentile is promoted from third choice to default.**
Both reports rank a rolling dB histogram percentile at or near the top: deterministic memory, exact
window expiration, no FFT, robust to impulses, transparent, and trivially replayable. It has no
distribution-specific bias table to get wrong.

**2.3 Gate-conditioned EMA must never ship alone.**
Two structural failures, both documented: staleness (a long transmission or a continuous ATIS/AWOS
channel starves the estimator of observations) and self-reinforcing lock-up (a false open freezes
the estimate that caused it). RTLSDR-Airband documents the staleness failure in production. The EMA
is retained as a measured baseline and as the smoothing layer, always paired with a slow shadow
tracker that never fully freezes and a watchdog that forces reacquisition.

**2.4 The carrier-at-DC method as originally conceived is not viable.**
At exactly zero IF, a genuine AM carrier and the receiver's own DC offset / LO leakage are both
constant complex terms. They are not identifiable from a single static zero-centred measurement.
This is an identifiability problem, not a calibration nuisance.

*Replacement:* deliberate offset low-IF. Rotate the wanted carrier to approximately +1.5 kHz in
complex baseband so the hardware DC spur remains at 0 Hz, then measure the carrier with a narrow
complex resonator. A numerical-rotation or LO-dither test distinguishes a real RF carrier (which
moves with the tuning reference) from an internal DC spur (which does not).

**2.5 The processing-gain claim needs correcting.**
`10*log10(B_ch / B_ENBW)` = 15.23 dB for a 300 Hz equivalent noise bandwidth in a 10 kHz channel.
This is correct only when the measurement bandwidth is the filter's **equivalent noise bandwidth**,
not its −3 dB width, and only for white noise. Critically, `P_narrow / P_total` is **not** a
carrier-to-noise ratio: the narrow band contains carrier, narrowband noise, DC leakage and
low-frequency modulation sidebands. The correct estimator subtracts the expected narrowband noise:

```
C_hat   = max(P_narrow - N0_hat * B_ENBW, 0)
CN_hat  = C_hat / (N0_hat * B_ch)
```

**2.6 A pre-roll buffer is required.**
A gain ramp removes the click at a gate transition but cannot restore a clipped consonant. The
direct solution to onset clipping is a short look-behind delay so a detected onset is emitted from
history. This adds latency to the audio path and is therefore user-configurable.

**2.7 There is a hard performance ceiling, and it must be documented.**
Energy detection cannot separate signal from noise below the SNR wall created by noise-power
uncertainty; more integration time does not help. For a bounded uncertainty of +/-0.5 dB the wall is
approximately -6.4 dB SNR; at +/-1 dB it is approximately -3.3 dB. Users must not be promised
performance the physics does not allow.

## 3. Verified codebase constraints

| Constraint | Evidence |
|---|---|
| The stream is already channel filtered; no adjacent-channel noise reference exists | `radio_module.h:88` `ifChain.init(vfo->output)`; `rx_vfo.h:24` |
| `gui::waterfall.selectedVFOSNR` is unusable as a DSP input (FFT rate, GUI coupled, selected VFO only) | `main_window.cpp:416` |
| `process()` receives a variable `count` | `stream.h:70` returns `dataSize` |
| `power_squelch.h` has zero samplerate awareness | 0 references in file |
| The mode socket already exists | `demod.h:26`, `radio_module.h:55`, `radio_module.h:675` |
| AM uses `REF_CENTER`, placing the nominal carrier at 0 Hz | `demodulators/am.h:83` |
| AM: 15 kHz IF rate, 10 kHz default bandwidth, 1-15 kHz range | `demodulators/am.h:76-80` |
| No noise reduction exists for AM | `demodulators/am.h:87-88` both return false |
| `afChain` has a high pass but no low pass | `radio_module.h:102` |
| `chain` imposes no sample-count parity across blocks; a latency-introducing block is safe to hot-enable | `chain.h` `enableBlock`; `fm_if.h` already buffers `_bins-1` samples inside `ifChain` |
| DSP block headers can be driven offline with no stream or thread | verified empirically: `init(NULL, ...)` plus a direct `process()` call compiles and runs standalone |
| The repository has no test framework | only `libcorrect/tests` (C) and `dsp/bench/speed_tester.h` |

## 4. Phase 0 — offline replay harness

Built first, and built by the project owner rather than by the implementer, because validation that
is authored by the party being validated is not validation.

**Location:** `tools/squelch_harness/`, its own CMake target, guarded by
`OPT_BUILD_SQUELCH_HARNESS` defaulting to `OFF`. It links the DSP headers directly and does not
require an SDR++ GUI build.

**Phase 0 scope — only what is needed to rank the three estimators:**

1. **Synthetic source.** Complex AWGN at selectable C/N, plus a gated speech-like envelope with
   ground-truth transmission start and stop sample indices. Realistic modulation is not required to
   rank noise-floor estimators.
2. **Driver.** Instantiates the block, feeds it with a **randomised `count` per call** drawn from a
   wide range. This is the direct test of constraint §3 row 3, and it must be present from the first
   run rather than added later.
3. **Trace output.** A CSV snapshot of the diagnostics, one row per `process()` call and at most one
   per 10 ms: `t, framePowerDb, noiseFloorDb, snrDb, occupancy, state, transitionReason`. The block
   publishes diagnostics every frame, but a caller can only observe them at call boundaries, so the
   trace is a sampled view rather than a complete frame log.
4. **Scorer.** Reports the timing metrics below against ground truth.
5. **Sweep runner.** Grid over estimator and parameters, emitting a comparison table.

**Phase 0b scope — required before Phase 1b, not before Phase 1:** A3E modulation of real speech,
receiver DC offset vector injection, carrier frequency offset and Doppler ramps, flat and frequency
selective fading, AGC steps, impulse bursts, adjacent-channel splatter, and a reader for real
labelled IQ captures (SDR++'s own recorder output plus a sidecar JSON of hand-labelled intervals).

**Metrics.** Standard VAD timing-error terminology is used deliberately so results are comparable
with published work.

| Metric | Definition |
|---|---|
| False-open time | Fraction of noise-only time passed unmuted |
| Miss rate | Fraction of labelled transmission time muted |
| FEC (front-end clipping) | Speech duration lost after a true onset |
| MSC (mid-speech clipping) | Number and duration of closures inside a transmission |
| OVER (overhang) | Noise duration passed after a true end |
| NDS | Noise detected as speech |
| Chatter | Gate transitions per transmission |
| Switching-artifact energy | Integrated output energy in a window around each transition, minus expected signal energy |
| Detection delay | Distribution, not mean |
| Recovery time | Time to reacquire after an AGC or noise-floor step |
| CPU | Nanoseconds per sample |

All metrics are reported **stratified by C/N**, never as a single aggregate.

**Cost function.** Defaults are selected by an explicit weighting, not by one accuracy figure.
Penalty order, heaviest first: missed transmissions and onset clipping, then false-open time, then
overhang noise, then chatter.

## 5. Phase 1 — the `AdaptiveSquelch` block

**New file:** `core/src/dsp/noise_reduction/adaptive_squelch.h`
**Edited file:** `decoder_modules/radio/src/radio_module.h`

```
                    variable count                          fixed 10 ms frames
  in ──┬──────────────────────────────────────┐
       │                                      │
       └─► U1 frame meter ──► framePowerDb ───┼──► U2 noise floor estimator ──► noiseFloorDb
                                              │         P | M | E (selectable)
                                              │                   │
                                              │                   ▼
                                              │        snrDb = framePowerDb - noiseFloorDb
                                              │                   │
                                              │                   ▼
                                              └──► U4 decision state machine ──► gain
                                                    CLOSED/QUALIFY/OPEN/HANG
                                                            │
   U3 pre-roll delay line ──────────────────────────────────┴──► out (in * ramped gain)

   U5 lifecycle and invalidation      U6 diagnostics snapshot
```

Each unit is separately constructible and separately testable. That is a requirement, not a
stylistic preference: the harness must be able to hold everything constant except the estimator.

### U1 — frame meter

Consumes a variable `count`, emits fixed-duration frames. Maintains a partial-frame accumulator
across `process()` calls so that frame boundaries are independent of block boundaries.

- Frame length: 10 ms, derived as `round(0.010 * samplerate)` samples. Never a fixed sample count.
- Statistic: mean `|x|^2` over the frame, converted to dB.
- Optional winsorization of the top 5% of sample magnitudes within a frame, for impulse robustness.
  Off by default; measured in the harness before being made a default.

**This unit is what makes every time constant in the design samplerate-correct and independent of
the variable `count`.** No other unit may see raw sample counts.

### U2 — noise floor estimator

Common interface. One method, one return value, no hidden coupling to the gate other than the
explicit argument:

```
float update(float framePowerDb, bool gateOpen) -> noiseFloorDb
void  reset()
```

**P — rolling dB histogram percentile (default).**
Fixed histogram, 0.25 dB bins spanning −140 to 0 dBFS (560 bins), plus a ring buffer of 1000 frame
bin indices giving an exact 10 s sliding window. Update is O(1): decrement the departing bin,
increment the arriving one. Query walks bins accumulating counts to the target rank; O(560) worst
case at 100 queries per second is negligible.

Report the 20th percentile by default. Note the occupancy assumption this encodes: a q-quantile
remains noise-representative only while at least q of the window is noise-only, so the 20th
percentile tolerates roughly 80% channel occupancy before the statistic is signal-dominated.

**Both bin index computations must be hard-clamped at both ends.** SDRPlusPlusBrown issue #73 is a
reported out-of-bounds histogram-index bug in exactly this kind of code. This is a required
behaviour with a named precedent, not a defensive-programming suggestion.

**M — MCRA-inspired soft recursive tracker.**
Maintain a smoothed frame power `S`, a slowly tracked lower envelope `M`, and a presence
probability `p` derived from the ratio `S/M` with temporal smoothing. Update the noise estimate with
a probability-controlled coefficient `alpha_d = alpha + (1 - alpha) * p`, so the estimate slows but
never fully stops during signal. Required guards: a non-zero maximum adaptation rate, a hard floor,
robust clipping so a single impulse cannot permanently raise the estimate, and a watchdog.

Documented as MCRA-*inspired*. The published MCRA and IMCRA results are per-bin STFT results and do
not transfer unchanged to a scalar broadband envelope.

**E — gate-conditioned EMA with shadow tracker and watchdog.**
Fast tracker, `tau` = 0.75 s, updated only on high-confidence noise frames. Slow shadow tracker,
`tau` = 20 s, always updating, upward adaptation confidence-gated. Watchdog: after a configurable
maximum continuous-open interval, permit a slow leak update or schedule reacquisition.

Retained as the cheap measured baseline. **Documented as not suitable as a default**, with the
staleness and self-lock reasons stated in the source comment.

### U3 — pre-roll delay line

A ring buffer holding the most recent N milliseconds of input. Output lags input by that amount, so
that when the state machine declares an onset it emits from history rather than from the present.

- Default 40 ms, user-configurable, **and user-disableable**.
- When disabled, delay is zero and onset clipping is accepted.
- The UI must state the latency cost in plain words. This is the only squelch mode in SDR++ that
  shifts the audio path, so it must not be silent.
- Enabling the block mid-stream produces a one-time gap of one pre-roll interval. Acceptable, and to
  be noted in the source.

### U4 — decision state machine

```
CLOSED --(snr > openMargin for qualifyMs)--> QUALIFY --> OPEN
OPEN --(snr < closeMargin)--> HANG --(hangMs elapsed, still below)--> CLOSED
HANG --(snr > openMargin)--> OPEN
```

- Two thresholds, never one. `openMargin` and `closeMargin` are both dB above the estimated noise
  floor; their difference is the hysteresis.
- `qualifyMs` prevents impulse-triggered opens.
- `hangMs` prevents word-ending truncation.
- Transitions apply a raised-cosine gain ramp rather than `memcpy` / `memset`.
- Every transition records a `transitionReason` for the diagnostics snapshot and the harness trace.

### U5 — lifecycle and invalidation

**Every persistent state needs an invalidation rule.** This is the single change most likely to
prevent unexplainable field failures.

New methods on the block: `setSamplerate(double)`, `setBandwidth(double)`, `reset()`.

Invalidate estimator state, histogram contents, delay line and state machine on: samplerate change,
bandwidth change, VFO retune, source gain change, source start/stop, and demodulator change.

Hook points to wire: `radio_module.h:519` (`vfo->setSampleRate`), `:569` (`vfo->setBandwidth`),
`:587`. `setSquelchLevel` at `radio_module.h:706-708` calls `powerSquelch.setLevel` unconditionally
and is deliberately left alone: `setSquelchMode` disables every squelch block before enabling one, so
in SNR mode the power squelch is not in the chain and the stored level is inert. The adaptive block
gets its own setters rather than sharing that path.

### U6 — diagnostics

One trivially-copyable snapshot struct, written by the DSP thread and read without locking:

```
noiseFloorDb, framePowerDb, snrDb, state, lastTransitionReason,
occupancyPercent, carrierConfidence, carrierOffsetHz
```

Consumed by two clients from the same source: the GUI diagnostics area (there is already a switch
for this at `radio_module.h:325`) and the harness CSV. Serving both from one struct is what makes
the observability cheap.

Prior art across SDR software is consistently opaque about its estimator internals. Exposing the
noise estimate, the current SNR, the state and the reason for the last transition is a deliberate
differentiator and is essential for field tuning.

### UI and configuration

- Uncomment `squelchModes.define("snr", "SNR", SQUELCH_MODE_SNR);` at `radio_module.h:55`.
- Implement `case SQUELCH_MODE_SNR:` at `radio_module.h:675`.
- Controls: estimator selector, open margin (dB above floor), hysteresis (dB), hang (ms), pre-roll
  (ms, with an off position), and a diagnostics readout.
- Persist all of it per demodulator, following the existing pattern at `radio_module.h:469-476`.
- **Occupancy warning.** When measured channel occupancy exceeds roughly 80%, the adaptive estimate
  is no longer trustworthy. Surface this and recommend the fixed-threshold power mode. This is the
  documented ATIS/AWOS failure and it must be visible rather than silent.

## 6. Phase 1b — AM carrier detection (separate handoff)

Deliberately not part of the Phase 1 handoff. This is a carrier acquisition and tracking subsystem,
not a small addition, and it must be built against a harness that can already synthesize Doppler
ramps and DC spurs.

1. **Offset low-IF.** Numerically rotate the wanted carrier to approximately +1.5 kHz so the
   hardware DC spur remains at 0 Hz. It stays comfortably inside a 10 kHz complex channel.
2. **Acquisition.** A bank of roughly 9-11 complex resonators spanning +/-500 Hz around the expected
   carrier, at 300 Hz ENBW, over 50 ms blocks; two consecutive valid blocks to declare acquisition.
   The +/-500 Hz range covers approximately 120 Hz of 1 ppm oscillator error at 120 MHz plus up to
   roughly 150 Hz of civil Doppler, with margin.
3. **Tracking.** Complex mixer plus a coherent accumulator and a second-order PLL, loop bandwidth
   about 15 Hz. **The NCO must be clamped to the acquisition range and the loop integrator cleared
   on unlock**, and the loop update gated on carrier confidence. An ungated loop wanders on noise
   when no carrier is present.
4. **Estimate.** Use the noise-subtracted form from §2.5. Never ship the raw narrow-to-total ratio
   as a carrier-to-noise figure.
5. **DC baseline.** Maintain a slowly updated per-device, per-gain DC offset vector estimated during
   confident-noise periods and frozen during carrier confidence. Require carrier evidence not
   explained by that baseline.
6. **Fusion.** Open when the carrier feature is valid and locked and the total energy is plausible,
   with a strong-energy fallback path. The carrier feature takes precedence when locked; energy
   remains the fallback. The carrier detector is a secondary feature, never the sole gate.

**Acceptance gate specific to this phase:** an injected-signal calibration and a numerical-rotation
or LO-dither test must demonstrate that a real RF carrier is distinguished from a receiver DC spur.
Until that passes, this feature does not ship enabled.

## 7. Phase 2 — AM audio chain

Strictly ordered. Each step must be measured before the next is begun.

**2a. Voice bandpass.** `afChain` currently has a 300 Hz high pass and no low pass
(`radio_module.h:102`). Add a proper voice bandpass for AM, defaulting to 300-2700 Hz, with 2500 Hz
and 3000 Hz upper cutoffs measured against it. Deterministic, negligible CPU, and structurally
incapable of producing musical noise. SDRangel documents that its 300 Hz lower cutoff improves weak
air-traffic readability.

**2b. Mild statistical noise reduction.** A Wiener or log-MMSE gain with a **hard attenuation cap**,
9 dB initially, strong time and frequency smoothing, and automatic bypass when the noise estimate is
unreliable. FFT based; FFTW is already a core dependency via `fm_if.h`. Spectral subtraction without
over-subtraction and a spectral floor is not acceptable because of musical noise.

**2c. RNNoise.** Research branch only. The stock model is trained on acoustic noise mixtures; no
controlled study was found showing it preserves intelligibility on demodulated A3E with RF hiss,
selective fading, clipping and impulse interference. Requires a domain intelligibility study before
it is considered.

**Release-blocking rule for all of Phase 2:** an improvement in perceived cleanliness that comes
with a measured loss of intelligibility is a failure, not a trade-off. Single-channel enhancers are
documented to improve quality scores while reducing intelligibility on already-degraded speech.
Testing must include word recognition, not only quality metrics.

## 8. Parameter defaults

Every value is marked with its evidence status. **Cited** means the number appears in a cited
implementation or standard. **Extrapolation** means it is a starting value to be tuned in the
harness; a nearby cited system may motivate the range without prescribing the value.

| Parameter | Default | Status |
|---|---:|---|
| Frame length | 10 ms | Extrapolation; matches G.729B's 10 ms VAD cadence |
| Frame statistic | mean \|x\|^2 in dB | Extrapolation |
| Histogram range / bin | −140 to 0 dBFS, 0.25 dB, clamped | Extrapolation; clamping required by Brown issue #73 |
| Rolling window | 10 s (1000 frames) | Extrapolation |
| Noise percentile | 20th | Extrapolation; encodes an 80% maximum occupancy assumption |
| Fast EMA time constant | 0.75 s (alpha = 0.9868 at 10 ms) | Extrapolation |
| Slow shadow time constant | 20 s (alpha = 0.99950 at 10 ms) | Extrapolation |
| Open margin above floor | 9.5 dB | **Cited**: RTLSDR-Airband default SNR threshold 9.54 dB, documented ~10 dB behaviour |
| Close margin | 6.5 dB | Extrapolation |
| Hysteresis | 3 dB (derived: open minus close) | Derived, not independently configurable |
| Open qualification | 30 ms | Extrapolation; within SDRangel's cited 10-500 ms gate range |
| Pre-roll | 40 ms, disableable | Extrapolation |
| Hang | 200 ms | Extrapolation; AMR VAD1 cited hangovers are 80 ms and 140 ms |
| Release after hang | 150 ms | Extrapolation |
| Gain ramp | 5 ms raised-cosine half window | Extrapolation; no standard found |
| Occupancy warning threshold | 80% | Extrapolation; follows from the percentile choice |
| AM carrier offset IF | +1.5 kHz | Extrapolation |
| AM acquisition search | +/-500 Hz | Extrapolation; 1 ppm at 120 MHz = 120 Hz, plus <=150 Hz civil Doppler |
| AM acquisition ENBW | 300 Hz | Extrapolation; yields the 15.23 dB white-noise bandwidth gain in 10 kHz |
| AM locked ENBW | 100-150 Hz | Extrapolation |
| AM PLL loop bandwidth | 15 Hz | Extrapolation |
| AM audio bandpass | 300-2700 Hz | Extrapolation; SDRangel cites the 300 Hz lower cutoff |
| NR attenuation cap | 9 dB | Extrapolation; deliberately conservative |

**Reference figures, for calibrating expectations rather than for configuration:**

- Doppler at 120 MHz: 102.9 Hz at 500 kn; approximately 123-144 Hz at 600-700 kn. Ground stations
  shift 0 Hz.
- Energy detection at P_FA = 1%, P_D = 90%: approximately 1,390 independent complex samples at
  −10 dB SNR, 13,300 at −15 dB, 131,000 at −20 dB. At a 10 kHz effective noise bandwidth that is
  0.14 s, 1.33 s and 13.1 s. Use effective noise bandwidth, not nominal sample rate.
- SNR wall, `gamma_wall = rho - 1/rho`: approximately −6.4 dB for +/-0.5 dB noise uncertainty,
  approximately −3.3 dB for +/-1 dB.
- 12 dB SINAD is a receiver sensitivity test point, not a squelch threshold. At 12 dB SINAD,
  S/(N+D) is approximately 11.7 dB.

## 9. Acceptance criteria

Phase 1 is complete when all of the following hold in the harness.

1. Correct behaviour under randomised `count`: identical frame-level output for the same input
   regardless of how the input is chunked.
2. Every time constant scales correctly across at least three samplerates.
3. All three estimators run behind one interface with an otherwise identical decision layer, and the
   harness produces a stratified comparison table across C/N.
4. Estimator P dominates estimator E somewhere in the sweep: either a lower false-open time at an
   equal miss rate, **or** a usable result at a C/N where E fails outright. A reference run of the
   plan's own code shows the second form — at 10 dB C/N, P holds a 0.97% miss rate while E misses
   100%. If neither form appears, the design assumption is wrong and must be revisited before
   shipping.
5. Zero out-of-bounds histogram accesses under an input sweep that deliberately exceeds the
   configured dB range at both ends.
6. Switching-artifact energy at gate transitions is measurably lower than `power_squelch.h` on the
   same input.
7. With pre-roll enabled, front-end clipping is measurably lower than with it disabled.
8. Every invalidation path is exercised and leaves no stale state.
9. CPU cost per sample is measured and reported; no target is set, but a regression against
   `power_squelch.h` must be quantified and justified.
10. No compiler warnings at the project's existing warning level; no dynamic allocation inside
    `process()`.

## 10. Explicitly out of scope

- Martin minimum statistics as a shipping estimator (§2.1).
- Any use of `gui::waterfall.selectedVFOSNR`.
- FFT anywhere in the squelch path. Phase 2b may use FFT; Phase 1 may not.
- `SQUELCH_MODE_DCS_MUTE` and `SQUELCH_MODE_DCS_DECODE`, which are also stubbed but unrelated.
- Any wider refactor of `radio_module.h`. Rough edges noted during the spike
  (`ctcss_squelch.h`'s "DEBUG ONLY" muting, `am.h`'s missing `setSamplerate`) are recorded but not
  in scope. `am.h`'s `setSamplerate` may become necessary in Phase 2 and will be scoped then.
- Upstream contribution. The fork exists because upstream declines code contributions and prohibits
  AI-assisted work; nothing here is written with a pull request in mind.

## 11. Risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| The AM carrier is not where the design assumes | Phase 1b invalid | Phase 0b measures actual carrier offset distribution from real captures before Phase 1b begins |
| Real airband occupancy exceeds 80% on the channels of interest | Percentile estimator degrades | Occupancy warning plus fixed-threshold fallback; measure occupancy from captures |
| Receiver DC spur cannot be separated even at +1.5 kHz offset | Carrier feature unshippable | Injected-signal and rotation tests are a hard gate before enabling |
| Pre-roll latency is unacceptable in use | Feature disabled by users | It is user-disableable by design |
| Harness scope grows and blocks the handoff | Schedule | Phase 0 is deliberately minimal; the impairment matrix is Phase 0b |

**Open question, blocking the implementation handoff only:** which model implements Phase 1, and
does it have repository access? An agentic implementer with the repo receives a task brief with
file paths and a build-and-test loop. A chat implementer receives a self-contained document with the
relevant code inline and a narrower single-shot scope. The specification above is written to be
model-agnostic; the handoff prompt cannot be.

## 12. References

Cited sources behind the numbers in this document. Full annotated bibliographies are in the two
research reports, which are not redistributed here.

- R. Martin, "Noise Power Spectral Density Estimation Based on Optimal Smoothing and Minimum
  Statistics", IEEE Trans. Speech and Audio Processing 9(5), 2001. DOI 10.1109/89.928915
- R. Martin, "Bias Compensation Methods for Minimum Statistics Noise Power Spectral Density
  Estimation", Signal Processing 86, 2006
- I. Cohen and B. Berdugo, "Noise Estimation by Minima Controlled Recursive Averaging for Robust
  Speech Enhancement", IEEE Signal Processing Letters 9(1), 2002
- I. Cohen, "Noise Spectrum Estimation in Adverse Environments: Improved Minima Controlled Recursive
  Averaging", IEEE Trans. Speech and Audio Processing 11(5), 2003. DOI 10.1109/TSA.2003.811544
- R. Tandra and A. Sahai, "SNR Walls for Signal Detection", IEEE JSTSP 2(1), 2008
- F. F. Digham, M.-S. Alouini and M. K. Simon, "On the Energy Detection of Unknown Signals over
  Fading Channels", IEEE Trans. Communications 55(1), 2007
- D. R. Pauluzzi and N. C. Beaulieu, "A Comparison of SNR Estimation Techniques for the AWGN
  Channel", IEEE Trans. Communications 48(10), 2000. DOI 10.1109/26.871393
  (evaluates PSK in AWGN; **not** a validation of moment-based CNR estimation for analogue A3E)
- B. Razavi, "Design Considerations for Direct-Conversion Receivers", IEEE TCAS-II 44(6), 1997
- R. Svitek and S. Raman, "DC Offsets in Direct-Conversion Receivers", IEEE Microwave Magazine, 2005
- S. M. Kay, "A Fast and Accurate Single Frequency Estimator", IEEE TASSP 37(12), 1989
- R. Jain and I. Chlamtac, "The P-Square Algorithm for Dynamic Calculation of Quantiles and
  Histograms Without Storing Observations", CACM 28(10), 1985
- S. Boll, "Suppression of Acoustic Noise in Speech Using Spectral Subtraction", IEEE TASSP 27(2),
  1979
- M. Berouti, R. Schwartz and J. Makhoul, "Enhancement of Speech Corrupted by Acoustic Noise",
  ICASSP 1979
- Y. Ephraim and D. Malah, MMSE-STSA, IEEE TASSP 32(6), 1984; log-MMSE, IEEE TASSP 33(2), 1985
- Y. Hu and P. C. Loizou, "Subjective Comparison and Evaluation of Speech Enhancement Algorithms",
  Speech Communication 49, 2007
- J.-M. Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement", 2018
- 3GPP TS 26.094, AMR Voice Activity Detector (VAD1 hangover constants)
- ITU-T G.729 Annex B (10 ms VAD cadence, 16 x 8-frame minima history)
- ETSI EN 300 086-1 (land mobile receiver measurement methods; SINAD)
- RTLSDR-Airband, manual squelch documentation (~10 dB open margin, 9.54 dB default SNR threshold,
  documented continuous-channel failure)
- SDRangel NFM and AM demodulator documentation (10-500 ms gate range; 300 Hz audio lower cutoff)
- SDRPlusPlusBrown, `misc_modules/noise_reduction_logmmse/src` and issue #73 (histogram index bug)
- ATCOSIM and ATCO2 air traffic control speech corpora
