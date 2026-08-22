# Adaptive Squelch Implementation Plan — Phase 2 (AM audio chain)

> Executed on branch `feature/am-audio-chain`. This document is the plan of
> record including as-executed amendments; each task lists the commit that
> landed it. Code lives in the tree rather than inline here.

**Goal:** The AM audio chain in design order — **2a** voice bandpass
(300–2700 Hz default, 2500/3000 measured against it), then **2b** mild
statistical noise reduction (capped log-MMSE gain, strong time/frequency
smoothing, automatic bypass on unreliable noise estimates) — test-driven
against the offline harness and wired into `afChain` for AM only.

**Spec:** `design.md` §7 (strict ordering, release-blocking rule), §8
(defaults), §9 (acceptance style). Requires Phase 1 only.

## Decisions locked before execution

| # | Decision |
|---|---|
| D1 | Upper cutoff fixed at 2700 Hz; 2500/3000 are harness sweep variants only |
| D2 | Voice chain applies to AM only |
| D3 | Legacy High Pass control stays; an active voice bandpass forces its effective state off while preserving the stored preference |
| D4 | Noise reduction is opt-in, OFF by default; WARMUP/ACTIVE/BYPASS state published atomically for GUI and harness |
| D5 | STOI implemented in-repo (`tools/squelch_harness/python/stoi_check.py`, numpy/scipy only) |
| D6 | Intelligibility material: synthetic formant speech plus optional local WAV import; recordings never committed |
| D7 | Gain rule is log-MMSE (Ephraim-Malah 1985); plain Wiener pre-declared as fallback if unstable |

## Global constraints

Carried from Phase 0/1 and extended:

- C++17; no dynamic allocation inside `process()`; buffers sized at init.
- Time constants in seconds derived from samplerate (`stream.h:75` returns
  `dataSize`; the design's `:70` cite points at the function declaration).
- Output independent of input chunking; every index computation hard-clamped;
  NaN treated as out-of-range.
- Every persistent state has invalidation rules.
- FFT permitted only inside Phase 2b blocks.
- Strictly ordered: 2a measured before any 2b work began.
- Release builds whenever CPU numbers matter.
- Transient metrics fed via randomized chunk sizes.
- Never weaken an existing assertion; the 52 Phase-0/1 tests must keep passing.

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `core/src/dsp/audio/voice_bandpass.h` | U-P2a samplerate-aware stereo bandpass | P2-5 |
| `core/src/dsp/noise_reduction/am_nr/stft.h` | U-P2b1 sqrt-Hann WOLA engine | P2-8 |
| `core/src/dsp/noise_reduction/am_nr/noise_psd.h` | U-P2b2 per-bin tracker + reliability | P2-9 |
| `core/src/dsp/noise_reduction/am_nr/gain_rule.h` | U-P2b3 capped log-MMSE gain | P2-10 |
| `core/src/dsp/noise_reduction/am_nr.h` | AmNR block composing b1-b3 | P2-11 |
| `tools/squelch_harness/src/wav_io.h` | RIFF reader/writer (PCM16/f32) | P2-1 |
| `tools/squelch_harness/src/audio_gen.h` | formant scenes, labels, AWGN | P2-2 |
| `tools/squelch_harness/src/audio_metrics.h` | chain metrics + triplet export | P2-3 |
| `tools/squelch_harness/python/stoi_check.py` | STOI scorer + batch gate | P2-4 |
| `tools/squelch_harness/python/make_abx.py` | blind trial builder/scorer | P2-13 |
| `tools/squelch_harness/src/af_sweep.cpp` | cutoff + NR sweeps, CPU bench | P2-7/P2-13 |
| `decoder_modules/radio/src/radio_module.h` | wirings, interlock, UI, persistence | P2-6/P2-12 |
| `docs/adaptive-squelch/phase2-bandpass-sweep.txt` | 2a measurement record | P2-7 |
| `docs/adaptive-squelch/phase2-nr-sweep.txt` | 2b measurement record | P2-13 |
| `docs/adaptive-squelch/phase2-abx-protocol.md` | human listening protocol | P2-13 |

## Tasks

Each task landed tests-red then green, one commit, message matching the task.

### P2-1 Branch + WAV I/O
`wav_io.h`: RIFF parse/walk, PCM16 and float32, mono downmix, truncation
rejected. Landed `04ee1f5b`.

### P2-2 Audio scene generator
`audio_gen.h`: Klatt-style three-resonator voiced source, word-segmented
labels inside transmission labels, noise at requested active SNR,
byte-deterministic. Landed `f577ab3d`.

### P2-3 Audio metrics + export
Median gap reduction, delay-compensated fidelity, jump guard, WAV triplet
export keyed to reported block delays. Landed `c12fa526`.

### P2-4 STOI tool
Taal et al. implementation; batch mode gates processed variants against
baselines. Landed `8f113d58`.

### P2-5 U-P2a VoiceBandpass unit
FIR subclass, odd-length Type-I taps via `taps::bandPass`, group delay
reported for exact compensation. Impulse test asserts the energy centroid
(the exact invariant of a symmetric response), not argmax - a modulated
sinc places its magnitude peak off-centre when the carrier crosses zero
there. Landed `64f38285`.

### P2-6 Bandpass wiring
Between HPF and de-emphasis; per-demod persistence; retap on audio
samplerate change; interlock greys the legacy checkbox without rewriting its
preference. Landed `62392242`.

### P2-7 Cutoff study + gate
2500/2700/3000 swept; gate PASS everywhere; 2700 justified, 3000 buys
nothing, 2500 spends margin. Record:
`phase2-bandpass-sweep.txt`. Landed `4b97328e`.

### P2-8 U-P2b1 STFT engine
Pow2 >= 10 ms frames, half-frame hop, sqrt-Hann pair (overlap sum exactly 1),
FFTW r2c/c2r per fm_if lifecycle. Output time-aligned; emission multirate
(emit what is final); bitwise chunk-invariant; first hop fades in under
single-lobe coverage as in any zero-history streaming STFT.
Landed `169151f1`.

### P2-9 U-P2b2 noise PSD tracker
Presence-gated recursive averaging per bin. Initial minimum-based
classification was rebuilt during bring-up after the chi-square convergence
test exposed structural bias; see the amendment log below.
Landed `c0de4c93`, amended by `efc49643`.

### P2-10 U-P2b3 capped log-MMSE gain
Ephraim-Malah gain, decision-directed xi, series/continued-fraction E1,
hard floor 10^(-9/20). Hostile inputs sanitized; outputs always finite and
in range. Wiener fallback not needed. Landed `532b46b0`.

### P2-11 AmNR block
Composes engine/tracker/rule; shared gain applied to both channels'
spectra; bypass leaves spectra untouched so bypassed audio is bit-identical
to disabled over the post-staleness era. Also fixed a latent accumulator-ring
hazard in stft.h (drain only on triggers let sub-hop callers lap the ring).
Landed `3e96824e`.

### P2-12 NR wiring
Opt-in checkbox off by default, persisted per demodulator; live state text;
samplerate rebuild; bandwidth-change reset mirrors squelch discipline.
Landed `dcc55ac9`.

### P2-13 NR evaluation + release gate
Conditions off/bp/bp_nr across five SNRs. Gate PASS at 15/20 dB SNR, FAIL
at 6/10 dB (-0.011/-0.013 STOI vs baselines) while delivering 7.7-8.2 dB of
additional suppression; recorded as a failure per design section 7, feature
stays opt-in/off. Includes make_abx.py and phase2-abx-protocol.md.
Landed `0e91e4e7` plus alignment/smoothing fixes `d0030192`.

### P2-14 Hygiene + docs
Zero warnings at `-Wall -Wextra` (unhidden base init overload; sign-compare
in freshly instantiated setTaps); allocwatch double-run shows allocation does
not scale with audio length; this document and README status.

## Defaults (Phase 2 rows of spec section 8)

| Parameter | Default | Status |
|---|---:|---|
| Bandpass cutoffs | 300-2700 Hz | 300 cited (SDRangel); 2700 extrapolated, measured in P2-7 |
| Transition width | 200 Hz | Extrapolation (~913 taps, ~9.5 ms group delay @48k) |
| NR frame / hop | pow2 >= 10 ms (512 @48k) / N/2, sqrt-Hann | Extrapolation |
| Decision-directed alpha | 0.98 | Extrapolation |
| Presence reference | slow conditional mean, ratio 3 (~5 dB) | Extrapolation; replaces rejected min-based scheme |
| Absence-gated tau (both directions) | 0.50 s | Extrapolation; asymmetric pair rejected on measured bias |
| Attenuation cap | 9 dB | Cited (design section 8) |
| Gain floor | 0.35481 | Derived |
| Reliability | warm-up 1 s; stale > 5 s; occupancy > 80%; escape resync 4 s | Extrapolation |
| NR default state | OFF | Decision D4 |

## CPU (Release, 48 kHz stereo)

| Stage | Measured | Budget note |
|---|---:|---|
| VoiceBandpass | ~125 ns/pair (~0.6% core) | Above the planned 50 ns estimate; 913-tap kernel runs at full SIMD efficiency, absolute cost negligible |
| AmNR | ~125 ns/pair additional | Within budget |
| Full bp+nr chain | ~255 ns/pair (~1.2% core) | Regression quantified per acceptance A9 |

## Acceptance criteria outcome

1. Chunking invariance bitwise: PASS (bandpass, STFT roundtrip, AmNR).
2. Timing/response scales across samplerates: PASS.
3. Cap enforced exactly: PASS (adversarial unit tests).
4. Cleanliness measured stratified: PASS as measurement; see records.
5. Intelligibility gate: **bandpass PASS; noise reduction FAIL at 6/10 dB**
   - recorded, not tuned away; ABX protocol shipped for human confirmation.
6. Bypass behaviour scripted and asserted: PASS.
7. No dynamic allocation in process(): PASS (allocwatch double-run).
8. Zero warnings: PASS after P2-14 fixes.
9. CPU measured in Release, budgets met or justified: PASS.
10. Invalidation matrix exercised: PASS (code paths + app build; live
    checklist items remain for the operator).
11. All 52 prior tests untouched and passing: PASS (suite now 93).

## As-executed amendments

- Scene generator shapes clean audio to the EN 300 086 channel and defines
  scene SNR in-channel; wideband-only definitions made correlation baselines
  read near-perfect regardless of hiss (P2-7).
- Unvoiced broadband phonemes added so correlation metrics see spectral
  spread comparable to real speech (P2-7).
- Exports align on displacement only: bandpass group delay displaces the
  waveform; STFT availability latency does not. An earlier sum-of-both shift
  produced an artificial ten-point STOI collapse (P2-13).
- Gain rule consumes the tracker's smoothed periodogram; raw frame powers
  gated enhancement within syllables (P2-13).
- Tracker classification rebuilt from biased minimum statistics onto a slow
  conditional mean with frame-level gating and a bounded escape resync
  (amendment to P2-9, commit `efc49643`).

## Out of scope

RNNoise beyond a research pointer (design section 7 2c). Phase 1b carrier
detection. Phase 0b impairment matrix. Any change to Phase 1
`adaptive_squelch` files.
