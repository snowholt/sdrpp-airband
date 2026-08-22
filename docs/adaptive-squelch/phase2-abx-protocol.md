# Phase 2 listening protocol - blind ABX and word recognition

Design reference: `design.md` section 7 release-blocking rule. An
improvement in perceived cleanliness that comes with a measured loss of
intelligibility is a failure, not a trade-off, and testing must include
word recognition rather than quality metrics alone.

Status: the numeric STOI sub-gate FAILED at 6 and 10 dB SNR (see
`phase2-nr-sweep.txt`). This protocol exists so that any future
candidate that passes the numeric gate also faces a human check before
shipping, and so the current failure can be confirmed or contradicted
by ear.

## Materials

Two tiers, matching implementation-plan-phase2 decision D6:

1. **Synthetic tier (available now).** The harness exports WAV triplets
   of EN 300 086-shaped formant speech plus wideband hiss: unprocessed,
   bandpassed, and bandpassed+NR, at 20/15/10/6/0 dB in-channel SNR.
   Synthetic material carries no lexical content, so it supports
   discrimination and preference judgements only.

2. **Real-material tier (required for shipping word-recognition
   evidence; automation is a documented follow-up).** Local recordings
   are never committed. Convert with sox to 48 kHz mono float:
   `sox input.any -r 48000 -c 1 -e float out.wav`, feed through the
   chain conditions using the same af_sweep pipeline via the harness
   WAV import path, then run the trial builder below. Transcription
   scoring applies to this tier: pick material with known word lists,
   score words correct per condition, and require no error increase
   from processing.

## Trial generation

```
./build-harness-rel/tools/squelch_harness/af_sweep --export-dir /tmp/phase2
python3 tools/squelch_harness/python/make_abx.py build \
    --manifest /tmp/phase2_nr_manifest.csv --out TRIALS_DIR \
    --trials 12 --conditions bp_nr --seed <N>
```

Each trial contains A.wav, B.wav (degraded-unprocessed vs processed in
seeded hidden order) and REF.wav (the clean ground truth, for familiarisation
only - it is not part of the judgement). `key.csv` inside the trial folder
holds the mapping and stays hidden from the listener.

## Listening procedure

1. Listen to REF.wav once for calibration.
2. For each trial, listen to A.wav then B.wav, in order, as many times as
   needed but always in A-then-B order.
3. Record two judgements per trial on the answer sheet (`trial,cleaner`
   rows, values A or B):
   a. Which segment sounds cleaner (forced choice, no ties).
   b. For real-material trials only, transcribe every word understood in
      B.wav.
4. The listener must not see key.csv, the manifest, or which condition
   maps to which file.

## Scoring

```
python3 tools/squelch_harness/python/make_abx.py score \
    --answers answers.csv --key TRIALS_DIR/key.csv
```

Reports how often the processed variant was preferred against a binomial
sign test at chance.

## Acceptance

All three must hold:

| Check | Threshold |
|---|---|
| STOI gate (numeric) | processed within 0.01 of both baselines at every tested SNR |
| ABX preference | no significant preference FOR the unprocessed variant (binomial p >= 0.05 against processed-preferred) |
| Word recognition (real tier) | transcription errors(processed) <= errors(unprocessed) |

A failure of any one blocks default-on rollout of the noise reduction;
the feature remains available behind its opt-in checkbox while failing,
since it ships disabled by default.
