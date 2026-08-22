# Adaptive squelch and AM noise reduction

Design record for the first feature of this project.

| Document | What it is |
|---|---|
| [`design.md`](design.md) | The specification. What is being built, and **why each decision was made and what was rejected**. Section 2 records decisions that reverse earlier positions; read it before proposing changes. |
| [`implementation-plan.md`](implementation-plan.md) | Twelve test-driven tasks with the exact files, tests, code and commands for the offline harness and the `AdaptiveSquelch` block. Its appendix holds a reference sweep produced by compiling and running the plan's own code. |
| [`implementation-plan-phase2.md`](implementation-plan-phase2.md) | The Phase 2 plan: AM voice bandpass (2a) then capped log-MMSE noise reduction (2b), with the intelligibility release gate. |

## Status

Phase 0 and Phase 1 complete: harness plus SNR squelch shipped on
`main`. Phase 2 complete on `feature/am-audio-chain`: the AM voice
bandpass passes its gate at every tested SNR; the noise reduction
delivers near-cap suppression at low SNR but FAILED the intelligibility
gate there by 0.001-0.013 STOI against its baselines, so it ships opt-in
and OFF behind its checkbox until a candidate passes
`phase2-abx-protocol.md`. Measurements:
[`phase2-bandpass-sweep.txt`](phase2-bandpass-sweep.txt),
[`phase2-nr-sweep.txt`](phase2-nr-sweep.txt).

RNNoise remains research-only pending an A3E-domain intelligibility study
(design §7 2c; Valin 2018).

## Scope

In scope for the plan: an offline replay harness (Phase 0) and the `AdaptiveSquelch` DSP block
wired to the `SQUELCH_MODE_SNR` mode that upstream defined but left unimplemented (Phase 1).

Out of scope, each requiring its own design pass: AM carrier detection at an offset low IF
(Phase 1b, see design section 6) and the AM audio chain (Phase 2, see design section 7).
