# Adaptive squelch and AM noise reduction

Design record for the first feature of this project.

| Document | What it is |
|---|---|
| [`design.md`](design.md) | The specification. What is being built, and **why each decision was made and what was rejected**. Section 2 records decisions that reverse earlier positions; read it before proposing changes. |
| [`implementation-plan.md`](implementation-plan.md) | Twelve test-driven tasks with the exact files, tests, code and commands. Its appendix holds a reference sweep produced by compiling and running the plan's own code. |

## Status

Design complete. Implementation not started.

## Scope

In scope for the plan: an offline replay harness (Phase 0) and the `AdaptiveSquelch` DSP block
wired to the `SQUELCH_MODE_SNR` mode that upstream defined but left unimplemented (Phase 1).

Out of scope, each requiring its own design pass: AM carrier detection at an offset low IF
(Phase 1b, see design section 6) and the AM audio chain (Phase 2, see design section 7).
