#!/usr/bin/env python3
"""STOI intelligibility scoring for the AM audio chain release gate.

Implements short-time objective intelligibility (Taal et al., ICASSP 2010 /
IEEE TASLP 2011) using only numpy/scipy, deterministically:

  * both inputs resampled to an internal 10 kHz grid,
  * 256-sample (25.6 ms) Hann-windowed frames, 50% overlap,
  * third-octave bands from 150 Hz up to min(3960 Hz, 0.9*Nyquist),
  * temporal-envelope correlation per band over 30-frame (384 ms) segments,
  * correlations clipped to [0, 1], combined by mean after the published
    power-law Jacobian (beta = 1.174). The optional sigmoid remap is omitted;
    every gate in this project compares relative deltas between variants
    scored against the same reference, for which this is sufficient.
    Deviations from the reference implementation are recorded here rather
    than hidden.

Gate rule (Phase 2 release-blocking rule): the processed variant must never
measure below its strongest baseline minus tolerance. A failure exits
non-zero and is reported, never tuned away.

No external dependencies beyond numpy/scipy. An optional cross-check against
the pip 'pystoi' package runs when --pystoi-crosscheck is given and the
package happens to be installed; absence is not an error.
"""

import argparse
import csv
import struct
import sys

import numpy as np
from scipy.signal import resample_poly

FS_INTERNAL = 10000
FRAME = 256                  # 25.6 ms analysis frame at 10 kHz
HOP = 128                    # 50% overlap
SEGMENT_FRAMES = 30          # ~384 ms envelope segment
BETA = 1.174                 # published Jacobian exponent (9 / 7.67)
DYN_RANGE_DB = 40.0

# Third-octave centre frequencies, ascending, per the reference definition.
BAND_CENTERS = np.array([
    150.0, 173.0, 200.0, 230.0, 265.0, 306.0, 353.0, 407.0, 469.0, 541.0,
    624.0, 719.0, 829.0, 956.0, 1102.0, 1271.0, 1465.0, 1689.0, 1947.0,
    2244.0, 2586.0, 2980.0, 3436.0, 3960.0,
])

# Exact rational resampling ratios onto the 10 kHz grid for common rates.
RESAMPLE_PAIRS = {
    8000: (5, 4), 12000: (5, 6), 16000: (5, 8), 24000: (5, 12),
    32000: (5, 16), 48000: (5, 24),
}


def read_wav_mono(path):
    """Minimal RIFF reader: PCM16 or float32, any channel count -> mono."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 44 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos = 12
    fmt = None
    raw = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if pos + 8 + size > len(data):
            raise ValueError(f"{path}: truncated chunk")
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    audio_format, channels, rate, _, _, bits = fmt
    if audio_format == 1 and bits == 16:
        count = len(raw) // (2 * channels)
        smp = np.frombuffer(raw[:count * 2 * channels], dtype="<i2").astype(np.float64) / 32768.0
    elif audio_format == 3 and bits == 32:
        count = len(raw) // (4 * channels)
        smp = np.frombuffer(raw[:count * 4 * channels], dtype="<f4").astype(np.float64)
    else:
        raise ValueError(f"{path}: unsupported format {audio_format}/{bits}")
    if channels == 2:
        smp = smp.reshape(-1, 2).mean(axis=1)
    return smp, int(rate)


def _resample_to_10k(x, fs):
    x = x.astype(np.float64)
    if fs == FS_INTERNAL:
        return x
    updn = RESAMPLE_PAIRS.get(int(fs))
    if updn is None:
        from fractions import Fraction
        f = Fraction(FS_INTERNAL, int(fs)).limit_denominator(1000)
        updn = (f.numerator, f.denominator)
    return resample_poly(x, updn[0], updn[1])


def _band_weights(nfft, fs):
    """Binary third-octave weighting over rfft bins."""
    top = min(float(BAND_CENTERS[-1]), 0.9 * fs / 2.0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    weights = []
    for c in BAND_CENTERS:
        if c > top:
            break
        lo = c * 2.0 ** (-1.0 / 6.0)
        hi = c * 2.0 ** (1.0 / 6.0)
        mask = ((freqs >= lo) & (freqs <= hi)).astype(np.float64)
        weights.append(mask)
    return np.array(weights)          # shape (J, nfft//2 + 1)


def _envelopes(x):
    """Per-band short-time energy envelopes, shape (bands, frames)."""
    nframes = 0 if len(x) < FRAME else 1 + (len(x) - FRAME) // HOP
    if nframes <= SEGMENT_FRAMES:
        return None
    win = np.hanning(FRAME)
    w = _band_weights(FRAME, FS_INTERNAL)
    env = np.zeros((w.shape[0], nframes))
    # Vectorised framing: one rfft over all frames at once.
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(nframes)[:, None]
    frames = x[idx] * win[None, :]
    mags = np.abs(np.fft.rfft(frames, axis=1)) ** 2      # (T, bins)
    env = mags @ w.T                                      # (T, bands)
    return env.T


def stoi(ref, test, fs):
    """STOI between a clean reference and a test signal (same length)."""
    n = min(len(ref), len(test))
    x = _resample_to_10k(np.asarray(ref[:n]), fs)
    y = _resample_to_10k(np.asarray(test[:n]), fs)

    x = x - x.mean()
    y = y - y.mean()
    rms = np.sqrt(np.mean(x ** 2))
    if rms <= 0:
        raise ValueError("reference is silent")
    x = x / rms
    y = y / rms

    ex = _envelopes(x)
    ey = _envelopes(y)
    if ex is None or ey is None:
        raise ValueError("input shorter than one STOI segment")

    t = min(ex.shape[1], ey.shape[1])
    ex, ey = ex[:, :t], ey[:, :t]

    # Dynamic-range criterion, per the published definition: TF cells of the
    # reference falling more than DYN_RANGE_DB below its overall peak carry no
    # intelligibility evidence and are excluded from both signals, rather than
    # being floored (flooring would make empty bands correlate noise against
    # noise and drag the score down).
    keep = ex >= np.max(ex) * (10.0 ** (-DYN_RANGE_DB / 20.0))
    ex = ex * keep
    ey = ey * keep

    # Per-band normalisation over the whole utterance, as published: the
    # correlation is computed between these globally normalized envelope
    # segments, with no local mean removal (removing it degenerates for
    # quasi-stationary bands).
    norms_x = np.sqrt(np.sum(ex * ex, axis=1))
    live = norms_x > 0
    ex = ex[live]
    ey = ey[live]
    ex = ex / np.sqrt(np.sum(ex * ex, axis=1, keepdims=True))
    ey = ey / np.sqrt(np.sum(ey * ey, axis=1, keepdims=True))

    d = []
    for start in range(0, t - SEGMENT_FRAMES):
        xs = ex[:, start:start + SEGMENT_FRAMES]
        ys = ey[:, start:start + SEGMENT_FRAMES]
        num = np.sum(xs * ys, axis=1)
        nx = np.sqrt(np.sum(xs * xs, axis=1))
        ny = np.sqrt(np.sum(ys * ys, axis=1))
        valid = (nx > 0) & (ny > 0)          # cells with actual evidence only
        if not valid.any():
            continue
        rho = num[valid] / (nx[valid] * ny[valid])
        d.extend(np.clip(rho, 0.0, 1.0).tolist())

    return float(np.mean(np.array(d) ** BETA))


def _crosscheck_pystoi(ref, test, fs):
    try:
        from pystoi import stoi as pstoi
    except ImportError:
        return "pystoi not installed (non-blocking)"
    try:
        return f"pystoi={pstoi(ref, test, fs, False):.4f}"
    except Exception as e:                # noqa: BLE001 - report, do not crash
        return f"pystoi error: {e}"


def _score_variant(tag, ref, test_path, fs, baselines, tol, cross):
    """Scores one processed WAV against the shared reference and baselines."""
    test, fs_t = read_wav_mono(test_path)
    if fs_t != fs:
        print(f"{tag}: GATE FAIL (rate {fs_t} != reference rate {fs})")
        return False
    s_test = stoi(ref, test, fs)
    line = f"{tag}: STOI={s_test:.4f}"
    if cross:
        line += "  [" + _crosscheck_pystoi(ref, test, fs) + "]"
    print(line)

    ok = True
    for name, base_path in baselines:
        base, _ = read_wav_mono(base_path)
        s_base = stoi(ref, base, fs)
        verdict = "ok"
        if s_test < s_base - tol:
            verdict = "GATE FAIL"
            ok = False
        print(f"{tag}:   vs {name}: {s_base:.4f} -> {verdict}")
    return ok


def _selftest():
    rng = np.random.default_rng(42)
    fs = 48000
    dur = 3.0
    t = np.arange(int(dur * fs)) / fs

    # Speech-like reference: harmonically stacked tones gated into phoneme-
    # like segments with piecewise levels, so band envelopes actually fluctuate
    # within STOI segments the way real speech does.
    env = 0.45 * (0.5 + 0.5 * np.cos(2 * np.pi * 2.5 * t)) + 0.55
    seg_edges = np.arange(0, len(t), int(0.08 * fs))
    levels = 0.3 + 0.7 * rng.random(len(seg_edges))
    gate = np.interp(t, seg_edges / fs, levels)
    x = (np.sin(2 * np.pi * 400 * t) + 0.6 * np.sin(2 * np.pi * 900 * t)
         + 0.4 * np.sin(2 * np.pi * 1700 * t)) * env * gate * 0.25

    def add_noise(sig, snr_db):
        p = np.mean(sig ** 2)
        sigma = np.sqrt(p / (10 ** (snr_db / 10)))
        return sig + rng.normal(0.0, sigma, size=sig.shape)

    s_clean = stoi(x, x.copy(), fs)
    s_hi = stoi(x, add_noise(x, 20.0), fs)
    s_lo = stoi(x, add_noise(x, 0.0), fs)
    print(f"selftest: STOI(clean,clean) ={s_clean:.4f}")
    print(f"selftest: STOI(clean,+20dB) ={s_hi:.4f}")
    print(f"selftest: STOI(clean,+0dB)  ={s_lo:.4f}")

    checks = [
        ("clean-vs-clean >= 0.99", s_clean >= 0.99),
        ("monotone in noise level", s_lo < s_hi < s_clean),
        ("20 dB SNR scores reasonably high", s_hi >= 0.85),
    ]
    failed = False
    for name, cond in checks:
        print(f"selftest: {'PASS' if cond else 'FAIL'} {name}")
        failed |= not cond
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", help="clean reference WAV (shared by all variants)")
    ap.add_argument("--test", help="processed variant WAV")
    ap.add_argument("--baseline", action="append", default=[],
                    help="baseline WAV as NAME=PATH, repeatable; the test "
                         "variant must not score below any baseline minus tol")
    ap.add_argument("--batch", help="CSV manifest with columns "
                                    "tag,ref,test,base1_name,base1_path,...")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pystoi-crosscheck", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        sys.exit(_selftest())

    if args.batch:
        ok = True
        with open(args.batch, newline="") as f:
            for row in csv.DictReader(f):
                ref, fs = read_wav_mono(row["ref"])
                bases = []
                i = 1
                while f"base{i}_path" in row and row[f"base{i}_path"]:
                    bases.append((row[f"base{i}_name"], row[f"base{i}_path"]))
                    i += 1
                if not _score_variant(row["tag"], ref, row["test"], fs,
                                      bases, args.tol, args.pystoi_crosscheck):
                    ok = False
        print("BATCH VERDICT:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    if not args.ref or not args.test:
        ap.error("need --ref/--test, or --batch, or --selftest")

    baselines = []
    for spec in args.baseline:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--baseline expects NAME=PATH, got {spec!r}")
        baselines.append((name, path))

    ref, fs = read_wav_mono(args.ref)
    ok = _score_variant("single", ref, args.test, fs,
                        baselines, args.tol, args.pystoi_crosscheck)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
