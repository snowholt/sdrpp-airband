#!/usr/bin/env python3
"""Blind ABX trial builder and scorer for the AM audio-chain listening test.

Reads the manifest written by af_sweep (tag,ref,test,base...), builds a
folder of blind trials, and scores a listener's answer sheet against a
hidden key. Seeded shuffling keeps every run reproducible while the
mapping stays hidden from the listener.

Build:  python3 make_abx.py build --manifest M.csv --out TRIALS_DIR \
            --trials 12 --conditions bp,bp_nr --seed 7
        For each trial it picks one manifest row, copies the degraded
        unprocessed export and the processed export to A.wav / B.wav in
        seeded random order, and appends the true mapping to a hidden
        key.csv inside TRIALS_DIR.

Score:  python3 make_abx.py score --answers ANSWERS.csv --key TRIALS_DIR/key.csv
        Answers CSV needs columns trial,cleaner where cleaner is A or B.
        Reports preference counts for the processed variant and a two-sided
        binomial sign-test note against chance.

No dependencies beyond the standard library.
"""

import argparse
import csv
import math
import os
import random
import shutil


def read_manifest(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("test"):
                continue
            # A candidate is scored against its strongest listed baseline.
            base = None
            i = 1
            best = None
            while f"base{i}_path" in row and row[f"base{i}_path"]:
                best = (row[f"base{i}_name"], row[f"base{i}_path"])
                i += 1
            if best is not None:
                base = best
            rows.append({
                "tag": row.get("tag", ""),
                "ref": row["ref"],
                "test": row["test"],
                "baseline_name": base[0] if base else "off",
                "baseline_path": base[1] if base else row["ref"],
            })
    return rows


def cmd_build(args):
    rows = read_manifest(args.manifest)
    wanted = set(filter(None, args.conditions.split(","))) if args.conditions else None

    pool = []
    for r in rows:
        name = os.path.basename(r["test"]).rsplit("_proc.wav", 1)[0]
        # phase2_nr_<condition>_snr<NN> -> <condition>
        cond = name.split("_snr")[0].split("phase2_nr_")[-1]
        if wanted is not None and cond not in wanted:
            continue
        pool.append(r)
    if not pool:
        raise SystemExit("no matching manifest rows")

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    key_path = os.path.join(args.out, "key.csv")
    with open(key_path, "w", newline="") as kf:
        kw = csv.writer(kf)
        kw.writerow(["trial", "cleaner_is", "tag", "processed_file", "baseline_file"])
        for i in range(1, args.trials + 1):
            row = pool[(i - 1) % len(pool)]
            processed_first = rng.random() < 0.5
            a_src, b_src = (
                (row["test"], row["baseline_path"])
                if processed_first else (row["baseline_path"], row["test"])
            )
            shutil.copyfile(a_src, os.path.join(args.out, f"trial{i:02d}_A.wav"))
            shutil.copyfile(b_src, os.path.join(args.out, f"trial{i:02d}_B.wav"))
            ref_dst = os.path.join(args.out, f"trial{i:02d}_REF.wav")
            if not os.path.exists(ref_dst):
                shutil.copyfile(row["ref"], ref_dst)
            cleaner = "A" if processed_first else "B"
            kw.writerow([f"trial{i:02d}", cleaner, row["tag"],
                         os.path.abspath(row["test"]), os.path.abspath(row["baseline_path"])])
    print(f"wrote {args.trials} trials to {args.out} (key: {key_path} - keep hidden)")
    print("listener sheet columns: trial,cleaner   (values A or B)")


def cmd_score(args):
    key = {}
    with open(args.key, newline="") as f:
        for row in csv.DictReader(f):
            key[row["trial"]] = row["cleaner_is"]
    total = correct = 0
    per_tag = {}
    with open(args.answers, newline="") as f:
        for row in csv.DictReader(f):
            t = row["trial"]
            if t not in key:
                print(f"warning: {t} not in key, skipped")
                continue
            total += 1
            hit = (row["cleaner"] == key[t])
            correct += hit
            tag = key.get(t, "")
            per_tag.setdefault(t, hit)
    if total == 0:
        raise SystemExit("no answers matched the key")
    print(f"picked processed as cleaner: {correct}/{total}")
    print(f"two-sided binomial p (chance 0.5): "
          f"{sum(math.comb(total, k) for k in range(correct, total + 1)) * 2 / 2 ** total:.4f}")
    verdict = "PREFERENCE SUPPORTED" if correct > total / 2 else "NO PREFERENCE FOR PROCESSED"
    print(verdict)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--manifest", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--trials", type=int, default=12)
    b.add_argument("--conditions", default="bp_nr",
                   help="comma-separated condition prefixes from tags (default: bp_nr)")
    b.add_argument("--seed", type=int, default=7)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score")
    s.add_argument("--answers", required=True)
    s.add_argument("--key", required=True)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
