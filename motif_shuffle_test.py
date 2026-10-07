"""
Motif over-representation check by shuffling (for the SLiMFinder / RxL comment).

For each pattern, counts (overlapping) matches in the interacting sequences
of Table S3, then compares with counts in randomly shuffled versions of the
same sequences (length and amino acid composition preserved).

Expected = mean count over shuffles; range = central 95% of shuffled counts;
P = fraction of shuffles with at least as many matches as the real data.

Phi (hydrophobic) = I L M F V W Y

Usage:
    python3 motif_shuffle_test.py Willich2026_TableS3.csv [n_shuffles] [seed]
"""

import csv
import random
import re
import sys

PHI = "ILMFVWY"
PATTERNS = {
    "[RK]xL(x)Phi": f"(?=[RK].L.?[{PHI}])",
    "[RK]xL (core only)": "(?=[RK].L)",
    "ERL-type [FVIPWGLAM](x)xER[LMV]": "(?=[FVIPWGLAM].{1,2}ER[LMV])",
    "NLxxL": "(?=NL..L)",
    "LxF": "(?=L.F)",
    "PxF": "(?=P.F)",
}


def count(pattern, seqs):
    return sum(len(re.findall(pattern, s)) for s in seqs)


def main(path, n_shuffles=2000, seed=1):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["interacting_sequence"].strip()]
    seqs = [r["interacting_sequence"].strip().upper() for r in rows]
    names = [r["Gene_name"] or r["Systematic_id"] for r in rows]
    print(f"{len(seqs)} sequences, {sum(map(len, seqs))} residues, "
          f"{n_shuffles} shuffles, seed {seed}\n")

    for label, pat in PATTERNS.items():
        obs = count(pat, seqs)
        n_with = sum(bool(re.search(pat, s)) for s in seqs)
        random.seed(seed)
        bg = sorted(
            count(pat, [''.join(random.sample(s, len(s))) for s in seqs])
            for _ in range(n_shuffles)
        )
        mean = sum(bg) / n_shuffles
        lo = bg[int(0.025 * n_shuffles)]
        hi = bg[int(0.975 * n_shuffles) - 1]
        p = sum(b >= obs for b in bg) / n_shuffles
        p_txt = f"<{1 / n_shuffles:.4f}" if p == 0 else f"{p:.4f}"
        print(f"{label}: observed {obs} in {n_with}/{len(seqs)} sequences; "
              f"expected {mean:.2f} (95% range {lo}-{hi}); P(>=obs) {p_txt}")
        if label == "[RK]xL(x)Phi":
            for n, s in zip(names, seqs):
                hits = [s[m.start():m.start() + 5] for m in re.finditer(pat, s)]
                if hits:
                    print(f"    {n}: {hits}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1],
         int(sys.argv[2]) if len(sys.argv) > 2 else 2000,
         int(sys.argv[3]) if len(sys.argv) > 3 else 1)
