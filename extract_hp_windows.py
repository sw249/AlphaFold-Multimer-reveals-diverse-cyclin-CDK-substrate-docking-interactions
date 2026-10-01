"""
For substrates predicted to contact the Cdc13 hydrophobic patch (region 3,
229-253) or the two regions C-terminal to it (region 4, 270-281; region 5,
295-310) -- as defined in the paper and identified by
pae15_slimfinder_combined.py -- extract a +/-10 residue window of the
substrate's own sequence around those contact residues, for re-running
SLiMFinder, matching the original Methods approach ("substrate residues
predicted to bind to a region of Cdc13-L-Cdc2 (+10 residues up- and
downstream)").

For each qualifying substrate, all contact residues in regions 3/4/5 are
pooled (since these three regions represent one contiguous HP-adjacent
surface), and the window spans (min contact residue - 10) to
(max contact residue + 10), clipped to the substrate's actual sequence
length. Sequence is read directly from each substrate's own ranked_0.pdb
(chain C), not a separately downloaded FASTA, so it's guaranteed to match
exactly what was actually folded and what the residue numbers refer to.

Usage:
    python3 extract_hp_windows.py pae15_slimfinder_with_regions.csv <parent_dir> hp_windows.fasta

    e.g. python3 extract_hp_windows.py pae15_slimfinder_with_regions.csv . hp_windows.fasta
"""

import sys
import os
import csv
from collections import defaultdict
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'MSE': 'M',
}

HP_AND_FLANKING_REGIONS = {3, 4, 5}  # hydrophobic patch, 270-281, 295-310
SUBSTRATE_CHAIN = "C"
WINDOW = 10


def get_substrate_sequence(pdb_path):
    """Return {residue_number: one_letter_code} for chain C of a structure."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    seq = {}
    for res in structure[0][SUBSTRATE_CHAIN]:
        if is_aa(res):
            seq[res.get_id()[1]] = THREE_TO_ONE.get(res.get_resname(), "X")
    return seq


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 extract_hp_windows.py <regions_csv> <parent_dir> <output.fasta>")
        sys.exit(1)

    regions_csv, parent_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(regions_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    # Pool all qualifying (Cdc13-chain, region 3/4/5) substrate contact residues per substrate
    contacts_by_substrate = defaultdict(set)
    for r in rows:
        if r["Kinase_Chain"] != "Cdc13":
            continue
        try:
            region = int(r["Region"])
        except ValueError:
            continue
        if region not in HP_AND_FLANKING_REGIONS:
            continue
        acc = r["Uniprot_ID"]
        contacts_by_substrate[acc].add(int(r["Substrate_Residue"]))

    print(f"{len(contacts_by_substrate)} substrates have a contact in the "
          f"hydrophobic patch / flanking regions (3, 4, 5).")

    written, skipped = 0, []
    with open(out_path, "w") as out_f:
        for acc, residues in contacts_by_substrate.items():
            job_dir = os.path.join(parent_dir, f"P10815_and_P04551_and_{acc}")
            pdb_path = os.path.join(job_dir, "ranked_0.pdb")
            if not os.path.exists(pdb_path):
                skipped.append((acc, f"missing {pdb_path}"))
                continue

            seq = get_substrate_sequence(pdb_path)
            if not seq:
                skipped.append((acc, "no chain C sequence found"))
                continue

            min_res, max_res = min(residues), max(residues)
            win_start = max(min(seq), min_res - WINDOW)
            win_end = min(max(seq), max_res + WINDOW)

            window_seq = "".join(seq.get(i, "X") for i in range(win_start, win_end + 1))

            out_f.write(f">{acc}\n{window_seq}\n")
            written += 1

    print(f"\nWrote {written} sequence windows to {out_path}.")
    if skipped:
        print(f"\n{len(skipped)} substrate(s) skipped:")
        for acc, reason in skipped:
            print(f"  {acc}: {reason}")