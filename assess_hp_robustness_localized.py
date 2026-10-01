"""
For each of the likely-binder substrates (folded with 5 models x 5
predictions/model = 25 total structures each; Cdc13, Cdc2 and substrate as
three SEPARATE chains A/B/C), assess how robustly the SPECIFIC sequence
window originally identified (in the first, single-model screen) as
interacting with the Cdc13 hydrophobic patch continues to show that same
contact across the full 25-model ensemble.

This directly answers Reviewer 2 point 1.3 ("how robust is the prediction
of interaction for the SEQUENCE LOCALISED TO THE HP") -- unlike a check of
"does any part of the substrate contact the HP", which could report
"robust" even if a completely different, previously-uninvolved region
happens to touch the patch in the new models.

For each substrate:
  1. Looks up its originally-identified HP-interacting sequence window from
     Hydrophobic_patch_interactions_AlphaPulldown.txt (keyed by UniProt
     entry name, e.g. "MCM10_SCHPO"), resolved to this rerun's UniProt
     accession-keyed folders via gene_name_to_uniprot.csv.
  2. For each of the 25 models, extracts the substrate chain's full
     sequence (with residue numbers) from the structure, locates the
     original window within it by substring search, and restricts the
     PAE/distance contact check to ONLY those residues -- not the whole
     substrate.

Applies the same two-step criterion used throughout this project: PAE<15
against at least 10 residues of the combined Cdc13+Cdc2 kinase complex
(confidence), AND within 8A of a hydrophobic-patch residue on Cdc13
(229-253) specifically (contact/location), but now scoped to just the
localised window.

PAE is read directly from each model's result_*.pkl (needs jax importable
to unpickle).

Usage:
    python3 assess_hp_robustness_localized.py <parent_dir> <fasta_path> <gene_name_to_uniprot_csv> <output.csv>

    Matches any folder whose name CONTAINS "P10815_and_P04551_and_",
    extracting the UniProt accession as whatever follows that substring.
    Substrates whose window can't be located in a given model's sequence,
    or that have no entry in the FASTA file / mapping CSV, are flagged in
    the printed output and CSV rather than silently skipped.
"""

import sys
import os
import json
import pickle
import csv
import numpy as np

FOLDER_MARKER = "P10815_and_P04551_and_"
CDC13_CHAIN = "A"
CDC2_CHAIN = "B"
SUBSTRATE_CHAIN = "C"

HP_RESIDUES = set(range(229, 254))
PAE_CUTOFF = 15
DIST_CUTOFF = 8.0
MIN_LOW_PAE_RESIDUES = 10

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'MSE': 'M',
}


def load_fasta(path):
    """Return {entry_name: sequence}."""
    seqs = {}
    label = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                label = line[1:].strip()
                seqs[label] = ""
            elif label:
                seqs[label] += line.strip()
    return seqs


def load_accession_to_entry_name(csv_path):
    """Return {accession: entry_name} from gene_name_to_uniprot.csv (which maps entry_name -> accession)."""
    mapping = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("uniprot_id"):
                mapping[row["uniprot_id"]] = row["gene_name"]
    return mapping


def parse_all_atoms_and_sequence(pdb_path):
    """Return ({(chain, resnum): {atom_name: xyz}}, {(chain, resnum): one_letter_code})."""
    residues = {}
    seq = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name.startswith("H") or (atom_name and atom_name[0].isdigit() and "H" in atom_name):
                continue
            chain_id = line[21].strip()
            resnum = int(line[22:26])
            resname = line[17:20].strip()
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            residues.setdefault((chain_id, resnum), {})[atom_name] = xyz
            seq[(chain_id, resnum)] = THREE_TO_ONE.get(resname, "X")
    return residues, seq


def build_residue_index_map(residues):
    """0-indexed PAE-matrix position per (chain, resnum), in file order."""
    index_map = {}
    for idx, key in enumerate(residues.keys()):
        index_map[key] = idx
    return index_map


def locate_window(seq_dict, chain, window_seq):
    """Find window_seq as a substring of chain's full sequence (in residue-number order).
    Returns the list of residue numbers spanned, or None if not found."""
    chain_residues = sorted(r for (ch, r) in seq_dict if ch == chain)
    full_seq = "".join(seq_dict[(chain, r)] for r in chain_residues)

    start = full_seq.find(window_seq)
    if start == -1:
        return None
    end = start + len(window_seq)
    return chain_residues[start:end]


def assess_one_model(job_dir, rank, model_name, window_seq):
    pdb_path = os.path.join(job_dir, f"ranked_{rank}.pdb")
    pkl_path = os.path.join(job_dir, f"result_{model_name}.pkl")
    if not os.path.exists(pdb_path) or not os.path.exists(pkl_path):
        return None, "missing pdb/pkl"

    residues, seq = parse_all_atoms_and_sequence(pdb_path)
    present_chains = {ch for (ch, _) in residues}
    if not {CDC13_CHAIN, CDC2_CHAIN, SUBSTRATE_CHAIN} <= present_chains:
        return None, "missing expected chains"

    window_residues = locate_window(seq, SUBSTRATE_CHAIN, window_seq)
    if window_residues is None:
        return None, "window sequence not found in this model's substrate chain"

    index_map = build_residue_index_map(residues)

    with open(pkl_path, "rb") as f:
        result = pickle.load(f)
    pae_matrix = np.asarray(result["predicted_aligned_error"])
    iptm = float(result.get("iptm", float("nan")))

    kinase_residues = [(ch, r) for (ch, r) in residues if ch in (CDC13_CHAIN, CDC2_CHAIN)]
    hp_coords = {r: residues[(CDC13_CHAIN, r)] for r in HP_RESIDUES if (CDC13_CHAIN, r) in residues}

    qualifies = False
    for s_res in window_residues:
        s_atoms = residues.get((SUBSTRATE_CHAIN, s_res))
        s_idx = index_map.get((SUBSTRATE_CHAIN, s_res))
        if s_atoms is None or s_idx is None:
            continue

        low_pae_count = 0
        for (ch, c_res) in kinase_residues:
            c_idx = index_map.get((ch, c_res))
            if c_idx is None:
                continue
            pae = min(pae_matrix[s_idx][c_idx], pae_matrix[c_idx][s_idx])
            if pae < PAE_CUTOFF:
                low_pae_count += 1
        if low_pae_count < MIN_LOW_PAE_RESIDUES:
            continue

        for h_res, h_atoms in hp_coords.items():
            for s_xyz in s_atoms.values():
                for h_xyz in h_atoms.values():
                    if np.linalg.norm(s_xyz - h_xyz) <= DIST_CUTOFF:
                        qualifies = True
                        break
                if qualifies:
                    break
            if qualifies:
                break
        if qualifies:
            break

    return (qualifies, iptm), None


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 assess_hp_robustness_localized.py <parent_dir> <fasta_path> <gene_name_to_uniprot_csv> <output.csv>")
        sys.exit(1)

    parent_dir, fasta_path, mapping_csv, out_path = sys.argv[1:5]

    fasta_windows = load_fasta(fasta_path)
    accession_to_entry = load_accession_to_entry_name(mapping_csv)

    substrate_dirs = sorted(
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d)) and FOLDER_MARKER in d
    )
    print(f"Found {len(substrate_dirs)} substrate folders.")

    summary_rows = []
    detail_rows = []

    for folder in substrate_dirs:
        job_dir = os.path.join(parent_dir, folder)
        accession = folder.split(FOLDER_MARKER)[-1]

        entry_name = accession_to_entry.get(accession)
        if entry_name is None:
            print(f"  {accession}: SKIPPED -- no entry in {mapping_csv}")
            continue
        window_seq = fasta_windows.get(entry_name)
        if window_seq is None:
            print(f"  {accession} ({entry_name}): SKIPPED -- no window sequence in {fasta_path}")
            continue

        ranking_path = os.path.join(job_dir, "ranking_debug.json")
        if not os.path.exists(ranking_path):
            print(f"  {accession}: SKIPPED -- no ranking_debug.json")
            continue

        with open(ranking_path) as f:
            ranking = json.load(f)
        model_order = ranking["order"]

        n_qualify = 0
        n_window_found = 0
        iptms = []
        for rank, model_name in enumerate(model_order):
            result, error = assess_one_model(job_dir, rank, model_name, window_seq)
            if error is not None:
                detail_rows.append({
                    "substrate": accession, "entry_name": entry_name, "rank": rank,
                    "model_name": model_name, "hp_contact": "", "iptm": "", "note": error,
                })
                continue
            qualifies, iptm = result
            n_window_found += 1
            iptms.append(iptm)
            if qualifies:
                n_qualify += 1
            detail_rows.append({
                "substrate": accession, "entry_name": entry_name, "rank": rank,
                "model_name": model_name, "hp_contact": qualifies, "iptm": iptm, "note": "",
            })

        n_total = len(model_order)
        summary_rows.append({
            "substrate": accession,
            "entry_name": entry_name,
            "n_models_window_found": n_window_found,
            "n_models_hp_contact": n_qualify,
            "n_models_total": n_total,
            "fraction_hp_contact": round(n_qualify / n_total, 3) if n_total else "",
            "iptm_min": round(min(iptms), 4) if iptms else "",
            "iptm_max": round(max(iptms), 4) if iptms else "",
        })
        print(f"  {accession} ({entry_name}): window located in {n_window_found}/{n_total} models, "
              f"{n_qualify}/{n_total} show HP contact at that window")

    summary_path = out_path
    detail_path = out_path.replace(".csv", "_by_model.csv")

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()) if detail_rows else [])
        writer.writeheader()
        writer.writerows(detail_rows)

    print(f"\nWrote per-substrate summary to {summary_path}.")
    print(f"Wrote per-model detail to {detail_path}.")
