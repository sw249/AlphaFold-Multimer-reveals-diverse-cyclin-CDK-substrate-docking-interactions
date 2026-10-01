"""
For each ERL motif instance (folded individually with AlphaPulldown against
Cdc13-Cdc2), check whether the TOP-RANKED model shows the instance predicted
to contact the Cdc13 hydrophobic patch SPECIFICALLY (residues 229-253) --
not the flanking regions, not the phospho-pocket, just the HP itself this
time.

Uses the same two-step criterion as throughout this project: an instance
residue must have PAE<15 against at least 10 residues of the combined
Cdc13+Cdc2 kinase complex (confidence), AND be within 8A of a
hydrophobic-patch residue (229-253) specifically (contact/location).

Reports per-instance YES/NO, plus an overall count/fraction of instances
that qualify.

PAE is read directly from the top-ranked model's result_*.pkl (needs jax
importable to unpickle).

Usage:
    python3 assess_erl_instance_hp_contact.py <parent_dir> <output.csv>

    Matches any folder whose name CONTAINS "P10815_and_P04551_and_",
    extracting whatever follows as the instance label (handles numeric
    prefixes and instance-suffixed names like
    "0007_P10815_and_P04551_and_G2TRK1_instance1"). Expects each folder to
    contain ranking_debug.json and matching result_*.pkl files.
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

HP_RESIDUES = set(range(229, 254))  # hydrophobic patch ONLY this time
PAE_CUTOFF = 15
DIST_CUTOFF = 8.0
MIN_LOW_PAE_RESIDUES = 10


def parse_ca_atoms(pdb_path):
    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            chain_id = line[21].strip()
            resnum = int(line[22:26])
            coords[(chain_id, resnum)] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    return coords


def build_residue_index_map(pdb_path):
    index_map = {}
    idx = 0
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            chain_id = line[21].strip()
            resnum = int(line[22:26])
            key = (chain_id, resnum)
            if key in seen:
                continue
            seen.add(key)
            index_map[key] = idx
            idx += 1
    return index_map


def check_hp_contact(job_dir, rank, model_name):
    pdb_path = os.path.join(job_dir, f"ranked_{rank}.pdb")
    pkl_path = os.path.join(job_dir, f"result_{model_name}.pkl")
    if not os.path.exists(pdb_path) or not os.path.exists(pkl_path):
        return None

    coords = parse_ca_atoms(pdb_path)
    present_chains = {ch for (ch, _) in coords}
    if not {CDC13_CHAIN, CDC2_CHAIN, SUBSTRATE_CHAIN} <= present_chains:
        return None
    index_map = build_residue_index_map(pdb_path)

    with open(pkl_path, "rb") as f:
        result = pickle.load(f)
    pae_matrix = np.asarray(result["predicted_aligned_error"])
    iptm = float(result.get("iptm", float("nan")))

    kinase_residues = [(ch, r) for (ch, r) in coords if ch in (CDC13_CHAIN, CDC2_CHAIN)]
    hp_coords = {r: coords[(CDC13_CHAIN, r)] for r in HP_RESIDUES if (CDC13_CHAIN, r) in coords}
    substrate_coords = {r: xyz for (ch, r), xyz in coords.items() if ch == SUBSTRATE_CHAIN}

    qualifies = False
    best_contact = None
    for s_res, s_xyz in substrate_coords.items():
        s_idx = index_map.get((SUBSTRATE_CHAIN, s_res))
        if s_idx is None:
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

        for h_res, h_xyz in hp_coords.items():
            dist = np.linalg.norm(s_xyz - h_xyz)
            if dist <= DIST_CUTOFF:
                qualifies = True
                if best_contact is None or dist < best_contact[2]:
                    best_contact = (s_res, h_res, dist)

    return qualifies, iptm, best_contact


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 assess_erl_instance_hp_contact.py <parent_dir> <output.csv>")
        sys.exit(1)

    parent_dir, out_path = sys.argv[1], sys.argv[2]

    instance_dirs = sorted(
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d)) and FOLDER_MARKER in d
    )
    print(f"Found {len(instance_dirs)} instance folders.")

    rows = []
    n_qualify = 0
    n_checked = 0

    for folder in instance_dirs:
        job_dir = os.path.join(parent_dir, folder)
        instance_label = folder.split(FOLDER_MARKER)[-1]
        ranking_path = os.path.join(job_dir, "ranking_debug.json")
        if not os.path.exists(ranking_path):
            print(f"  {instance_label}: SKIPPED -- no ranking_debug.json")
            continue

        with open(ranking_path) as f:
            ranking = json.load(f)
        top_model_name = ranking["order"][0]

        result = check_hp_contact(job_dir, 0, top_model_name)
        if result is None:
            print(f"  {instance_label}: SKIPPED -- missing/unreadable ranked_0 files")
            continue

        qualifies, iptm, best_contact = result
        n_checked += 1
        if qualifies:
            n_qualify += 1

        rows.append({
            "instance": instance_label,
            "hp_contact": qualifies,
            "iptm": round(iptm, 4),
            "closest_instance_residue": best_contact[0] if best_contact else "",
            "closest_hp_residue": best_contact[1] if best_contact else "",
            "closest_dist": round(best_contact[2], 2) if best_contact else "",
        })
        print(f"  {instance_label}: {'YES' if qualifies else 'no'} (ipTM={iptm:.3f})")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{n_qualify}/{n_checked} instances show top-model contact with the hydrophobic patch (229-253).")
    print(f"Wrote {len(rows)} rows to {out_path}.")
