###### This script gives you the distance (A) of a substrate residue, which has a PAE <15 
###### with at least 10 residues of Cdc13-L-Cdc2 (as long as the substrate residue is maximum
###### 8A away from Cdc13-L-Cdc), to the Cdc13-L-Cdc2 residue it is predicted to be in contact with
#### It does this only for the ranked_0 prediction
##### It also required your PAE file to be accessible as a csv file 

import os
import csv
import requests
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from math import dist


######## This script requires your folder name to be Cdc13-L-Cdc2_and_substrate uniprot id
##### These first functions are to extract the gene name from uniprot based on the uniprot id
def get_uniprot_description(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    fasta_data = response.text
    description = fasta_data.split("\n")[0]
    return description

def get_gene_name(uniprot_description):
    try:
        parts = uniprot_description.split('GN=')
        if len(parts) > 1:
            gene_part = parts[1]
            gene_parts = gene_part.split(' ')
            gene_name = gene_parts[0]
            return gene_name
        else:
            return "Unknown"
    except IndexError:
        return "Unknown"

def get_protein_name(uniprot_description):
    parts = uniprot_description.split('|')
    protein_part = parts[2]
    protein_parts = protein_part.split(' ')
    protein_name = protein_parts[0]
    return protein_name

output_data = []

for folder in os.listdir('.'):
    if os.path.isdir(folder) and folder.startswith('Cdc13-L-Cdc2_and_'):
        print(f"Processing folder: {folder}")
        uniprot_id = folder.split('_')[-1]
        protein_name = get_protein_name(get_uniprot_description(uniprot_id))
        gene_name = get_gene_name(get_uniprot_description(uniprot_id))

        pae_file = os.path.join(folder, 'ranked_0_PAE.csv')
        pdb_file = os.path.join(folder, 'ranked_0.pdb')

        if not os.path.exists(pae_file):
            print(f"Skipping folder {folder} due to missing ranked_0_PAE.csv file")
            continue

        pae_df = pd.read_csv(pae_file, index_col=0)
        filtered_substrate_residues = pae_df.iloc[794:].apply(lambda col: sum(col[:794] < 15) >= 10, axis=1) ### PAE <15 with at least 10 Cdc13-L-Cdc2 residues (Cdc13-L-Cdc2 = 794 amino acids long)
        substrate_residues = [int(x) for x in filtered_substrate_residues.index[filtered_substrate_residues]]

        print(f"Filtered substrate residues: {substrate_residues}")

        parser = PDBParser()
        structure = parser.get_structure('Cdc13-L-Cdc2_and_Substrate', pdb_file)

        # Determine which chains to process - had issue that chain A was empty and chain B was Cdc13-L-Cdc2 and chain C the substrate for some predictions
        if 'C' in structure[0]:
            chain_ids = ('B', 'C')
        else:
            chain_ids = ('A', 'B')

        for residue_b in structure[0][chain_ids[0]]:
            if is_aa(residue_b):
                for residue_c in structure[0][chain_ids[1]]:
                    # Adjust the substrate residue number based on the chain being processed
                    substrate_residue_number_offset = 794 if chain_ids[1] == 'C' else 0
                    substrate_residue_number = int(residue_c.get_id()[1]) + substrate_residue_number_offset
                    
                    if is_aa(residue_c) and substrate_residue_number in substrate_residues:
                        distance = dist(residue_b['CA'].get_coord(), residue_c['CA'].get_coord())
                        if distance < 8:
                            output_data.append([protein_name, gene_name, residue_b.get_id()[1], substrate_residue_number, distance])

        print(f"Finished processing folder: {folder}")

print("Writing output to pae15interactionssubstratesCdc13-L-Cdc2.csv")
with open('pae15interactionssubstratesCdc13-L-Cdc2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Protein_Name', 'Gene_name', 'Cdc13-L-Cdc2_Residue', 'Substrate_Residue', 'Distance'])
    writer.writerows(output_data)

print("Script completed")
