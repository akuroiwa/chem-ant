import argparse
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel

def smiles_to_pdbqt(smiles, output_file):
    """Convert SMILES to PDBQT format using RDKit and OpenBabel."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    pdb_file = output_file.replace(".pdbqt", ".pdb")
    Chem.MolToPDBFile(mol, pdb_file)

    # Convert PDB to PDBQT using OpenBabel Python API
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdbqt")
    obMol = openbabel.OBMol()
    obConversion.ReadFile(obMol, pdb_file)
    obConversion.WriteFile(obMol, output_file)

def select_top_ligands(csv_file, output_dir, top_n):
    """Select top N ligands from a CSV file and convert them to PDBQT format."""
    df = pd.read_csv(csv_file)
    top_ligands = df.head(top_n)

    # Check if 'Fragment' or 'smiles' column exists
    if 'Fragment' in df.columns:
        smiles_column = 'Fragment'
    elif 'smiles' in df.columns:
        smiles_column = 'smiles'
    else:
        raise ValueError("CSV file must contain either 'Fragment' or 'smiles' column")

    for i, row in top_ligands.iterrows():
        ligand_smiles = row[smiles_column]
        ligand_file = os.path.join(output_dir, f"ligand_{i + 1}.pdbqt")
        smiles_to_pdbqt(ligand_smiles, ligand_file)

def main():
    """Main function to parse arguments and select top ligands."""
    parser = argparse.ArgumentParser(description="Select top ligands from CSV and create ligand files")
    parser.add_argument("csv_file", help="Input CSV file with generated SMILES")
    parser.add_argument("-o", "--output", default="ligands", help="Output directory for ligand files")
    parser.add_argument("-n", "--top_n", type=int, default=10, help="Number of top ligands to select")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    select_top_ligands(args.csv_file, args.output, args.top_n)

if __name__ == "__main__":
    main()
