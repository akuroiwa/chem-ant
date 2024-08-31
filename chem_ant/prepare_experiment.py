import argparse
import os
import configparser
import shlex
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from rdkit import Chem
from rdkit.Chem import BRICS
try:
    from .create_vina_config import download_structure, create_vina_config, parse_residue_selection
except ImportError:
    from create_vina_config import download_structure, create_vina_config, parse_residue_selection

def create_config_file(output_dir, target_fragments):
    """Create a configuration file for the experiment."""
    target_fragments_list = shlex.split(target_fragments)
    first_fragment = target_fragments_list[0]
    config = configparser.ConfigParser()

    config['DEFAULT'] = {
    # common_settings = {
        'file_name': 'fragments.csv',
        'target': first_fragment,
        'include': 'True',
        'molecule': target_fragments,
    }

    config['similarity-ant'] = {
        # **common_settings,
        'population': '300',
        'generation': '15',
        'loop': '1',
        'experiment': '3',
        'generate': '10',
    }

    config['similarity-mcts'] = {
        # **common_settings,
        'loop': '2',
        'experiment': '3',
        'iteration_limit': '5',
        'generate': '10',
    }

    with open(os.path.join(output_dir, 'config.ini'), 'w') as configfile:
        config.write(configfile)

def extract_fragments(structure_file, residue_selection, chain_id=None, output_fragments=True, num_smiles=10):
    """Extract fragments from the structure file based on residue selection."""
    parser = PDBParser() if structure_file.endswith('.pdb') else MMCIFParser()
    structure = parser.get_structure("protein", structure_file)
    residue_numbers = parse_residue_selection(residue_selection)
    fragments_smiles = []

    # Create a dictionary for amino acid conversion
    aa_dict = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    # Extract FASTA sequence from selected residues
    fasta_sequence = ''
    for model in structure:
        for chain in model:
            if chain_id is None or chain.id == chain_id:
                for residue in chain:
                    if residue.id[1] in residue_numbers:
                        fasta_sequence += aa_dict.get(residue.resname, '')

    # Convert FASTA to molecule
    mol = Chem.MolFromFASTA(fasta_sequence)
    allfrags = set(BRICS.BRICSDecompose(mol, returnMols=True))
    builder = BRICS.BRICSBuild(allfrags)

    # Generate fragments SMILES or SMILES of molecule objects
    if output_fragments:
        for frag in allfrags:
            fragments_smiles.append(Chem.MolToSmiles(frag))
    else:
        for _ in range(num_smiles):
            try:
                mol = next(builder)
                mol.UpdatePropertyCache(strict=True)
                fragments_smiles.append(Chem.MolToSmiles(mol))
            except StopIteration:
                break

    # Output SMILES to CSV
    fragments_csv = os.path.join(os.path.dirname(structure_file), 'fragments.csv')
    with open(fragments_csv, 'w') as f:
        f.write("Fragment\n")
        for frag_smiles in fragments_smiles:
            f.write(frag_smiles + "\n")

    return fragments_smiles

def main():
    """Main function to prepare the experiment."""
    parser = argparse.ArgumentParser(description="Prepare Mold for SMILES Casting experiment")
    parser.add_argument("pdb_id", help="PDB ID for the experiment")
    parser.add_argument("-f", "--format", choices=["pdb", "cif"], default="pdb", help="File format: pdb or mmcif (default: pdb)")
    parser.add_argument("-o", "--output", default="test-{pdb_id}", help="Output directory name")
    parser.add_argument("--output_fragments", action="store_true", help="Instead of generating molecules from fragments, output the fragments themselves. Both are in SMILES notation.")
    parser.add_argument("-n", "--num_smiles", type=int, default=10, help="Number of SMILES to generate")
    parser.add_argument("-r", "--residues", help="Residue selection string (e.g., '100-105,110,115-120')")
    parser.add_argument("-c", "--chain", help="Chain ID to select (e.g., 'A')")
    args = parser.parse_args()

    output_dir = args.output.format(pdb_id=args.pdb_id) if args.output else os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    structure_content = None
    try:
        structure_content = download_structure(args.pdb_id, args.format)
    except Exception as e:
        print(e)
        return

    structure_file = os.path.join(output_dir, f"{args.pdb_id}.{args.format}")
    with open(structure_file, 'wb') as f:
        f.write(structure_content)

    if args.residues:
        residue_selection = args.residues
    else:
        residue_selection = '100-105,110,115-120'
    config_file_path = os.path.join(output_dir, "config.txt")
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            for line in f:
                if line.startswith("# Residue selection:"):
                    residue_selection = line.split(':')[1].strip()
                elif line.startswith("# Chain ID:"):
                    chain_id = line.split(':')[1].strip()
                    if chain_id == "All":
                        chain_id = None
    else:
        create_vina_config(structure_file, os.path.join(output_dir, "config.txt"), "ligand.pdbqt", residue_selection=residue_selection, chain_id=args.chain)

    target_fragments = extract_fragments(structure_file, residue_selection, args.chain, args.output_fragments, args.num_smiles)
    create_config_file(output_dir, " ".join(target_fragments))

    print(f"Preparation completed. Structure file, config.ini, and fragments.csv have been created in {output_dir}")

if __name__ == "__main__":
    main()
