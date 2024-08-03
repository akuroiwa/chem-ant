import argparse
import os
import multiprocessing
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser

def parse_residue_selection(selection_string):
    residues = []
    for part in selection_string.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            residues.extend(range(start, end + 1))
        else:
            residues.append(int(part))
    return residues

def calculate_center_and_size_from_residues(structure, residue_numbers):
    x_coords, y_coords, z_coords = [], [], []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[1] in residue_numbers:
                    for atom in residue:
                        x_coords.append(atom.coord[0])
                        y_coords.append(atom.coord[1])
                        z_coords.append(atom.coord[2])

    if not x_coords:
        raise ValueError("No atoms found for the specified residue numbers.")

    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    center_z = sum(z_coords) / len(z_coords)

    min_coords = [min(x_coords), min(y_coords), min(z_coords)]
    max_coords = [max(x_coords), max(y_coords), max(z_coords)]

    size_x = max_coords[0] - min_coords[0] + 5
    size_y = max_coords[1] - min_coords[1] + 5
    size_z = max_coords[2] - min_coords[2] + 5

    return center_x, center_y, center_z, size_x, size_y, size_z

def get_center_and_size(structure):
    atoms = list(structure.get_atoms())
    coords = [atom.coord for atom in atoms]
    min_coords = [min(coord[i] for coord in coords) for i in range(3)]
    max_coords = [max(coord[i] for coord in coords) for i in range(3)]

    center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
    size = [max_coords[i] - min_coords[i] + 5 for i in range(3)]  # Add 5 Angstroms to each dimension

    return center, size

def create_vina_config(input_file, output_file, ligand_file, residue_selection=None):
    _, ext = os.path.splitext(input_file)

    if ext.lower() == '.pdb':
        parser = PDBParser()
        structure = parser.get_structure("protein", input_file)
    elif ext.lower() in ['.cif', '.mmcif']:
        parser = MMCIFParser()
        structure = parser.get_structure("protein", input_file)
    else:
        raise ValueError("Unsupported file format. Please use .pdb or .cif/.mmcif")

    if residue_selection:
        residue_numbers = parse_residue_selection(residue_selection)
        center_x, center_y, center_z, size_x, size_y, size_z = calculate_center_and_size_from_residues(structure, residue_numbers)
    else:
        center, size = get_center_and_size(structure)
        center_x, center_y, center_z = center
        size_x, size_y, size_z = size

    cpu_count = multiprocessing.cpu_count()

    config = f"""receptor = {input_file}
ligand = {ligand_file}

center_x = {center_x:.3f}
center_y = {center_y:.3f}
center_z = {center_z:.3f}

size_x = {size_x:.3f}
size_y = {size_y:.3f}
size_z = {size_z:.3f}

out = out.pdbqt
log = log.txt

exhaustiveness = 8
num_modes = 9
energy_range = 3

cpu = {cpu_count}
"""

    with open(output_file, 'w') as f:
        f.write(config)

    print(f"Vina configuration file '{output_file}' has been created.")

def main():
    parser = argparse.ArgumentParser(description="Create a Vina configuration file from a PDB or mmCIF structure file.")
    parser.add_argument("input_file", help="Input PDB or mmCIF file")
    parser.add_argument("-o", "--output", default="config.txt", help="Output configuration file name (default: config.txt)")
    parser.add_argument("-l", "--ligand", default="ligand.pdbqt", help="Ligand file name (default: ligand.pdbqt)")
    parser.add_argument("-r", "--residues", help="Residue selection string (e.g., '100-105,110,115-120')")

    args = parser.parse_args()

    create_vina_config(args.input_file, args.output, args.ligand, args.residues)

if __name__ == "__main__":
    main()
