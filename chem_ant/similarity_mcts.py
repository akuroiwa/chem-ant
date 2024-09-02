from rdkit import Chem, DataStructs
# from rdkit.Chem.Fingerprints import FingerprintMols
# from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, BRICS
# from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import rdFingerprintGenerator
import argparse
from copy import deepcopy
# from mcts import mcts
try:
    from mcts_solver.mcts_solver import AntLionMcts
except ImportError:
    from mcts_solver import AntLionMcts

import pandas as pd
import os
# from global_chem import GlobalChem
# from global_chem_extensions.cheminformatics.cheminformatics import ChemInformatics
import sys
try:
    from global_chem import GlobalChem
    from global_chem_extensions.cheminformatics.cheminformatics import ChemInformatics
except ImportError:
    print("Error: Required dependencies not found.")
    print("Please install the required packages using the following commands:")
    print("pip install git+https://github.com/Global-Chem/global-chem.git#subdirectory=global_chem")
    print("pip install git+https://github.com/Global-Chem/global-chem.git#subdirectory=global_chem_extensions")
    sys.exit(1)

class SimilarityState():

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def __init__(self, jewel, vera,
                 loop=1, experiment=3, file_name="generated_smiles.csv", json=False, verbose=False):
        self.jewel = jewel
        self.vera = vera
        self.hercule = []
        # self.max_suspect = None
        # self.generated_mols = None
        self.smiles = None
        # self.morgan_fps = None
        # self.little_gray_cells = None
        # self.lipinski = None
        self.currentPlayer = 1
        self.loop = loop
        self.experiment = experiment
        self.file_name = file_name
        self.json = json
        self.verbose = verbose

    def getCurrentPlayer(self):
        # return 1 if len(self.hercule) % 2 == 0 else -1
        return self.currentPlayer

    def getPossibleActions(self):
        if self.verbose:
            print("getPossibleActions")
            print("Vera: {}".format(self.vera))

        try:
            set1 = set(self.vera)
            set2 = set(self.hercule)
            self.vera = list(set1 - set2)
            return self.vera
        except ValueError:
            self.isTerminal = True

    # @property
    def setPrevious(self, previous):
        if self.verbose:
            print("setPrevious")

        try:
            set1 = set(self.vera)
            set2 = set(previous)
            self.vera = list(set1 - set2)
        except ValueError:
            self.isTerminal = True

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.hercule.append(action)
        # newState.currentPlayer = self.currentPlayer * -1
        newState.currentPlayer = self.currentPlayer
        self.currentPlayer *= -1
        if self.verbose:
            print("takeAction")
        return newState

    def isTerminal(self):
        if self.verbose:
            print("isTerminal")
            print("Length of hercule: {}".format(len(self.hercule)))
            print("Hercule: {}".format(self.hercule))
            print("Smiles: {}".format(self.smiles))
            print("Vera: {}".format(self.vera))
        return True if (len(self.hercule) >= self.loop) or (not self.vera) or (self.hercule and not self.smiles) else False

    @classmethod
    def genMols(cls, jewel, hercule, loop=3, file_name="generated_smiles.csv", json=False, verbose=False):
        """Please refer to the links below for validation of smiles.
        `Improve error handling when trying to import an invalid .mol file. #642 <https://github.com/rdkit/rdkit/issues/642>`__
        `Validating SMILES with RDKit, PySMILES, MolVS, and PartialSMILES
        <https://sharifsuliman.medium.com/validating-smiles-with-rdkit-pysmiles-molvs-and-partialsmiles-5b65e800235f>`__
        """
    # @staticmethod
    # def genMols(jewel, hercule, loop=3, file_name="generated_smiles.csv"):
        jewel_mol = Chem.MolFromSmiles(jewel)
        jewel_mol.UpdatePropertyCache(strict=True)
        Chem.SanitizeMol(jewel_mol)
        # jewel_fp = AllChem.GetMorganFingerprintAsBitVect(jewel_mol, 2, 2048)
        jewel_fp = cls.morgan_gen.GetFingerprint(jewel_mol)
        # jewel_fp = cls.morgan_gen.GetFingerprintAsNumPy(jewel_mol)

        # jewel_fp = Pairs.GetAtomPairFingerprint(jewel_mol)
        # jewel_fp = FingerprintMols.FingerprintMol(jewel_mol)

        allfrags = set()

        for smiles in hercule:
            try:
                # Validate smiles.  This code was suggested by chatGPT.
                # mol = Chem.MolFromSmiles(smiles)
                # if mol is not None:
                #     frags = BRICS.BRICSDecompose(mol, returnMols=True)
                #     allfrags.update(frags)
                # else:
                #     print('Invalid SMILES')

                # Validate smiles.  This code was suggested by chatGPT.
                mol = Chem.MolFromSmiles(smiles)
                try:
                    frags = BRICS.BRICSDecompose(mol, returnMols=True)
                    allfrags.update(frags)
                except:
                    print('Error processing molecule')
                    # raise ValueError('Error processing molecule')

                # allfrags.update(BRICS.BRICSDecompose(Chem.MolFromSmiles(smiles), returnMols=True))
                if verbose:
                    print("allfrags update")
            except TypeError:
                if verbose:
                    print("TypeError")
                    print(allfrags)
                # continue

        if not allfrags:
            raise ValueError("Any fragments is not generated.")

        builder = BRICS.BRICSBuild(allfrags)
        if verbose:
            print("builder")

        generated_mols = []
        smiles = []
        morgan_fps = []
        little_gray_cells = []
        lipinski = []
        ghose = []
        veber = []
        rule_of_3 = []
        reos = []
        drug_like = []
        pass_all_filters = []

        for i in range(loop):
            try:
                mol = next(builder)
                if verbose:
                    print("Mol generated.")
                mol.UpdatePropertyCache(strict=True)
                if verbose:
                    print("SanitizeMol")

                if Chem.SanitizeMol(mol, catchErrors=True) == 0:
                    generated_mols.append(mol)
                    smiles.append(Chem.MolToSmiles(mol))
                    if verbose:
                        print("generated_mols append")
                    # morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                    morgan_fp = cls.morgan_gen.GetFingerprint(mol)
                    # morgan_fp = cls.morgan_gen.GetFingerprintAsNumPy(mol)
                    morgan_fps.append(morgan_fp)
                    little_gray_cells.append(DataStructs.DiceSimilarity(jewel_fp, morgan_fp))
                    lipinski.append(bool_lipinski(mol))
                    ghose.append(bool_ghose(mol))
                    veber.append(bool_veber(mol))
                    rule_of_3.append(bool_rule_of_3(mol))
                    reos.append(bool_reos(mol))
                    drug_like.append(bool_drug_like(mol))
                    pass_all_filters.append(bool_pass_all_filters(mol))
                else:
                    del mol
                    raise ValueError("SanitizeMol error")
            except StopIteration:
                if verbose:
                    print("StopIteration")
                break
            except ValueError as e:
                if verbose:
                    print(f"Error: {e}")
                continue

        if json:
            json_file_name = os.path.splitext(file_name)[0] + '.json'
            jewel_set = set()
            jewel_set.update(BRICS.BRICSDecompose(Chem.MolFromSmiles(jewel)))
            jewel_token = ' '.join(jewel_set)
            smiles_set = set()
            for s in smiles:
                smiles_set.update(BRICS.BRICSDecompose(Chem.MolFromSmiles(s)))
            poirot = ' '.join(smiles_set)
            max_suspect = max(little_gray_cells)
            if os.path.exists(json_file_name):
                gen_df_old = pd.read_json(json_file_name)
                # gen_df_new = pd.DataFrame([{"text_a": jewel, "text_b": poirot, "labels": max_suspect}])
                gen_df_new = pd.DataFrame([{"text_a": jewel_token, "text_b": poirot, "labels": max_suspect}])
                gen_df = pd.concat([gen_df_old, gen_df_new], axis=0)
            else:
                gen_df = pd.DataFrame([{"text_a": jewel_token, "text_b": poirot, "labels": max_suspect}])
            gen_df.sort_values("labels", inplace=True, ascending=False)
            gen_df.drop_duplicates(inplace=True)
            gen_df.reset_index(drop=True).to_json(json_file_name)

        if os.path.exists(file_name):
            gen_df_old = pd.read_csv(file_name, index_col=0)
            gen_df_new = pd.DataFrame({"smiles": smiles, "dice_similarity": little_gray_cells,
                                       "lipinski": lipinski,
                                       "ghose": ghose,
                                       "veber": veber,
                                       "rule_of_3": rule_of_3,
                                       "reos": reos,
                                       "drug_like": drug_like,
                                       "pass_all_filters": pass_all_filters
                                       })
            # gen_df = pd.concat([gen_df_old, gen_df_new], axis=0)

            # FutureWarning: The behavior of array concatenation with empty entries is deprecated. In a future version, this will no longer exclude empty items when determining the result dtype. To retain the old behavior, exclude the empty entries before the concat operation.
            if not gen_df_old.empty and not gen_df_new.empty:
                gen_df = pd.concat([gen_df_old, gen_df_new], axis=0)
            elif not gen_df_old.empty:
                gen_df = gen_df_old
            else:
                gen_df = gen_df_new
        else:
            gen_df = pd.DataFrame({"smiles": smiles, "dice_similarity": little_gray_cells,
                                   "lipinski": lipinski,
                                   "ghose": ghose,
                                   "veber": veber,
                                   "rule_of_3": rule_of_3,
                                   "reos": reos,
                                   "drug_like": drug_like,
                                   "pass_all_filters": pass_all_filters
                                   })
        gen_df.sort_values(["lipinski", "dice_similarity"], inplace=True, ascending=False)
        gen_df.drop_duplicates(inplace=True)
        gen_df.astype({
            "lipinski": "bool",
            "ghose": "bool",
            "veber": "bool",
            "rule_of_3": "bool",
            "reos": "bool",
            "drug_like": "bool",
            "pass_all_filters": "bool"
        })
        gen_df.reset_index(drop=True).to_csv(file_name)

        return (generated_mols, smiles, morgan_fps, little_gray_cells, lipinski)

    @classmethod
    def get_smiles_from_global_chem(cls, database_names):
        gc = GlobalChem()
        gc.build_global_chem_network()
        smiles = []
        for database_name in database_names:
            smarts = list(gc.get_node_smarts(database_name).values())
            for i in smarts:
                mol = Chem.MolFromSmarts(i)
                smi = Chem.MolToSmiles(mol)
                smiles.append(smi)
        return smiles

    def getReward(self):
        try:
            (generated_mols, self.smiles, morgan_fps, little_gray_cells, lipinski) = self.genMols(self.jewel, self.hercule, self.experiment, self.file_name, self.json, self.verbose)
            if self.verbose:
                print("getReward genMols")
        except:
            if self.verbose:
                print("getReward genMols except")
            return -1

        if not self.smiles:
            if self.verbose:
                print(self.smiles is None)
            return -1
        # elif (not self.generated_mols) or (not self.smiles) or (not self.morgan_fps) or (not self.little_gray_cells):
        #     return -1

        if not filter_smiles(self.smiles):
            if self.verbose:
                print("getReward filter smiles is None.")
            return -1


        try:
            max_suspect = max(little_gray_cells)
            max_suspect_index = little_gray_cells.index(max_suspect)
            if self.verbose:
                print(smiles[max_suspect_index], little_gray_cells[max_suspect_index])

        # except ValueError:
        except:
            if self.verbose:
                print("return -1")
            return -1

        if self.verbose:
            print("getReward")
        if max_suspect <= 0.5:
            return 0
        elif max_suspect > 0.5:
            return 1
        else:
            return -1


def console_script2():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--target", dest='target', default=None, type=str, help="Target smile.")
    parser.add_argument("-m", "--molecule", dest='molecule', default=None, type=str, help="List of smiles from which the fragments are made.", nargs='+')
    parser.add_argument("-b", "--generate", dest='generate', default=10, type=int, help="Numbers of molecule generation.  Default is 10.")
    parser.add_argument("-p", "--path", dest="path", default="gen_smiles", type=str, help="Directory where you want to save.  Default is gen_smiles.")
    parser.add_argument("-f", "--file", dest="file_name", default="generated_smiles.csv", type=str, help="File name.  Default is generated_smiles.csv.")
    parser.add_argument("-j", "--json", dest="json", action="store_true", help="Output json file for similarity_classification.py.")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()

    os.makedirs(args.path, exist_ok=True)
    file_name = os.path.join(args.path, args.file_name)
    jewel = args.target
    double_clue = args.molecule
    json = args.json
    (generated_mols, smiles, morgan_fps, little_gray_cells, lipinski) = SimilarityState.genMols(jewel, double_clue, args.generate, file_name, json, args.verbose)
    if args.verbose:
        print("Generated smiles: {}".format(smiles))

def filter_smiles(smiles_list):
    """
    This is based on the paper of
    `Sharif, Suliman. Understanding drug-likeness filters with RDKit and exploring the WITHDRAWN database. (2020).
    <https://sharifsuliman1.medium.com/understanding-drug-likeness-filters-with-rdkit-and-exploring-the-withdrawn-database-ebd6b8b2921e>`__
    and the code of `global-chem <https://github.com/Sulstice/global-chem>`__.
    """

    filtered_smiles = ChemInformatics.filter_smiles_by_criteria(
        smiles_list,
        lipinski_rule_of_5=True,
        ghose=False,
        veber=False,
        rule_of_3=False,
        reos=False,
        drug_like=False,
        pass_all_filters=False
        )

def bool_lipinski(mol) -> bool:
    smi = Chem.MolToSmiles(mol)
    filter_results = ChemInformatics.filter_smiles_by_criteria([smi], lipinski_rule_of_5=True)
    return True if filter_results else False

def bool_ghose(mol) -> bool:
    """Ghose filter for drug-likeness"""
    smi = Chem.MolToSmiles(mol)
    filter_results = ChemInformatics.filter_smiles_by_criteria([smi], ghose=True)
    return True if filter_results else False

def bool_veber(mol) -> bool:
    """Veber filter for drug-likeness"""
    smi = Chem.MolToSmiles(mol)
    filter_results = ChemInformatics.filter_smiles_by_criteria([smi], veber=True)
    return True if filter_results else False

def bool_rule_of_3(mol) -> bool:
    smi = Chem.MolToSmiles(mol)
    filter_results = ChemInformatics.filter_smiles_by_criteria([smi], rule_of_3=True)
    return True if filter_results else False

def bool_reos(mol) -> bool:
    smi = Chem.MolToSmiles(mol)
    filter_results = ChemInformatics.filter_smiles_by_criteria([smi], reos=True)
    return True if filter_results else False

def bool_drug_like(mol) -> bool:
    smi = Chem.MolToSmiles(mol)
    filter_results = ChemInformatics.filter_smiles_by_criteria([smi], drug_like=True)
    return True if filter_results else False

def bool_pass_all_filters(mol) -> bool:
    smi = Chem.MolToSmiles(mol)
    filter_results = ChemInformatics.filter_smiles_by_criteria([smi], pass_all_filters=True)
    return True if filter_results else False


def console_script():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--target", dest='target', default=None, type=str, help="Target smile.")
    parser.add_argument("-m", "--molecule", dest='molecule', default=None, type=str, help="List of smiles from which the fragments are made.", nargs='+')
    parser.add_argument('-c', '--GlobalChem', type=str, default=None,
                        help='Include SMILES from global-chem database. Options: global_chem, emerging_perfluoroalkyls, montmorillonite_adsorption, common_monomer_repeating_units, electrophilic_warheads_for_kinases, common_organic_solvents, open_smiles, lanthipeptides', nargs='+')
    # parser.add_argument("-p", "--previous", dest='previous', default=None, type=str, help="List of previously selected smiles.", nargs='+')
    parser.add_argument("-l", "--loop", dest='loop', default=1, type=int, help="For loop.  Default is 2.")
    parser.add_argument("-i", "--include", dest='include', action='store_true', help="Include target smiles in list of smiles.")
    parser.add_argument("-e", "--experiment", dest='experiment', default=3, type=int, help="How many times you want to generate molecules in getReward process.  Default is 3.")
    parser.add_argument("-r", "--iterationLimit", dest='iterationLimit', default=5, type=int, help="MCTS iterationLimit.  Default is 5.")
    parser.add_argument("-b", "--generate", dest='generate', default=10, type=int, help="Numbers of molecule generation.  Default is 10.")
    parser.add_argument("-s", "--select", dest='select', default=None, type=int, help="Select list of N smiles from generated_smiles.csv or -f file_name.  Must be an integer.")
    parser.add_argument("-p", "--path", dest="path", default="gen_smiles", type=str, help="Directory where you want to save.  Default is gen_smiles.")
    parser.add_argument("-f", "--file", dest="file_name", default="generated_smiles.csv", type=str, help="File name.  Default is generated_smiles.csv.")
    parser.add_argument("-j", "--json", dest="json", action="store_true", help="Output json file for similarity_classification.py.")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()

    os.makedirs(args.path, exist_ok=True)
    # file_name = os.path.join(args.path, args.file_name)
    file_name = os.path.join(args.path, os.path.splitext(args.file_name)[0] + '.csv')
    json = args.json
    verbose = args.verbose

    if args.target == None:
        # jewel = "C1CCNC(C1)C(C2=CC(=NC3=C2C=CC=C3C(F)(F)F)C(F)(F)F)O"
        # Nirmatrelvir
        jewel = "CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C"
        # jewel = "CC1([C@@H]2[C@H]1[C@H](N(C2)C(=O)[C@H](C(C)(C)C)NC(=O)C(F)(F)F)C(=O)N[C@@H](C[C@@H]3CCNC3=O)C#N)C"
        # YH-53
        # jewel = "CC(C)CC(C(=O)NC(CC1CCNC1=O)C(=O)C2=NC3=CC=CC=C3S2)NC(=O)C4=CC5=C(N4)C=CC=C5OC"
        # jewel = "COc1cccc2c1cc([nH]2)C(=O)N[C@H](C(=O)N[C@H](C(=O)c1nc2c(s1)cccc2)C[C@@H]1CCNC1=O)CC(C)C"
    else:
        jewel = args.target
    if args.select:
        # vera = pd.read_csv('generated_smiles.csv', header=0, usecols=[1], nrows=args.select).squeeze().values.tolist()
        vera = pd.read_csv(file_name,
                           header=0, usecols=[1], nrows=args.select).squeeze().values.tolist()
    elif args.molecule:
        vera = args.molecule
    else:
        # vera = pd.read_csv('smiles.csv', header=None, usecols=[2]).squeeze().values.tolist()
        try:
            if os.path.isfile(os.path.join(os.getcwd(), 'smiles.csv')):
                vera = pd.read_csv(os.path.join(os.getcwd(), 'smiles.csv'), header=None, usecols=[2]).squeeze().values.tolist()
            else:
                vera = pd.read_csv(os.path.join(os.path.dirname(__file__), 'smiles.csv'), header=None, usecols=[2]).squeeze().values.tolist()
        except OSError:
            print("A file smiles.csv not found.")
        # vera = pd.read_csv(os.path.join(os.getcwd(), 'smiles.csv'), header=0, usecols=[2]).squeeze().values.tolist()
    if args.GlobalChem:
        vera.extend(SimilarityState.get_smiles_from_global_chem(args.GlobalChem))
        print("Select list extended: {}".format(args.GlobalChem))

    initialState = SimilarityState(jewel, vera,
                                   args.loop, args.experiment, file_name, json, verbose)
    # mymcts = mcts(iterationLimit=args.iterationLimit)
    mymcts = AntLionMcts(iterationLimit=args.iterationLimit)

    double_clue = set()
    for i in range(args.loop):
        newState = deepcopy(initialState)
        newState.setPrevious(double_clue)
        action = mymcts.search(initialState=newState)
        double_clue.add(action)
    if args.include:
        double_clue.add(jewel)
    print("Material candidates: {}".format(double_clue))
    (generated_mols, smiles, morgan_fps, little_gray_cells, lipinski) = initialState.genMols(jewel, double_clue, args.generate, file_name, json, verbose)
    if verbose:
        print("Generated smiles: {}".format(smiles))

if __name__ == "__main__":
    console_script()
