import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import pandas as pd
import typing as T
import pulp
import os
import networkx as nx
import ast
from data.datasets import Tree
from sklearn.model_selection import GroupKFold
from torch_geometric.utils import to_networkx
from itertools import combinations
from pathlib import Path
from myopic_mces.myopic_mces import MCES
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs, Draw
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Scaffolds import MurckoScaffold
from huggingface_hub import hf_hub_download
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from standardizeUtils.standardizeUtils import (
    standardize_structure_with_pubchem,
    standardize_structure_list_with_pubchem,
)



def load_massspecgym():
    df = pd.read_csv(hugging_face_download("MassSpecGym.tsv"), sep="\t")
    df = df.set_index("identifier")
    df['mzs'] = df['mzs'].apply(parse_spec_array)
    df['intensities'] = df['intensities'].apply(parse_spec_array)
    return df


def pad_spectrum(
    spec: np.ndarray, max_n_peaks: int, pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad a spectrum to a fixed number of peaks by appending zeros to the end of the spectrum.
    
    Args:
        spec (np.ndarray): Spectrum to pad represented as numpy array of shape (n_peaks, 2).
        max_n_peaks (int): Maximum number of peaks in the padded spectrum.
        pad_value (float, optional): Value to use for padding.
    """
    n_peaks = spec.shape[0]
    if n_peaks > max_n_peaks:
        raise ValueError(
            f"Number of peaks in the spectrum ({n_peaks}) is greater than the maximum number of peaks."
        )
    else:
        return np.pad(
            spec,
            ((0, max_n_peaks - n_peaks), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )


def morgan_fp(mol: Chem.Mol, fp_size=2048, radius=2, to_np=True):
    """
    Compute Morgan fingerprint for a molecule.
    
    Args:
        mol (Chem.Mol): _description_
        fp_size (int, optional): Size of the fingerprint.
        radius (int, optional): Radius of the fingerprint.
        to_np (bool, optional): Convert the fingerprint to numpy array.
    """

    fp = Chem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    if to_np:
        fp_np = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, fp_np)
        fp = fp_np
    return fp


def tanimoto_morgan_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    return DataStructs.TanimotoSimilarity(morgan_fp(mol1, to_np=False), morgan_fp(mol2, to_np=False))


def standardize_smiles(smiles: T.Union[str, T.List[str]]) -> T.Union[str, T.List[str]]:
    """
    Standardize SMILES representation of a molecule using PubChem standardization.
    """
    if isinstance(smiles, str):
        return standardize_structure_with_pubchem(smiles, 'smiles')
    elif isinstance(smiles, list):
        return standardize_structure_list_with_pubchem(smiles, 'smiles')
    else:
        raise ValueError("Input should be a SMILES tring or a list of SMILES strings.")


def mol_to_inchi_key(mol: Chem.Mol, twod: bool = True) -> str:
    """
    Convert a molecule to InChI Key representation.
    
    Args:
        mol (Chem.Mol): RDKit molecule object.
        twod (bool, optional): Return 2D InChI Key (first 14 characers of InChI Key).
    """
    inchi_key = Chem.MolToInchiKey(mol)
    if twod:
        inchi_key = inchi_key.split("-")[0]
    return inchi_key


def smiles_to_inchi_key(mol: str, twod: bool = True) -> str:
    """
    Convert a SMILES molecule to InChI Key representation.
    
    Args:
        mol (str): SMILES string.
        twod (bool, optional): Return 2D InChI Key (first 14 characers of InChI Key).
    """
    mol = Chem.MolFromSmiles(mol)
    return mol_to_inchi_key(mol, twod)


def hugging_face_download(file_name: str) -> str:
    """
    Download a file from the Hugging Face Hub and return its location on disk.
    
    Args:
        file_name (str): Name of the file to download.
    """
    return hf_hub_download(
        repo_id="roman-bushuiev/MassSpecGym",
        filename="data/" + file_name,
        repo_type="dataset",
    )


def init_plotting(figsize=(6, 2), font_scale=1.0, style="whitegrid"):
    # Set default figure size
    plt.show()  # Does not work without this line for some reason
    sns.set_theme(rc={"figure.figsize": figsize})
    mpl.rcParams['svg.fonttype'] = 'none'
    # Set default style and font scale
    sns.set_style(style)
    sns.set_context("paper", font_scale=font_scale)
    sns.set_palette(["#009473", "#D94F70", "#5A5B9F", "#F0C05A", "#7BC4C4", "#FF6F61"])


def get_smiles_bpe_tokenizer() -> ByteLevelBPETokenizer:
    """
    Return a Byte-level BPE tokenizer trained on the SMILES strings from the
    `MassSpecGym_test_fold_MCES2_disjoint_molecules_4M.tsv` dataset.
    TODO: refactor to a well-organized class.
    """
    # Initialize the tokenizer
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
    smiles_tokenizer = ByteLevelBPETokenizer()
    smiles = pd.read_csv(hugging_face_download(
        "molecules/MassSpecGym_test_fold_MCES2_disjoint_molecules_4M.tsv"
    ), sep="\t")["smiles"]
    smiles_tokenizer.train_from_iterator(smiles, special_tokens=special_tokens)

    # Enable padding
    smiles_tokenizer.enable_padding(direction='right', pad_token="<pad>")

    # Add template processing to include start and end of sequence tokens
    smiles_tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", smiles_tokenizer.token_to_id("<s>")),
            ("</s>", smiles_tokenizer.token_to_id("</s>")),
        ],
    )
    return smiles_tokenizer


def parse_spec_array(arr: str) -> np.ndarray:
    return np.array(list(map(float, arr.split(","))))


def plot_spectrum(spec, hue=None, xlim=None, ylim=None, mirror_spec=None, highl_idx=None,
                  figsize=(6, 2), colors=None, save_pth=None):

    if colors is not None:
        assert len(colors) >= 3
    else:
        colors = ['blue', 'green', 'red']

    # Normalize input spectrum
    def norm_spec(spec):
        assert len(spec.shape) == 2
        if spec.shape[0] != 2:
            spec = spec.T
        mzs, ins = spec[0], spec[1]
        return mzs, ins / max(ins) * 100
    mzs, ins = norm_spec(spec)

    # Initialize plotting
    init_plotting(figsize=figsize)
    fig, ax = plt.subplots(1, 1)

    # Setup color palette
    if hue is not None:
        norm = matplotlib.colors.Normalize(vmin=min(hue), vmax=max(hue), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.cool)
        plt.colorbar(mapper, ax=ax)

    # Plot spectrum
    for i in range(len(mzs)):
        if hue is not None:
            color = mcolors.to_hex(mapper.to_rgba(hue[i]))
        else:
            color = colors[0]
        plt.plot([mzs[i], mzs[i]], [0, ins[i]], color=color, marker='o', markevery=(1, 2), mfc='white', zorder=2)

    # Plot mirror spectrum
    if mirror_spec is not None:
        mzs_m, ins_m = norm_spec(mirror_spec)

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            label = str(round(-x)) if x < 0 else str(round(x))
            return label

        for i in range(len(mzs_m)):
            plt.plot([mzs_m[i], mzs_m[i]], [0, -ins_m[i]], color=colors[2], marker='o', markevery=(1, 2), mfc='white',
                     zorder=1)
        ax.yaxis.set_major_formatter(major_formatter)

    # Setup axes
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    else:
        plt.xlim(0, max(mzs) + 10)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel('m/z')
    plt.ylabel('Intensity [%]')

    if save_pth is not None:
        raise NotImplementedError()


def show_mols(mols, legends='new_indices', smiles_in=False, svg=False, sort_by_legend=False, max_mols=500,
              legend_float_decimals=4, mols_per_row=6, save_pth: T.Optional[Path] = None):
    """
    Returns svg image representing a grid of skeletal structures of the given molecules. Copy-pasted
     from https://github.com/pluskal-lab/DreaMS/blob/main/dreams/utils/mols.py

    :param mols: list of rdkit molecules
    :param smiles_in: True - SMILES inputs, False - RDKit mols
    :param legends: list of labels for each molecule, length must be equal to the length of mols
    :param svg: True - return svg image, False - return png image
    :param sort_by_legend: True - sort molecules by legend values
    :param max_mols: maximum number of molecules to show
    :param legend_float_decimals: number of decimal places to show for float legends
    :param mols_per_row: number of molecules per row to show
    :param save_pth: path to save the .svg image to
    """
    if smiles_in:
        mols = [Chem.MolFromSmiles(e) for e in mols]

    if legends == 'new_indices':
        legends = list(range(len(mols)))
    elif legends == 'masses':
        legends = [ExactMolWt(m) for m in mols]
    elif callable(legends):
        legends = [legends(e) for e in mols]

    if sort_by_legend:
        idx = np.argsort(legends).tolist()
        legends = [legends[i] for i in idx]
        mols = [mols[i] for i in idx]

    legends = [f'{l:.{legend_float_decimals}f}' if isinstance(l, float) else str(l) for l in legends]

    img = Draw.MolsToGridImage(mols, maxMols=max_mols, legends=legends, molsPerRow=min(max_mols, mols_per_row),
                         useSVG=svg, returnPNG=False)

    if save_pth:
        with open(save_pth, 'w') as f:
            f.write(img.data)

    return img


class MyopicMCES():
    def __init__(
        self,
        ind: int = 0,  # dummy index
        solver: str = pulp.listSolvers(onlyAvailable=True)[0],  # Use the first available solver
        threshold: int = 15,  # MCES threshold
        always_stronger_bound: bool = True, # "False" makes computations a lot faster, but leads to overall higher MCES values
        solver_options: dict = None
    ):
        self.ind = ind
        self.solver = solver
        self.threshold = threshold
        self.always_stronger_bound = always_stronger_bound
        if solver_options is None:
            solver_options = dict(msg=0)  # make ILP solver silent
        self.solver_options = solver_options

    def __call__(self, smiles_1: str, smiles_2: str) -> float:
        retval = MCES(
            s1=smiles_1,
            s2=smiles_2,
            ind=self.ind,
            threshold=self.threshold,
            always_stronger_bound=self.always_stronger_bound,
            solver=self.solver,
            solver_options=self.solver_options
        )
        dist = retval[1]
        return dist
    

def parse_paths_from_df(df):
    all_tree_paths = []

    # first init the defaultdict with all precursor_mz values
    all_precursor_mz = []
    for ms_level, precursor_mz, smi in zip(df["ms_level"], df["precursor_mz"], df["smiles"]):
        # use ms_level to detect when a spectrum is MS2
        if int(ms_level) == 2:
            all_precursor_mz.append(precursor_mz)
            all_tree_paths.append([smi, precursor_mz, []])

    # then iterate over the df and when msn_precursor_mzs is NaN, go to the next item in all_tree_paths
    # assuming that the first ms_level == 2
    idx_cur_precursors_mz = -1
    for _, row in df.iterrows():
        if int(row["ms_level"]) == 2:
            # if the spectrum is MS2, go to the next tree
            idx_cur_precursors_mz += 1
            continue
        else:
            # else add the path at the appropriate index
            cur_path_group = all_tree_paths[idx_cur_precursors_mz][2]
            msn_precursor_mzs = ast.literal_eval(row["msn_precursor_mzs"])
            # replace the first value of msn_precursor_mzs with precursor_mz
            msn_precursor_mzs[0] = all_precursor_mz[idx_cur_precursors_mz]
            cur_path_group.append(msn_precursor_mzs)

    return all_tree_paths


def find_duplicate_smiles(all_smiles):
    occurrences = {}  # {smiles1: {path1, path2...}, smiles2: {path1, path2...}...}
    
    for dataset_path, smiles_list in all_smiles.items():
        for smi in smiles_list:
            if smi in occurrences:
                # if the specific SMILES is in occurrences, add only the path to it
                occurrences[smi].add(dataset_path)
            else:
                occurrences[smi] = {dataset_path}
    
    # Extract items that appear in more than one list
    duplicates = {smi: paths for smi, paths in occurrences.items() if len(paths) > 1}
    
    if duplicates:
        # Track pairs of paths where duplicates are found
        path_pairs = {}

        for smi, paths in duplicates.items():
            for path_pair in combinations(paths, 2):
                sorted_pair = tuple(sorted(path_pair))
                if sorted_pair in path_pairs:
                    path_pairs[sorted_pair].append(smi)
                else:
                    path_pairs[sorted_pair] = [smi]

        # Print the duplicates for each pair of files
        for path_pair, smi_list in path_pairs.items():
            print(f"""Duplicates found in {path_pair[0]} and {path_pair[1]}:
{', '.join(smi_list)}\n""")
    else:
        print("No duplicates found.")


def add_identifiers(df):
    id_counter = 0
    id_list = []  

    for _, row in df.iterrows():
        ms_level = int(row["ms_level"])
        if ms_level == 2:
            id_counter += 1
        # for MSn levels, maintain the ID based on the current precursor_mz group

        msn_id = f"MSnID{id_counter:07d}"
        id_list.append(msn_id)

    df.insert(0, 'identifier', id_list)

    return df


def visualize_MSn_tree(tree_or_pygdata):
    if isinstance(tree_or_pygdata, Tree):
        # Convert Tree class to PyG Data
        data_obj = tree_or_pygdata.to_pyg_data()
    else:
        data_obj = tree_or_pygdata
    # Convert PyG Data to NetworkX graph
    graph = to_networkx(data_obj, to_undirected=True)

    # Extract node features (assuming "x" contains node features)
    node_features = data_obj.x.numpy()

    # Draw the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)  # change to specific for trees

    # Draw nodes with annotations
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', edge_color='grey',
            linewidths=1, font_size=10, node_size=500)

    # Annotate nodes with their feature values
    for node, (x, y) in pos.items():
        # Create a string representation of node features
        feature_str = ', '.join(f'{idx}: {val:.2f}' for idx, val in enumerate(node_features[node]))
        
        # Display the feature string next to the node
        plt.text(x, y, s=feature_str, bbox=dict(facecolor='white', alpha=0.7),
                 horizontalalignment='center', verticalalignment='center')

    # Display the plot
    plt.show()        


def smiles_to_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    return scaffold_smiles


def train_val_test_split(all_smiles, scaffolds, n_splits=10):
    X = all_smiles
    y = np.random.rand(len(all_smiles))  # Dummy target variable

    groups = np.array(scaffolds)

    group_kfold = GroupKFold(n_splits=n_splits)

    train = []
    validation = []
    test = []

    # calculate the number of splits for each set based on the ratios
    n_test_splits = int(n_splits * 0.1)
    n_val_splits = int(n_splits * 0.1)
    n_train_splits = n_splits - n_test_splits - n_val_splits

    # ensure that both validation and test splits aren't empty
    if n_test_splits == 0:
        n_test_splits = 1
        n_train_splits -= 1

    if n_val_splits == 0:
        n_val_splits = 1
        n_train_splits -= 1

    # perform GroupKFold split
    for fold_num, (_, test_index) in enumerate(group_kfold.split(X, y, groups=groups)):
        if fold_num < n_test_splits:
            test.extend(test_index)
        elif fold_num < n_test_splits + n_val_splits:
            validation.extend(test_index)
        else:
            train.extend(test_index)
    
    return train, validation, test


def create_split_file(msn_dataset, train_idxs, val_idxs, test_idxs, filepath):
    if os.path.exists(filepath):
        print(f"split tsv file already exists at {filepath}")
        return

    split_df = pd.DataFrame(columns=["identifier", "fold"])
    all_indexes = [("train", train_idxs), 
                   ("val", val_idxs),
                    ("test", test_idxs)]

    rows = []

    # create the dataframe row by row
    for fold, split_idxs in all_indexes:
        for idx in split_idxs:
            msn_id = msn_dataset.identifiers[idx]
            rows.append({"identifier": msn_id, "fold": fold})

    # concatenate the rows into the dataframe
    split_df = pd.concat([split_df, pd.DataFrame(rows)], ignore_index=True)
    split_df.to_csv(filepath, sep='\t', index=False)
    print(f"split tsv file was created successfully at {filepath}")