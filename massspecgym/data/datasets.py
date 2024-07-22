import pandas as pd
import json
import typing as T
import numpy as np
import torch
import matchms
import massspecgym.utils as utils
from pathlib import Path
from typing import Optional
from rdkit import Chem
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data
from matchms.importing import load_from_mgf
from massspecgym.data.transforms import SpecTransform, MolTransform, MolToInChIKey
from massspecgym.utils import add_identifiers, parse_paths_from_df


class MassSpecDataset(Dataset):
    """
    Dataset containing mass spectra and their corresponding molecular structures. This class is
    responsible for loading the data from disk and applying transformation steps to the spectra and
    molecules.
    """

    def __init__(
        self,
        spec_transform: Optional[SpecTransform] = None,
        mol_transform: Optional[MolTransform] = None,
        pth: Optional[Path] = None,
        return_mol_freq: bool = True,
        return_identifier: bool = True,
        dtype: T.Type = torch.float32
    ):
        """
        Args:
            mgf_pth (Optional[Path], optional): Path to the .tsv or .mgf file containing the mass spectra.
                Default is None, in which case the MassSpecGym dataset is downloaded from HuggingFace Hub.
        """
        self.pth = pth
        self.spec_transform = spec_transform
        self.mol_transform = mol_transform
        self.return_mol_freq = return_mol_freq

        if self.pth is None:
            self.pth = utils.hugging_face_download("MassSpecGym.tsv")

        if isinstance(self.pth, str):
            self.pth = Path(self.pth)

        if self.pth.suffix == ".tsv":
            self.metadata = pd.read_csv(self.pth, sep="\t")
            self.spectra = self.metadata.apply(
                lambda row: matchms.Spectrum(
                    mz=np.array([float(m) for m in row["mzs"].split(",")]),
                    intensities=np.array(
                        [float(i) for i in row["intensities"].split(",")]
                    ),
                    metadata={"precursor_mz": row["precursor_mz"]},
                ),
                axis=1,
            )
            self.metadata = self.metadata.drop(columns=["mzs", "intensities"])
        elif self.pth.suffix == ".mgf":
            self.spectra = list(load_from_mgf(str(self.pth)))
            self.metadata = pd.DataFrame([s.metadata for s in self.spectra])
        else:
            raise ValueError(f"{self.pth.suffix} file format not supported.")
        
        if self.return_mol_freq:
            if "inchikey" not in self.metadata.columns:
                self.metadata["inchikey"] = self.metadata["smiles"].apply(utils.smiles_to_inchi_key)
            self.metadata["mol_freq"] = self.metadata.groupby("inchikey")["inchikey"].transform("count")

        self.return_identifier = return_identifier
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(
        self, i: int, transform_spec: bool = True, transform_mol: bool = True
    ) -> dict:
        spec = self.spectra[i]
        spec = (
            self.spec_transform(spec)
            if transform_spec and self.spec_transform
            else spec
        )
        spec = torch.as_tensor(spec, dtype=self.dtype)

        metadata = self.metadata.iloc[i]
        mol = metadata["smiles"]
        mol = self.mol_transform(mol) if transform_mol and self.mol_transform else mol
        if isinstance(mol, np.ndarray):
            mol = torch.as_tensor(mol, dtype=self.dtype)

        item = {"spec": spec, "mol": mol}

        # TODO: Add other metadata to the item. Should it be just done in subclasses?
        item.update({
            k: metadata[k] for k in ["precursor_mz", "adduct"]
        })

        if self.return_mol_freq:
            item["mol_freq"] = metadata["mol_freq"]

        if self.return_identifier:
            item["identifier"] = metadata["identifier"]

        return item

    @staticmethod
    def collate_fn(batch: T.Iterable[dict]) -> dict:
        """
        Custom collate function to handle the outputs of __getitem__.
        """
        return default_collate(batch)


class RetrievalDataset(MassSpecDataset):
    """
    Dataset containing mass spectra and their corresponding molecular structures, with additional
    candidates of molecules for retrieval based on spectral similarity.
    """

    def __init__(
        self,
        mol_label_transform: MolTransform = MolToInChIKey(),
        candidates_pth: T.Optional[T.Union[Path, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.candidates_pth = candidates_pth
        self.mol_label_transform = mol_label_transform

        # Download candidates from HuggigFace Hub
        if self.candidates_pth is None:
            self.candidates_pth = utils.hugging_face_download(
                "molecules/MassSpecGym_retrieval_candidates_mass.json"
            )
        elif isinstance(self.candidates_pth, str):
            self.candidates_pth = utils.hugging_face_download(candidates_pth)

        # Read candidates_pth from json to dict: SMILES -> respective candidate SMILES
        with open(self.candidates_pth, "r") as file:
            self.candidates = json.load(file)

    def __getitem__(self, i) -> dict:
        item = super().__getitem__(i, transform_mol=False)

        # Save the original SMILES representation of the query molecule (for evaluation)
        item["smiles"] = item["mol"]

        # Get candidates
        if item["mol"] not in self.candidates:
            raise ValueError(f'No candidates for the query molecule {item["mol"]}.')
        item["candidates"] = self.candidates[item["mol"]]

        # Save the original SMILES representations of the canidates (for evaluation)
        item["candidates_smiles"] = item["candidates"]

        # Create neg/pos label mask by matching the query molecule with the candidates
        item_label = self.mol_label_transform(item["mol"])
        item["labels"] = [
            self.mol_label_transform(c) == item_label for c in item["candidates"]
        ]

        if not any(item["labels"]):
            raise ValueError(
                f'Query molecule {item["mol"]} not found in the candidates list.'
            )

        # Transform the query and candidate molecules
        item["mol"] = self.mol_transform(item["mol"])
        item["candidates"] = [self.mol_transform(c) for c in item["candidates"]]
        if isinstance(item["mol"], np.ndarray):
            item["mol"] = torch.as_tensor(item["mol"], dtype=self.dtype)
            # item["candidates"] = [torch.as_tensor(c, dtype=self.dtype) for c in item["candidates"]]

        return item

    @staticmethod
    def collate_fn(batch: T.Iterable[dict]) -> dict:
        # Standard collate for everything except candidates and their labels (which may have different length per sample)
        collated_batch = {}
        for k in batch[0].keys():
            if k not in ["candidates", "labels", "candidates_smiles"]:
                collated_batch[k] = default_collate([item[k] for item in batch])

        # Collate candidates and labels by concatenating and storing sizes of each list
        collated_batch["candidates"] = torch.as_tensor(
            np.concatenate([item["candidates"] for item in batch])
        )
        collated_batch["labels"] = torch.as_tensor(
            sum([item["labels"] for item in batch], start=[])
        )
        collated_batch["batch_ptr"] = torch.as_tensor(
            [len(item["candidates"]) for item in batch]
        )
        collated_batch["candidates_smiles"] = \
            sum([item["candidates_smiles"] for item in batch], start=[])

        return collated_batch


# TODO: Datasets for unlabeled data.


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = {}

    def add_child(self, child_value):
        if child_value not in self.children:
            self.children[child_value] = TreeNode(child_value)
        return self.children[child_value]

    def get_depth(self):
        if not self.children:
            return 0
        return 1 + max(child.get_depth() for child in self.children.values())

    def get_branching_factor(self):
        if not self.children:
            return 0
        return max(
            len(self.children),
            max(child.get_branching_factor() for child in self.children.values())
            )

    def get_edges(self):
        edges = []
        for child in self.children.values():
            edges.append((self.value, child.value))
            edges.extend(child.get_edges())
        return edges


class Tree:
    def __init__(self, root):
        self.root = TreeNode(root)
        self.paths = []
        
    def add_path(self, path):
        self.paths.append(path)
        current_node = self.root
        for node in path:
            current_node = current_node.add_child(node)

    def get_depth(self):
        return self.root.get_depth()

    def get_branching_factor(self):
        return self.root.get_branching_factor()

    def get_edges(self):
        # Exclude edges starting and ending at the root node
        edges = self.root.get_edges()
        edges = [(u, v) for u, v in edges if u != self.root.value or v != self.root.value]
        return edges
    
    def to_pyg_data(self):
        edges = self.get_edges()

        # Extract unique node indices
        nodes_set = set(sum(edges, ()))
        node_indices = {node: idx for idx, node in enumerate(nodes_set)}
        
        # Prepare edge_index tensor
        edge_index = torch.tensor([[node_indices[edge[0]], node_indices[edge[1]]] for edge in edges],
                                  dtype=torch.long).t().contiguous()

        # Prepare node features tensor
        node_list = list(nodes_set)
        x = torch.tensor(node_list, dtype=torch.float).view(-1, 1)

        # Create Data object
        data = Data(x=x, edge_index=edge_index)

        return data


class MSnDataset(MassSpecDataset):
    def __init__(self, pth=None, dtype=torch.float32, mol_transform=None):
        # load dataset using the parent class
        super().__init__(pth=pth)
        self.metadata = self.metadata[self.metadata["spectype"] == "ALL_ENERGIES"]

        # TODO: add identifiers (and split?) to the mgf file
        # add identifiers to the metadata
        self.metadata = add_identifiers(self.metadata)
        self.identifiers = np.unique(self.metadata["identifier"])

        # get paths from the metadata
        dataset_all_tree_paths = parse_paths_from_df(self.metadata)

        # generate trees from paths and ther corresponding MSnIDs
        self.trees = []
        self.pyg_trees = []
        self.smiles = []

        for smi, root, paths in dataset_all_tree_paths:
            tree = Tree(root)
# UNCOMMENT THIS AFTER TESTING WITH ONLY THE ROOT----------------------------------------------------------------------------------------
            """for path in paths:
                tree.add_path(path)"""
# UNCOMMENT THIS AFTER TESTING WITH ONLY THE ROOT----------------------------------------------------------------------------------------
            pyg_tree = tree.to_pyg_data()
            self.trees.append(tree)
            self.pyg_trees.append(pyg_tree)
            self.smiles.append(smi)

    def __len__(self):
        return len(self.pyg_trees)

    def __getitem__(self, idx, transform_mol=True):
        spec_tree = self.pyg_trees[idx]
        smi = self.smiles[idx]

        mol = self.mol_transform(smi) if transform_mol and self.mol_transform else smi
        if isinstance(mol, np.ndarray):
            mol = torch.as_tensor(mol, dtype=self.dtype)
        
        item  = {"spec_tree": spec_tree, "mol": mol}
        return item
    
    @staticmethod
    def collate_fn(batch: T.Iterable[dict]) -> dict:
        """
        Custom collate function to handle the outputs of __getitem__.
        """
        spec_trees = [item['spec_tree'] for item in batch]
        mols = [item['mol'] for item in batch]

        # Collate the spec_trees and mols using the default collate function
        batch_spec_trees = default_collate(spec_trees)
        batch_mols = default_collate(mols)

        return {'spec_tree': batch_spec_trees, 'mol': batch_mols}