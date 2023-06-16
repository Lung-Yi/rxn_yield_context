from collections import OrderedDict
from functools import partial
from random import Random
from typing import Callable, Dict, Iterator, List, Optional, Union
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from rdkit import Chem
from rdkit.Chem import AllChem,DataStructs
import pickle

### New reaction fingerprint: https://github.com/reymond-group/drfp
# from drfp import DrfpEncoder

# def create_drfp_fingerprint(rsmi, psmi, fpsize = 16384, radius = 2):
#     rxn_smiles = rsmi + '>>' + psmi
#     fps = DrfpEncoder.encode(rxn_smiles, n_folded_length = fpsize, radius = radius)
#     return fps[0]
    

def create_rxn_Morgan2FP_concatenate(rsmi, psmi, fpsize=16384, radius=2, useFeatures=False, calculate_rfp=True, useChirality=True):
    # Similar as the above function but takes smiles separately and returns pfp and rfp separately

    rsmi = rsmi.encode('utf-8')
    psmi = psmi.encode('utf-8')
    try:
        mol = Chem.MolFromSmiles(rsmi)
    except Exception as e:
        print(e)
        return
    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(
            mol=mol, radius=radius, nBits=fpsize, useFeatures=useFeatures, useChirality=useChirality)
        fp = np.empty(fpsize, dtype='float32')
        DataStructs.ConvertToNumpyArray(fp_bit, fp)
    except Exception as e:
        print("Cannot build reactant fp due to {}".format(e))
        return
    rfp = fp

    try:
        mol = Chem.MolFromSmiles(psmi)
    except Exception as e:
        return
    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(
            mol=mol, radius=radius, nBits=fpsize, useFeatures=useFeatures, useChirality=useChirality)
        fp = np.empty(fpsize, dtype='float32')
        DataStructs.ConvertToNumpyArray(fp_bit, fp)
    except Exception as e:
        print("Cannot build product fp due to {}".format(e))
        return
    pfp = fp
    rxn_fp = pfp - rfp
    final_fp = np.concatenate((pfp, rxn_fp))
    return final_fp

def get_rxn_data(input_path):
    f = open(input_path, 'r')
    data = f.readlines()
    f.close()
    for i in range(len(data)):
        data[i] = data[i].rstrip('\n').split('\t')
    data = list(zip(*data))
    rsmiles, psmiles, yield_, reagent, solvent = data
    return list(rsmiles), list(psmiles), list(yield_), list(reagent), list(solvent)

def get_classes(path):
    f = open(path, 'rb')
    dict_ = pickle.load(f)
    f.close()
    classes = sorted(dict_.items(), key=lambda d: d[1],reverse=True)
    classes = [(x,y) for x,y in classes]
    return classes

def create_target_with_classes(targets:str, classes:list):
    targets = targets.split('; ')
    vector = [x in targets for x in classes]
    return np.array(vector, dtype=float)

def mol_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    fp_bit = AllChem.GetMorganFingerprintAsBitVect(
        mol=mol, radius=2, nBits=4096, useFeatures=False, useChirality=True)
    fp = np.empty(4096, dtype='float32')
    DataStructs.ConvertToNumpyArray(fp_bit, fp)
    return fp

"""
2020/10/12 Append new classes: ReactionDataset, ReactionDataloader
"""
class ReactionDatapoint:
    """A :class:`ReactionDatapoint` contains reactant molecule,
    product moelcules and their associated features and targets."""

    def __init__(self,
                 r_smiles: str,
                 p_smiles: str,
                 fpsize: int = 16384,
                 radius: int = 2,
                 fp_type: str = 'morgan',
                 targets: List[Optional[float]] = None):
        """
        :param r_smiles: The SMILES string for the reactant molecule.
        :param p_smiles: The SMILES string fot the product molecule.
        :param targets: A list of targets for the molecule (contains None for unknown target values).
        """

        self.r_smiles = r_smiles
        self.p_smiles = p_smiles
        self.targets = targets
        
        self.fpsize = fpsize
        self.radius = radius
        
        self.solvent = None
        self.reagent = None
        if fp_type == 'morgan':
            self.produce_morgan()
        elif fp_type == 'drfp':
            self.produce_drfp()
        elif fp_type == 'rxn_bert':
            pass
        else:
            raise ValueError('Cannot recongnize the type of reaction fingerprint')
        
        
    def set_targets(self, targets) -> None:
        """
        Sets the targets of a molecule.

        :param targets: an 1-d np.array or a single value
        """
        self._targets = targets
    
    def set_solvent(self, solvent: np.array) -> None:
        """Set the input condition of solvent for second part model. 
           Or output targets of first part Multi-task model.
        """
        self.solvent = solvent
        
    def set_reagent(self, reagent: np.array) -> None:
        """Set the input condition of reagent for second part model. 
           Or output targets of first part Multi-task model.
        """
        self.reagent = reagent
    
    def produce_morgan(self):
        self.rxn_fp = create_rxn_Morgan2FP_concatenate(self.r_smiles, self.p_smiles, fpsize=self.fpsize, radius=self.radius)
    
#     def produce_drfp(self):
#         self.rxn_fp = create_drfp_fingerprint(self.r_smiles, self.p_smiles, fpsize = self.fpsize, radius = self.radius)

    
        
class ReactionDataset(Dataset):
    r"""A :class:`MoleculeDataset` contains a list of :class:`MoleculeDatapoint`\ s with access to their attributes."""

    def __init__(self, data: List[ReactionDatapoint]):#TODO: change
        r"""
        :param data: A list of :class:`MoleculeDatapoint`\ s.
        """
        self._data = data
        self.length = len(data)
        self._scaler = None
        self._batch_graph = None
        self._random = Random()
        self.rxn_fps = None


    def smiles(self):#TODO: change
        """
        Returns a list containing the SMILES associated with each molecule.

        :return: A list of SMILES strings.
        """
        
        return [(rxn.r_smiles, rxn.p_smiles) for rxn in self._data]

    def rxn_smiles(self):
        """
        :return: A list of reaction SMILES strings, which is used for rxn_bert.
        """

        return [rxn.r_smiles+'>>'+rxn.p_smiles for rxn in self._data]

    def morgan_fingerprint(self) -> List[np.array]:
        """
        set the morgan fingerprint for the model input

        """
        self.rxn_fps = [rxn.rxn_fp for rxn in self._data]
        return [rxn.rxn_fp for rxn in self._data]
    
    def solvents(self) -> List[np.array]:
        """Return all the solvent condition in ReactionDataset. """
        return [rxn.solvent for rxn in self._data]
    
    def reagents(self) -> List[np.array]:
        """Return all the reagent condition in ReactionDataset. """
        return [rxn.reagent for rxn in self._data]

    def targets(self) -> List[List[Optional[float]]]: #TODO: change
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        
        # Because self.r_dataset.targets() == self.p_dataset.targets()
        # we only need to return one list of targets
        return [rxn._targets for rxn in self._data]
    

    def shuffle(self, seed: int = None) -> None: #TODO: change
        """
        Shuffles the dataset.
        不確定是否有修改正確，training的時候先不要使用這個shuffle
        :param seed: Optional random seed.
        """
        if seed is not None:
            self._random.seed(seed)
        self._random.shuffle(self._data)
    
    def set_targets(self, targets: List[List[Optional[float]]]) -> None: #TODO: change
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        Targets of reactant and product are same.
        """
        assert self.length == len(targets), "data長度和target長度不符合"
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])


    def sort(self, key: Callable) -> None:#TODO: change
        """
        Sorts the dataset using the provided key.

        :param key: A function on a :class:`MoleculeDatapoint` to determine the sorting order.
        """
        self._data.sort(key=key)

    def __len__(self) -> int: #TODO: change
        """
        Returns the length of the dataset (i.e., the number of molecules).

        :return: The length of the dataset.
        """
        return self.length

    def __getitem__(self, item) -> List[ReactionDatapoint]:#TODO: change
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[item]

def construct_reaction_batch(data) -> ReactionDataset: #TODO: change
    r"""
    
    data: List[ReactionDatapoint]
    
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\ s.

    Additionally, precomputes the :class:`~chemprop.features.BatchMolGraph` for the constructed
    :class:`MoleculeDataset`.

    :param data: A list of :class:`ReactionDatapoint`\ s.

    :return: A :class:`MoleculeDataset` containing all the :class:`MoleculeDatapoint`\ s.
    """
    rxn_data = ReactionDataset(data)

    return rxn_data

class ReactionSampler(Sampler): #TODO: change
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`.
       Do not modify the class_balance attribute, so do not use this.
    """

    def __init__(self,
                 dataset: ReactionDataset,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Class balance is only available for single task
                              classification datasets. Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.shuffle = shuffle
        self._random = Random(seed)
        self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self._random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length


class ReactionDataLoader(DataLoader): #TODO: change
    """A :class:`ReactionDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`ReactionDataset`."""

    def __init__(self,
                 dataset: ReactionDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 shuffle: bool = False,
                 seed: int = 0
                 ):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param cache: Whether to store the individual :class:`~chemprop.features.MolGraph` featurizations
                      for each molecule in a global cache.
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Class balance is only available for single task
                              classification datasets. Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers

        self._shuffle = shuffle
        self._seed = seed

        self._sampler = ReactionSampler(
            dataset=self._dataset,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(ReactionDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=partial(construct_reaction_batch)
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """

        return [self._dataset[index][0].targets for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[ReactionDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(ReactionDataLoader, self).__iter__()
    
if __name__ == '__main__':
    rsmi = 'CCC'
    psmi = 'CCCBr'
    R1 = ReactionDatapoint(rsmi,psmi)
    R2 = ReactionDatapoint(psmi,rsmi)
    RDS = ReactionDataset([R1,R2])
    RDS.set_targets([1,2])