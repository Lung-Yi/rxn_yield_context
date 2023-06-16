# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:11:16 2021
This file aims to provide listwise model with specified inputs.

change to label processed file

@author: Lung-Yi
"""
from collections import OrderedDict
from functools import partial
import random
from random import Random
from typing import Callable, Dict, Iterator, List, Optional, Union
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

import pickle
import os
from rxn_yield_context.train_multilabel.args_train import TrainArgs_rxn
from rxn_yield_context.train_multilabel.data_utils import create_rxn_Morgan2FP_concatenate, get_classes
# from rxn_yield_context.evaluate_model.eval_utils import MultiTask_Evaluator

DEFAULT_SLATE_SIZE = 70
# data_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# data_path = os.path.join(os.path.join(os.path.join(data_path, 'All_LCC_Data'), 'processed_data_temp_large_2'), 'label_processed')

# solvent_classes = get_classes(os.path.join(data_path, 'class_names_solvent_labels_processed.pkl'))
# reagent_classes = get_classes(os.path.join(data_path, 'class_names_reagent_labels_processed.pkl'))

# len_solv = len(solvent_classes)
# len_reag = len(reagent_classes)

def one_hot(solvent: str, reagent: str, solvent_classes, reagent_classes):
    """Convert the gold answer solvent ans reagent names to the one-hot feature vector. 
    :param solvent: solvent string.
    :param reagent: reagent string.
    """
    # global solvent_classes
    # global reagent_classes
    solvent = solvent.split('; ')
    vec_solv = np.array([float(x[0] in solvent) for x in solvent_classes])
    reagent = reagent.split('; ')
    vec_reag = np.array([float(x[0] in reagent) for x in reagent_classes])
    
    return vec_solv, vec_reag

def Yield2Relevance(yields):
    """Context datapoint only saves yield, not relevance. -1 means padding, so do not use -1. """
    if (type(yields) == str) or (type(yields) == float):
        num = float(yields)
        # if num >= 0.70: return 4
        # elif num >= 0.3: return 3
        # elif num > 0: return 2
        if num > 0:
            return num * 2 + 2 # sort of scaling factor
        else: return 0
    
    relevance = []
    for num in yields:
        num = float(num)/100
        relevance.append(Yield2Relevance(num))
        # if num >= 0.70: relevance.append(4)
        # elif num >= 0.3: relevance.append(3)
        # elif num > 1: relevance.append(2)
        # elif num == 1: relevance.append(0.5) # TODO: 改回來
        # else: relevance.append(0)
    return torch.Tensor(relevance).view(-1,1)

class ContextDatapoint:
    """Random augmentation in every epoch --> New Version
       Only input gold true data when initilizing. Do not input augmentated data.
    """
    def __init__(self,
                 r_smiles: str,
                 p_smiles: str,
                 context_: List[tuple],
                 args: TrainArgs_rxn,
                 solvent_classes,
                 reagent_classes,
                 slate_size: int = DEFAULT_SLATE_SIZE):
        """
        :param r_smiles: The SMILES string for the reactant molecule.
        :param p_smiles: The SMILES string fot the product molecule.
        :param context: A list contains context tuple which is (yield, reagent, solvent).
        :param args: TrainArgs_rxn
        :param slate_size: Upper limit number in one slate for listwise ranking.
        """
        context = context_.copy()
        self.r_smiles = r_smiles
        self.p_smiles = p_smiles
        if len(context) > slate_size: # when making the context datapoint, we do not consider too many conditions.
            context = context[:slate_size]
        self.slate_size = slate_size
        self.num_context = len(context)
        self.gold_length = len(context)
        self.gold_yields, self.gold_reagents, self.gold_solvents, self.temperatures = list(zip(*context))
        self.gold_yields = [float(y) for y in self.gold_yields]
        
        rxn_fp = create_rxn_Morgan2FP_concatenate(self.r_smiles, self.p_smiles, fpsize=args.fpsize, radius=args.radius)
        self.rxn_fp = torch.Tensor([rxn_fp])
        # self.rxn_fp = rxn_fp.repeat(self.num_context,1)

        self.make_initial_context_feature(solvent_classes, reagent_classes)
        self.len_solv = len(solvent_classes)
        self.len_reag = len(reagent_classes)
        self.gold_solv_index = set(torch.nonzero(self.solvent_features)[:,1].numpy())
        self.gold_reag_index = set(torch.nonzero(self.reagent_features)[:,1].numpy())
        # self.make_relevance()
        self.random_aug = False # Always not augmentated whem initialize
    
    def make_initial_context_feature(self, solvent_classes, reagent_classes) -> None:
        """Make one-hot encoding for solvent and reagent information. """
        reagent_features = []
        solvent_features = []
        for i in range(self.num_context):
            vec_solv, vec_reag = one_hot(self.gold_solvents[i], self.gold_reagents[i], solvent_classes, reagent_classes)
            solvent_features.append(vec_solv)
            reagent_features.append(vec_reag)
        self.reagent_features = torch.Tensor(reagent_features)
        self.solvent_features = torch.Tensor(solvent_features)
        
    
    def padding_reagent(self, padding_value=0) -> torch.Tensor:
        pad_num = self.slate_size-self.num_context
        if self.random_aug and self.num_fake:
            fake_reag = []
            for idx_ in self.fake_reag: # convert index to tensor
                temp = torch.zeros(self.len_reag)
                temp[idx_] = 1.
                fake_reag.append(temp)
            fake_reag = torch.stack(fake_reag)
            concat = torch.cat((self.reagent_features, fake_reag), dim=0)
            return nn.functional.pad(concat,
                                     (0,0,0,pad_num),
                                     mode = 'constant', value=padding_value)            
        else:
            return nn.functional.pad(self.reagent_features,
                                     (0,0,0,pad_num),
                                     mode = 'constant', value=padding_value)
    
    def padding_solvent(self, padding_value=0) -> torch.Tensor:
        pad_num = self.slate_size-self.num_context
        if self.random_aug and self.num_fake:
            fake_solv = []
            for idx_ in self.fake_solv: # convert index to tensor
                temp = torch.zeros(self.len_solv)
                temp[idx_] = 1.
                fake_solv.append(temp)
            fake_solv = torch.stack(fake_solv)
            concat = torch.cat((self.solvent_features, fake_solv), dim=0)
            return nn.functional.pad(concat,
                                     (0,0,0,pad_num),
                                     mode = 'constant', value=padding_value)            
        else:
            return nn.functional.pad(self.solvent_features,
                                     (0,0,0,pad_num),
                                     mode = 'constant', value=padding_value)
    
    def padding_rxn_fp(self, padding_value=0) -> torch.Tensor:
        """Repeat rxnfp here to save storage in each ContextDatapoint """
        rxn_fp = self.rxn_fp.repeat(self.num_context, 1)
        pad_num = self.slate_size-self.num_context
        return nn.functional.pad(rxn_fp,
                                 (0,0,0,pad_num),
                                 mode = 'constant', value=padding_value)

    def padding_relevance(self, padding_value= -1) -> torch.Tensor:
        if self.random_aug:
            relevance = Yield2Relevance(self.fake_yields)
        else:
            relevance = Yield2Relevance(self.gold_yields)
        pad_num = self.slate_size-self.num_context
        return nn.functional.pad(relevance,
                                 (0,0,0,pad_num),
                                 mode = 'constant', value=padding_value)
    
    
    def random_augmentation(self, num_fold):
        self.random_aug = True
        self.num_fold = num_fold
        self.num_context = min(self.slate_size, self.gold_length*(self.num_fold+1) )
        # print(self.num_context)
        self.num_fake = self.num_context - self.gold_length

        # pool = all condition set - gold true set
        pool_solv = set(range(self.len_solv)) - self.gold_solv_index
        pool_reag = set(range(self.len_reag)) - self.gold_reag_index
        # random sample fake consition from the ppol
        self.fake_solv = [random.sample(pool_solv, random.randint(1,2)) for i in range(self.num_fake)]
        self.fake_reag = [random.sample(pool_reag, random.randint(1,2)) for i in range(self.num_fake)]
        self.fake_yields = self.gold_yields.copy()
        self.fake_yields += [0.]*self.num_fake
    
    def cutoff_augmentation(self, num_fold, Evaluator):
        self.random_aug = True
        self.num_fold = num_fold
        self.num_context = min(self.slate_size, self.gold_length*(self.num_fold+1) )
        self.num_fake = self.num_context - self.gold_length
        self.fake_yields = self.gold_yields.copy()
        
        fake_combinations = Evaluator.create_fake_data(self.rxn_fp, self.gold_solv_index, self.gold_reag_index, self.num_fake)
        if fake_combinations:
            self.fake_solv, self.fake_reag, fake_yields = list(zip(*fake_combinations))
            self.fake_yields += list(fake_yields)
        else:
            self.fake_solv = []; self.fake_reag = []
        
    
    def cancel_augmentation(self): 
        self.random_aug = False
        self.num_context = self.gold_length
    


class ContextDataset(Dataset):

    def __init__(self, data: List[ContextDatapoint]):
        """
        :param data: A list of :class: ContextDatapoint
        """
        self._data = data
        self.length = len(data)
        self._random = Random()

    def random_augmentation(self, num_fold):
        """
        Redo random sampling fake reaction conition data-augmentation. Use it in every epoch.
        """
        for context in self._data:
            context.random_augmentation(num_fold)
        return
    
    def cutoff_augmentation(self, num_fold, Evaluator):
        """
        Redo cutoff sampling fake reaction conition data-augmentation.
        """
        for context in self._data:
            context.cutoff_augmentation(num_fold, Evaluator)
        return
    
    def cancel_augmentation(self):
        for context in self._data:
            context.cancel_augmentation()
        return

    def smiles(self):
        """
        Returns a list containing the SMILES associated with each molecule.

        :return: A list of SMILES strings.
        """
        
        return [(context.r_smiles, context.p_smiles) for context in self._data]
    
    def morgan_fingerprint(self) -> torch.Tensor:
        """
        Return the morgan fingerprint for the model input, 3-dimension
        [batch_size, slate_size, feature_size]
        """
        return torch.stack([context.padding_rxn_fp() for context in self._data])
    
    def solvents(self) -> torch.Tensor:
        """Return all the solvent condition in ContextDataset. """
        return torch.stack([context.padding_solvent() for context in self._data])
    
    def reagents(self) -> torch.Tensor:
        """Return all the reagent condition in contextDataset. """
        return torch.stack([context.padding_reagent() for context in self._data])

    def targets(self) -> torch.Tensor:
        """
        Returns the targets associated with each molecule. (Relevance)
        3 means highly relevant, 2 means medially relevant, 1 means slightly relevant,
        0 means non-relevant, and -1 means padding value.
        """

        return torch.stack([context.padding_relevance() for context in self._data])
    

    def shuffle(self, seed: int = None) -> None:
        """
        Shuffles the dataset.
        :param seed: Optional random seed.
        """
        if seed is not None:
            self._random.seed(seed)
        self._random.shuffle(self._data)

    def sort(self, key: Callable) -> None:
        """
        Sorts the dataset using the provided key.

        :param key: A function on a :class:`MoleculeDatapoint` to determine the sorting order.
        """
        self._data.sort(key=key)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).

        :return: The length of the dataset.
        """
        return self.length

    def __getitem__(self, item) -> List[ContextDatapoint]:
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[item]

def construct_context_batch(data) -> ContextDataset:
    """
    data: List[ContextDatapoint]
    Constructs a :class:`ContextDataset` from a list of :class:`ContextDatapoint`\ s.
    :param data: A list of :class:`ContextDatapoint`\ s.
    :return: A :class:`ContextDataset` containing all the :class:`ContextDatapoint`\ s.
    """
    context_data = ContextDataset(data)
    return context_data


class ContextSampler(Sampler):
    """A :class:`ContextSampler` samples data from a :class:`ContextDataset` for a :class:`ContextDataLoader`.
    """

    def __init__(self,
                 dataset: ContextDataset,
                 shuffle: bool = False,
                 seed: int = 0):
        """
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


class ContextDataLoader(DataLoader):
    """A :class:`ContextDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`ContextDataset`."""

    def __init__(self,
                 dataset: ContextDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 shuffle: bool = True,
                 seed: int = 0
                 ):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param cache: Whether to store the individual :class:`~chemprop.features.MolGraph` featurizations
                      for each molecule in a global cache.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._seed = seed

        self._sampler = ContextSampler(
            dataset=self._dataset,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(ContextDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=partial(construct_context_batch)
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """

        return [self._dataset[index].targets() for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[ContextDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(ContextDataLoader, self).__iter__()

    
""" New dataset and datapoint function for temperature regression: """

class TemperatureDatapoint:
    """
    Temperature regression.
    """
    def __init__(self,
                 r_smiles: str,
                 p_smiles: str,
                 reagent: str,
                 solvent: str,
                 temperature: str,
                 solvent_classes, 
                 reagent_classes,
                 args: TrainArgs_rxn):
        """
        :param r_smiles: The SMILES string for the reactant molecule.
        :param p_smiles: The SMILES string fot the product molecule.
        :param reagent: Gold true reagent for this reaction condition (ex: sodium tris(acetoxy)borohydride).
        :param solvent: Gold true solvent for this reaction condition (ex: chloroform).
        :param args: TrainArgs_rxn
        """
        self.r_smiles = r_smiles
        self.p_smiles = p_smiles

        self.gold_reagents = reagent
        self.gold_solvents = solvent
        self.temperatures = torch.Tensor([float(temperature)])
        
        rxn_fp = create_rxn_Morgan2FP_concatenate(self.r_smiles, self.p_smiles, fpsize=args.fpsize, radius=args.radius)
        self.rxn_fp = torch.Tensor(rxn_fp)

        self.make_initial_context_feature(solvent_classes, reagent_classes)
        
    def make_initial_context_feature(self,solvent_classes, reagent_classes) -> None:
        """Make one-hot encoding for solvent and reagent information. """
        vec_solv, vec_reag = one_hot(self.gold_solvents, self.gold_reagents, solvent_classes, reagent_classes)
        self.reagent_features = torch.Tensor(vec_reag)
        self.solvent_features = torch.Tensor(vec_solv)

class TemperatureDataset(Dataset):
    def __init__(self, data: List[TemperatureDatapoint]):
        """
        :param data: A list of :class: TemperatureDatapoint
        """
        self._data = data
        self.length = len(data)
        self._random = Random()
        
    def smiles(self):
        """
        Returns a list containing the SMILES associated with each molecule.

        :return: A list of SMILES strings.
        """
        
        return [(context.r_smiles, context.p_smiles) for context in self._data]
    
    def morgan_fingerprint(self) -> torch.Tensor:
        """
        Return the morgan fingerprint for the model input, 3-dimension
        [batch_size, slate_size, feature_size]
        """
        return torch.stack([context.rxn_fp.view(1,-1) for context in self._data])
    
    def solvents(self) -> torch.Tensor:
        """Return all the solvent condition in ContextDataset. """
        return torch.stack([context.solvent_features.view(1,-1) for context in self._data])
    
    def reagents(self) -> torch.Tensor:
        """Return all the reagent condition in contextDataset. """
        return torch.stack([context.reagent_features.view(1,-1) for context in self._data])

    def targets(self) -> torch.Tensor:
        """
        Returns the targets associated with each molecule. (Relevance)
        3 means highly relevant, 2 means medially relevant, 1 means slightly relevant,
        0 means non-relevant, and -1 means padding value.
        """

        return torch.stack([context.temperatures.view(1,-1) for context in self._data])
    

    def shuffle(self, seed: int = None) -> None:
        """
        Shuffles the dataset.
        :param seed: Optional random seed.
        """
        if seed is not None:
            self._random.seed(seed)
        self._random.shuffle(self._data)

    def sort(self, key: Callable) -> None:
        """
        Sorts the dataset using the provided key.

        :param key: A function on a :class:`MoleculeDatapoint` to determine the sorting order.
        """
        self._data.sort(key=key)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).

        :return: The length of the dataset.
        """
        return self.length

    def __getitem__(self, item) -> List[ContextDatapoint]:
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[item]

def construct_temperature_batch(data) -> TemperatureDataset:
    temp_data = TemperatureDataset(data)
    return temp_data

class TemperatureDataLoader(DataLoader):
    """A :class:`TemperatureDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`TemperatureDataset`."""

    def __init__(self,
                 dataset: TemperatureDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 shuffle: bool = True,
                 seed: int = 0
                 ):
        """
        :param dataset: The :class:`TemperatureDataset` containing the reaction condition to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._seed = seed

        self._sampler = ContextSampler(
            dataset=self._dataset,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(TemperatureDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=partial(construct_temperature_batch)
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """

        return [self._dataset[index].targets() for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[ContextDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(TemperatureDataLoader, self).__iter__()


def create_ContextDataset_for_listwise(data, args, solvent_classes, reagent_classes, window =  4, stride = 3) -> ContextDataset:
    """The input must be the sorted data. Returns a ContextDataset for model training 
       index 0: Reaxys id, 1: reactant smiles, 2: product smiles, 3: All reaction yield and context.
    """
    CDP_list = []
    for rxn in data:
        rsmiles = rxn[1]
        psmiles = rxn[2]
        context = rxn[3] # TODO: 對於多反應條件的反應，需要修正一下
        total_length = len(context)
        j = 0 # current location
        while total_length > j + window :
            CDP = ContextDatapoint(rsmiles, psmiles, context[j:j+window], args, solvent_classes, reagent_classes)
            CDP_list.append(CDP)
            j += stride
        # try:
        CDP = ContextDatapoint(rsmiles, psmiles, context, args, solvent_classes, reagent_classes)
        # except:
        #     print('Error: reaction condition data: '+rsmiles+'>>'+psmiles)
        #     # return ContextDataset(CDP_list)
        CDP_list.append(CDP)        
        
    return ContextDataset(CDP_list)

def create_TemperatureDataset_for_regression(data, args, solvent_classes, reagent_classes) -> TemperatureDataset:
    """The input must be the sorted data. Returns a TemperatureDataset for model training (regression)
       index 0: Reaxys id, 1: reactant smiles, 2: product smiles, 3: All reaction yield and context.
    """
    TDP_list = []
    for rxn in data:
        rsmiles = rxn[1]
        psmiles = rxn[2]
        for context in rxn[3]:
            # context index -> 0: yield, 1: reagent names, 2: solvent name, 3: temperature
            y_, reagent, solvent, temp = context
            if (temp == 'None') or (temp == 'nan'): # do not train with the data without temperature information
                continue
            TDP = TemperatureDatapoint(r_smiles = rsmiles, p_smiles = psmiles, reagent = reagent, solvent = solvent, temperature = temp, solvent_classes = solvent_classes, reagent_classes = reagent_classes, args=args )
            TDP_list.append(TDP)        
        
    return TemperatureDataset(TDP_list)
