# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:52:00 2021

@author: Lung-Yi
"""
import torch
import torch.nn as nn
from typing import List
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from rxn_yield_context.train_multilabel.args_train import TrainArgs_rxn
from rxn_yield_context.train_multilabel.nn_utils import NoamLR
from argparse import ArgumentParser, Namespace

def build_optimizer(model: nn.Module, args: TrainArgs_rxn) -> Optimizer:
    """
    Builds a PyTorch Optimizer.

    :param model: The model to optimize.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing optimizer arguments.
    :return: An initialized Optimizer.
    """
    params = [{'params': model.parameters(), 'lr': args.init_lr, 'weight_decay': 0}]

    return Adam(params)


def build_lr_scheduler(optimizer: Optimizer, args: TrainArgs_rxn, total_epochs: List[int] = None) -> _LRScheduler:
    """
    Builds a PyTorch learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing learning rate arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs], # 2.0
        total_epochs=total_epochs or [args.epochs] * args.num_lrs,
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr], #1e-4
        max_lr=[args.max_lr], #1e-3
        final_lr=[args.final_lr] #1e-4
    )


def save_rxn_model_checkpoint(path: str,
                    model,
                    args: TrainArgs_rxn = None) -> None:
    """
    Saves a model checkpoint.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the data.
    :param features_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the features.
    :param args: The :class:`~chemprop.args.TrainArgs` object containing the arguments the model was trained with.
    :param path: Path where checkpoint will be saved.
    """
    # Convert args to namespace for backwards compatibility
    if args is not None:
        args = Namespace(**args.as_dict())

    state = {'args': args, 'state_dict': model.state_dict()}
    torch.save(state, path)
    
def build_optimizer_MTL(model, args: TrainArgs_rxn) -> Optimizer:
    """
    Builds a PyTorch MultiTask Optimizer. *Need to optimize homoscedastic uncertainty.

    :param model: The model to optimize.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing optimizer arguments.
    :return: An initialized Optimizer.
    """
    params = [{'params':([p for p in model.parameters()] + [model.log_var_s] + [model.log_var_r]), 
               'lr': args.init_lr, 'weight_decay': args.weight_decay}]

    return Adam(params)