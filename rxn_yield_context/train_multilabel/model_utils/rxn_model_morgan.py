# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:19:26 2020

@author: Lung-Yi
"""
from typing import List, Union

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from rxn_yield_context.train_multilabel.args_train import TrainArgs_rxn
from rxn_yield_context.train_multilabel.nn_utils import get_activation_function, initialize_weights
from rxn_yield_context.train_multilabel.data_utils import AsymmetricLossOptimized, FocalLoss # loss function for Multitask model


class ReactionModel_Morgan(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs_rxn):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(ReactionModel_Morgan, self).__init__()

        self.output_size = args.multilabel_num_classes
        self.sigmoid = nn.Sigmoid()
        #self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)


    def create_ffn(self, args: TrainArgs_rxn) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """

        first_linear_dim = args.fpsize*2
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers -->> to be the same on paper

        ffn = [
            dropout, # new , 2021/02/23
            nn.Linear(first_linear_dim, 800),
            activation,
            dropout,
            nn.Linear(800, 600),
        ]

        ffn.extend([
            activation,
            dropout,
            nn.Linear(600, self.output_size),
        ])
        self.ffn = nn.Sequential(*ffn)


    def forward(self,fp: torch.Tensor) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :return: The output of the :class:`MoleculeModel`, which is either property predictions
                 or molecule features if :code:`self.featurizer=True`.
        """
        output = self.ffn(fp)
        output = self.sigmoid(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output

class Multitask_Multilabel(nn.Module):

    def __init__(self, args: TrainArgs_rxn):
        """
        :param args: A :class:`args.TrainArgs` object containing model arguments.
        """
        super(Multitask_Multilabel, self).__init__()

        self.reagent_output_size = args.reagent_num_classes
        self.solvent_output_size = args.solvent_num_classes
        self.sigmoid = nn.Sigmoid()
        self.create_ffn(args)
        if args.loss == 'Focal':
            self.criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma) # TODO : loss funciton
        elif args.loss == 'Asym':
            self.criterion = AsymmetricLossOptimized()
        self.device = args.device
        self.batch_size = args.batch_size
        initialize_weights(self)
        self.log_var_s = torch.ones((1,), requires_grad=True, device=args.device) # solvent task
        self.log_var_r = torch.ones((1,), requires_grad=True, device=args.device) # reagent task

    def create_ffn(self, args: TrainArgs_rxn) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        if args.fp_type == 'morgan':
            first_linear_dim = args.fpsize*2
        elif args.fp_type == 'drfp':
            first_linear_dim = args.fpsize
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers -->> to be the same on paper

        ffn_share = [
            # dropout,
            nn.Linear(first_linear_dim, 512),
            activation,
            dropout,
        ]
        ########################## Elastic Part ############################
        append = [nn.Linear(512, 512),
                  activation,
                  dropout]*(args.num_shared_layer-1)
        ffn_share += append
        #################################################################### 
        
        ffn_reagent = [
            nn.Linear(512, 300),
            activation,
            dropout,

        ]
        ########################## Elastic Part ############################
        append = [nn.Linear(300, 300),
                  activation,
                  dropout]*(args.num_last_layer-1)
        ffn_reagent += append
        #################################################################### 
        ffn_reagent += [nn.Linear(300, self.reagent_output_size)]
        
        
        
        ffn_solvent = [
            nn.Linear(512, 100),
            activation,
            dropout,

        ]
        ########################## Elastic Part ############################
        append_s = [nn.Linear(100, 100),
                  activation,
                  dropout]*(args.num_last_layer-1)
        ffn_solvent += append_s
        #################################################################### 
        ffn_solvent += [nn.Linear(100, self.solvent_output_size)]
        
        self.ffn_share = nn.Sequential(*ffn_share) 
        self.ffn_reagent = nn.Sequential(*ffn_reagent)
        self.ffn_solvent = nn.Sequential(*ffn_solvent)

    def cal_loss(self, y_pred, y_true):    
        """
        Parameters
        ----------
        y_pred : tuple(torch.Tensor, torch.Tensor)
            tasks of input logits (solvent prediction, reagent prediction)
        y_true : tuple(torch.Tensor, torch.Tensor)
            targets (multi-label binarized vector)
        -------
        """

        loss = torch.zeros((y_pred[0].shape[0],), device = self.device)
        log_vars = [self.log_var_s, self.log_var_r]
        for i in range(len(y_pred)):
            precision = torch.exp(-log_vars[i])
            diff = self.criterion(y_pred[i], y_true[i])
            loss += torch.sum(precision * diff + log_vars[i], -1)
        
        return torch.mean(loss)

    def sep_loss(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred : tuple(torch.Tensor, torch.Tensor)
            tasks of input logits (solvent prediction, reagent prediction)
        y_true : tuple(torch.Tensor, torch.Tensor)
            targets (multi-label binarized vector)
        -------
        """
        loss = []
        for i in range(len(y_pred)):
            loss.append(self.criterion(y_pred[i], y_true[i]).item())
        
        return loss

        
        
    def forward(self,fp: torch.Tensor) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :return: The output of the :class:`MoleculeModel`, which is either property predictions
                 or molecule features if :code:`self.featurizer=True`.
        """
        output_share = self.ffn_share(fp)
        output_reagent = self.sigmoid(self.ffn_reagent(output_share))  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
        output_solvent = self.sigmoid(self.ffn_solvent(output_share))
        
        return output_solvent, output_reagent

class Multitask_Multilabel_BacthNorm(nn.Module):
    """No dropout, only one-dim batch normalization """
    def __init__(self, args: TrainArgs_rxn):
        """
        :param args: A :class:`args.TrainArgs` object containing model arguments.
        """
        super(Multitask_Multilabel_BacthNorm, self).__init__()

        self.reagent_output_size = args.reagent_num_classes
        self.solvent_output_size = args.solvent_num_classes
        self.sigmoid = nn.Sigmoid()
        self.create_ffn(args)
        if args.loss == 'Focal':
            self.criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma) # AsymmetricLossOptimized(reduction='batch_mean') # TODO : loss funciton
        elif args.loss == 'Asym':
            self.criterion = AsymmetricLossOptimized()
        self.device = args.device
        self.batch_size = args.batch_size
        initialize_weights(self)
        self.log_var_s = torch.ones((1,), requires_grad=True, device=args.device) # solvent task
        self.log_var_r = torch.ones((1,), requires_grad=True, device=args.device) # reagent task

    def create_ffn(self, args: TrainArgs_rxn) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        first_linear_dim = args.fpsize*2
        activation = get_activation_function(args.activation)

        # Create FFN layers -->> to be the same on paper

        ffn_share = [
            # dropout,
            nn.Linear(first_linear_dim, 512),
            nn.BatchNorm1d(num_features=512),
            activation,
        ]
        ########################## Elastic Part ############################
        append = [nn.Linear(512, 512),
                  nn.BatchNorm1d(num_features=512),
                  activation,
                  ]*(args.num_shared_layer-1)
        ffn_share += append
        #################################################################### 
        
        ffn_reagent = [
            nn.Linear(512, 300),
            nn.BatchNorm1d(num_features=300),
            activation,]
        ########################## Elastic Part ############################
        append = [nn.Linear(300, 300),
                  nn.BatchNorm1d(num_features=300),
                  activation,]*(args.num_last_layer-1)
        ffn_reagent += append
        #################################################################### 
        ffn_reagent += [nn.Linear(300, self.reagent_output_size)]
        
        ffn_solvent = [
            nn.Linear(512, 100),
            nn.BatchNorm1d(num_features=100),
            activation,]
        ########################## Elastic Part ############################
        append_s = [nn.Linear(100, 100),
                    nn.BatchNorm1d(num_features=100),
                    activation]*(args.num_last_layer-1)
        ffn_solvent += append_s
        #################################################################### 
        ffn_solvent += [nn.Linear(100, self.solvent_output_size)]
        
        self.ffn_share = nn.Sequential(*ffn_share) 
        self.ffn_reagent = nn.Sequential(*ffn_reagent)
        self.ffn_solvent = nn.Sequential(*ffn_solvent)

    def cal_loss(self, y_pred, y_true):    
        """
        Parameters
        ----------
        y_pred : tuple(torch.Tensor, torch.Tensor)
            tasks of input logits (solvent prediction, reagent prediction)
        y_true : tuple(torch.Tensor, torch.Tensor)
            targets (multi-label binarized vector)
        -------
        """

        loss = torch.zeros((y_pred[0].shape[0],), device = self.device)
        log_vars = [self.log_var_s, self.log_var_r]
        for i in range(len(y_pred)):
            precision = torch.exp(-log_vars[i])
            diff = self.criterion(y_pred[i], y_true[i])
            loss += torch.sum(precision * diff + log_vars[i], -1)
        
        return torch.mean(loss)
        
        
    def forward(self,fp: torch.Tensor) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :return: The output of the :class:`MoleculeModel`, which is either property predictions
                 or molecule features if :code:`self.featurizer=True`.
        """
        output_share = self.ffn_share(fp)
        output_reagent = self.sigmoid(self.ffn_reagent(output_share))  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
        output_solvent = self.sigmoid(self.ffn_solvent(output_share))
        
        return output_solvent, output_reagent

class ReactionModel_Pointwise(nn.Module):
    """A :class:`ReactionModel_Pointwise` is a model which ranks the reaction condition pointwise according to a scoring function"""

    def __init__(self, args: TrainArgs_rxn, len_solvent: int = 0, len_reagent: int = 0):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e., outputting the
                           learned features from the last layer prior to prediction rather than
                           outputting the actual property predictions.
        """
        super(ReactionModel_Pointwise, self).__init__()
        self.len_solvent = len_solvent
        self.len_reagent = len_reagent
        self.len_context = len_solvent + len_reagent
        self.output_size = 1
        self.relu = nn.ReLU()
        self.device = args.device
        if args.last_output_layer_pointwise == 'relu':
            self.last_output_layer_pointwise = nn.ReLU()
        elif args.last_output_layer_pointwise == 'sigmoid':
            self.last_output_layer_pointwise = nn.Sigmoid()

        self.create_ffn(args)

        initialize_weights(self)


    def create_ffn(self, args: TrainArgs_rxn) -> None:
        """ create ffn"""
        first_linear_dim = args.fpsize*2
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        
        ffn_h1_rxn_fp = [
            nn.Linear(first_linear_dim, args.h1_size_rxn_fp),
            activation,
            dropout
            ]
        
        ffn_h1_solvent = [
            nn.Linear(self.len_solvent, args.h_size_solvent),
            activation,
            dropout
            ]
        
        ffn_h1_reagent = [
            nn.Linear(self.len_reagent, args.h_size_reagent),
            activation,
            dropout
            ]
        
        h2_size_rxn_fp_input = args.h1_size_rxn_fp + args.h_size_solvent + args.h_size_reagent
        ffn_final = [
            nn.Linear(h2_size_rxn_fp_input, args.h2_size),
            activation,
            dropout,
            nn.Linear(args.h2_size, self.output_size)
            ]
        
        self.ffn1_rxn_fp = nn.Sequential(*ffn_h1_rxn_fp)
        self.ffn_h1_solvent = nn.Sequential(*ffn_h1_solvent)
        self.ffn_h1_reagent = nn.Sequential(*ffn_h1_reagent)
        self.ffn_final = nn.Sequential(*ffn_final)
        

    def forward(self,fp: torch.Tensor, condition_solvent: torch.Tensor, condition_reagent: torch.Tensor) -> torch.FloatTensor:
        """
        Runs the :class:`ReactionModel_Pointwise` on input.
        """
        # hidden_fp = self.ffn1(fp)
        # hidden_fp = torch.cat((hidden_fp, feature), 1)
        # output = self.ffn2(hidden_fp)
        # output = self.last_output_layer_pointwise(output)
        
        h1_rxn_fp = self.ffn1_rxn_fp(fp)
        h1_solvent = self.ffn_h1_solvent(condition_solvent)
        h1_reagent = self.ffn_h1_reagent(condition_reagent)
        
        h2_input = torch.cat((h1_rxn_fp, h1_solvent, h1_reagent), 1)
        output = self.ffn_final(h2_input)
        output = self.last_output_layer_pointwise(output)

        return output
    
    
class ReactionModel_Listwise(nn.Module):
    """A :class:`ReactionModel_Listwise` is a model which ranks the reaction condition listwise
       according to a scoring function.
    """
    def __init__(self, args: TrainArgs_rxn, len_solvent: int = 0, len_reagent: int = 0):
        super(ReactionModel_Listwise, self).__init__()
        self.len_solvent = len_solvent
        self.len_reagent = len_reagent
        self.len_context = len_solvent + len_reagent
        self.output_size = 1
        self.relu = nn.ReLU()
        self.device = args.device
        # don't apply last output activation now
        # use output activation on the loss function
        
        self.create_ffn(args)
        initialize_weights(self)
    
    def create_ffn(self, args: TrainArgs_rxn) -> None:
        """ create ffn"""
        first_linear_dim = args.fpsize*2
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        
        ffn_h1_rxn_fp = [
            #dropout, #
            nn.Linear(first_linear_dim, args.h1_size_rxn_fp),
            activation,
            dropout
            ]
        
        ffn_h1_solvent = [
            #dropout, #
            nn.Linear(self.len_solvent, args.h_size_solvent),
            activation,
            dropout
            ]
        
        ffn_h1_reagent = [
            #dropout, #
            nn.Linear(self.len_reagent, args.h_size_reagent),
            activation,
            dropout
            ]
        
        h2_size_rxn_fp_input = args.h1_size_rxn_fp + args.h_size_solvent + args.h_size_reagent
        ffn_final = [nn.Linear(h2_size_rxn_fp_input, args.h2_size),
                     activation,
                     dropout]
        append = [nn.Linear(args.h2_size, args.h2_size),
                  activation,
                  dropout]*(args.num_last_layer-1)
        ffn_final += append
        final = [nn.Linear(args.h2_size, self.output_size)]
        ffn_final += final
        
        self.ffn1_rxn_fp = nn.Sequential(*ffn_h1_rxn_fp)
        self.ffn_h1_solvent = nn.Sequential(*ffn_h1_solvent)
        self.ffn_h1_reagent = nn.Sequential(*ffn_h1_reagent)
        self.ffn_final = nn.Sequential(*ffn_final)
        

    def forward(self,fp: torch.Tensor, condition_solvent: torch.Tensor, condition_reagent: torch.Tensor) -> torch.FloatTensor:
        """
        Runs the :class:`ReactionModel_Pointwise` on input.
        """
        
        h1_rxn_fp = self.ffn1_rxn_fp(fp)
        h1_solvent = self.ffn_h1_solvent(condition_solvent)
        h1_reagent = self.ffn_h1_reagent(condition_reagent)
        # print(h1_rxn_fp.shape)
        # print(h1_solvent.shape)
        # print(h1_reagent.shape)
        
        h2_input = torch.cat((h1_rxn_fp, h1_solvent, h1_reagent), 2)
        output = self.ffn_final(h2_input)

        return output
    
class ReactionModel_LWTemp(nn.Module):
    """A :class:`ReactionModel_LWTemp` is a model which ranks the reaction condition listwise
       according to a scoring function and simultaneously predicts the temperature for this reaction condition.
    """
    def __init__(self, args: TrainArgs_rxn, len_solvent: int = 0, len_reagent: int = 0):
        super(ReactionModel_LWTemp, self).__init__()
        self.len_solvent = len_solvent
        self.len_reagent = len_reagent
        self.len_context = len_solvent + len_reagent
        self.output_size = 1
        self.relu = nn.ReLU()
        self.device = args.device
        # don't apply last output activation now
        # use output activation on the loss function
        
        self.create_ffn_share_layer(args)
        self.create_ffn_ranking_layer(args)
        self.create_ffn_temperature_layer(args)
        
        initialize_weights(self)
        self.log_var_rank = torch.ones((1,), requires_grad=True, device=args.device) # ranking task
        self.log_var_temp = torch.ones((1,), requires_grad=True, device=args.device) # temperature regression task
    
    def create_ffn_share_layer(self, args: TrainArgs_rxn) -> None:
        """ create the share layer for the input ( rxn_fp + reagent selection + solvent selection )."""
        first_linear_dim = args.fpsize*2
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        
        ffn_h1_rxn_fp = [
            #dropout, #
            nn.Linear(first_linear_dim, args.h1_size_rxn_fp),
            activation,
            dropout
            ]
        
        ffn_h1_solvent = [
            #dropout, #
            nn.Linear(self.len_solvent, args.h_size_solvent),
            activation,
            dropout
            ]
        
        ffn_h1_reagent = [
            #dropout, #
            nn.Linear(self.len_reagent, args.h_size_reagent),
            activation,
            dropout
            ]

        self.ffn1_rxn_fp = nn.Sequential(*ffn_h1_rxn_fp)
        self.ffn_h1_solvent = nn.Sequential(*ffn_h1_solvent)
        self.ffn_h1_reagent = nn.Sequential(*ffn_h1_reagent)
        
        
    def create_ffn_ranking_layer(self, args: TrainArgs_rxn) -> None:
        """ create the ranking layer after the hidden layer"""
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        
        h2_size_rxn_fp_input = args.h1_size_rxn_fp + args.h_size_solvent + args.h_size_reagent
        
        ffn_final = [nn.Linear(h2_size_rxn_fp_input, args.h2_size),
                     activation,
                     dropout]
        
        ########################## Elastic Part ############################
        append = [nn.Linear(args.h2_size, args.h2_size),
                  activation,
                  dropout]*(args.num_last_layer-1)
        ffn_final += append
        ####################################################################
        
        final = [nn.Linear(args.h2_size, self.output_size)]
        ffn_final += final
        self.ffn_final_ranking = nn.Sequential(*ffn_final)
        
    def create_ffn_temperature_layer(self, args: TrainArgs_rxn) -> None:
        """ create the temperature ranking layer after the hidden layer"""
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        
        h2_size_rxn_fp_input = args.h1_size_rxn_fp + args.h_size_solvent + args.h_size_reagent
        
        ffn_final = [nn.Linear(h2_size_rxn_fp_input, args.h2_size),
                     activation,
                     dropout]
        
        ########################## Elastic Part ############################
        append = [nn.Linear(args.h2_size, args.h2_size),
                  activation,
                  dropout]*(args.num_last_layer-1)
        ffn_final += append
        ####################################################################
        final = [nn.Linear(args.h2_size, self.output_size)]
        ffn_final += final
        self.ffn_final_temperature = nn.Sequential(*ffn_final)
        
    def forward(self,fp: torch.Tensor, condition_solvent: torch.Tensor, condition_reagent: torch.Tensor) -> torch.FloatTensor:
        """
        Runs the :class:`ReactionModel_Pointwise` on input.
        """
        
        h1_rxn_fp = self.ffn1_rxn_fp(fp)
        h1_solvent = self.ffn_h1_solvent(condition_solvent)
        h1_reagent = self.ffn_h1_reagent(condition_reagent)

        # h1_rxn_fp.dim() == 3 
        # [0: batch_size, 1: slate_size, 2: feature_size]
        h2_input = torch.cat((h1_rxn_fp, h1_solvent, h1_reagent), 2)
        
        # return (1) ranking result (2) temperature prediction
        return self.ffn_final_ranking(h2_input), self.ffn_final_temperature(h2_input)


class Simple_Multilabel(nn.Module):

    def __init__(self, args: TrainArgs_rxn):
        """
        :param args: A :class:`args.TrainArgs` object containing model arguments.
        """
        super(Simple_Multilabel, self).__init__()

        self.reagent_output_size = args.reagent_num_classes
        self.solvent_output_size = args.solvent_num_classes
        self.sigmoid = nn.Sigmoid()
        self.create_ffn(args)
        if args.loss == 'Focal':
            self.criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma) # AsymmetricLossOptimized(reduction='batch_mean') # TODO : loss funciton
        elif args.loss == 'Asym':
            self.criterion = AsymmetricLossOptimized()
        self.device = args.device
        self.batch_size = args.batch_size
        initialize_weights(self)
        self.log_var_s = torch.ones((1,), requires_grad=True, device=args.device) # solvent task
        self.log_var_r = torch.ones((1,), requires_grad=True, device=args.device) # reagent task

    def create_ffn(self, args: TrainArgs_rxn) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        if args.fp_type == 'morgan':
            first_linear_dim = 256
        elif args.fp_type == 'drfp':
            first_linear_dim = 256
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers -->> to be the same on paper

        ffn_share = [
            # dropout,
            nn.Linear(first_linear_dim, 512),
            activation,
            dropout,
        ]
        ########################## Elastic Part ############################
        append = [nn.Linear(512, 512),
                  activation,
                  dropout]*(args.num_shared_layer-1)
        ffn_share += append
        #################################################################### 
        
        ffn_reagent = [
            nn.Linear(512, 300),
            activation,
            dropout,

        ]
        ########################## Elastic Part ############################
        append = [nn.Linear(300, 300),
                  activation,
                  dropout]*(args.num_last_layer-1)
        ffn_reagent += append
        #################################################################### 
        ffn_reagent += [nn.Linear(300, self.reagent_output_size)]
        
        
        
        ffn_solvent = [
            nn.Linear(512, 200),
            activation,
            dropout,

        ]
        ########################## Elastic Part ############################
        append_s = [nn.Linear(200, 200),
                  activation,
                  dropout]*(args.num_last_layer-1)
        ffn_solvent += append_s
        #################################################################### 
        ffn_solvent += [nn.Linear(200, self.solvent_output_size)]
        
        self.ffn_share = nn.Sequential(*ffn_share) 
        self.ffn_reagent = nn.Sequential(*ffn_reagent)
        self.ffn_solvent = nn.Sequential(*ffn_solvent)

    def cal_loss(self, y_pred, y_true):    
        """
        Parameters
        ----------
        y_pred : tuple(torch.Tensor, torch.Tensor)
            tasks of input logits (solvent prediction, reagent prediction)
        y_true : tuple(torch.Tensor, torch.Tensor)
            targets (multi-label binarized vector)
        -------
        """

        loss = torch.zeros((y_pred[0].shape[0],), device = self.device)
        log_vars = [self.log_var_s, self.log_var_r]
        for i in range(len(y_pred)):
            precision = torch.exp(-log_vars[i])
            diff = self.criterion(y_pred[i], y_true[i])
            loss += torch.sum(precision * diff + log_vars[i], -1)
        
        return torch.mean(loss)
        
        
    def forward(self,fp: torch.Tensor) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :return: The output of the :class:`MoleculeModel`, which is either property predictions
                 or molecule features if :code:`self.featurizer=True`.
        """
        output_share = self.ffn_share(fp)
        output_reagent = self.sigmoid(self.ffn_reagent(output_share))  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
        output_solvent = self.sigmoid(self.ffn_solvent(output_share))
        
        return output_solvent, output_reagent


if __name__ == '__main__':
    A = torch.Tensor([[[1,2,3],[2,2,1]],[[4,2,1],[5,2,3]],[[6,2,1],[0,0,1]]])
    B = A + 1
    C = torch.cat((A,B), 2)
















