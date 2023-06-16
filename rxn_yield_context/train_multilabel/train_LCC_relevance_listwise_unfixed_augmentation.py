# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:30:01 2020

@author: Lung-Yi

Apply data augmentation in every epoch.

"""

import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer

from tqdm import trange, tqdm
from torch.optim.lr_scheduler import ExponentialLR
from rxn_yield_context.train_multilabel.model_utils import ReactionModel_Listwise, ReactionModel_LWTemp
from rxn_yield_context.train_multilabel.args_train import TrainArgs_rxn

from rxn_yield_context.train_multilabel.data_utils import ContextDatapoint, ContextDataset, ContextDataLoader, get_classes
from rxn_yield_context.train_multilabel.data_utils import TemperatureDatapoint, TemperatureDataset, TemperatureDataLoader
from rxn_yield_context.train_multilabel.data_utils import listNet_top_one, listMLE # loss function for listwise
from rxn_yield_context.train_multilabel.data_utils import create_ContextDataset_for_listwise, create_TemperatureDataset_for_regression
from rxn_yield_context.train_multilabel.data_utils import create_rxn_Morgan2FP_concatenate

from rxn_yield_context.train_multilabel.nn_utils import NoamLR
from rxn_yield_context.preprocess_data import sort_out_data
import pickle
from rxn_yield_context.train_multilabel.train_utils import build_optimizer, build_lr_scheduler, save_rxn_model_checkpoint
from rxn_yield_context.evaluate_model.eval_utils import get_answer, compare_answer_and_combinations, evaluate_overall, sort_string
from rxn_yield_context.evaluate_model.eval_utils import MultiTask_Evaluator

def get_data(input_path):
    f = open(input_path, 'r')
    data = f.readlines()
    f.close()
    return sort_out_data(data)

def build_optimizer_MTL_2(model, args: TrainArgs_rxn) -> Optimizer:
    """
    Builds a PyTorch MultiTask Optimizer. *Need to optimize homoscedastic uncertainty.

    :param model: The model to optimize.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing optimizer arguments.
    :return: An initialized Optimizer.
    """
    params = [{'params':([p for p in model.parameters()] + [model.log_var_rank] + [model.log_var_temp]), 
               'lr': args.init_lr, 'weight_decay': args.weight_decay}]
    return Adam(params)

def cal_multitask_loss(rxn_model, loss_rank, loss_temp):      
    loss_total = torch.exp(-rxn_model.log_var_rank)*loss_rank + torch.exp(-rxn_model.log_var_temp)*loss_temp + (rxn_model.log_var_rank + rxn_model.log_var_temp)*2
    return loss_total

def print_args_info(args):
    print('batch size:', args.batch_size)
    print('epoches:', args.epochs)
    print('dropout probabilty:', args.dropout)
    print('weight decay for optmizer:', args.weight_decay)
    print('initial learning rate:', args.init_lr)
    print('max learning rate:', args.max_lr)
    print('final learning rate:', args.final_lr)
    print('warm up epochs:', args.warmup_epochs)
    print('Model save path: {}'.format(args.save_dir))


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TrainArgs_rxn()
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
#################################################### delete
    # args.epochs = 10
    # args.save_dir = "D:/Retro"
    # args.train_path = "../All_LCC_Data/processed_data_temp_large_2"
    # args.checkpoint_path = "../save_model/MultiTask_temp_large_2/multitask_model_epoch-50.checkpoint" # load multitask model
    # args.dropout = 0.3
    # args.num_workers = 0
    # args.num_fold = 6
    # args.cutoff_solv = 0.15 # cutoff of solvent for data augmentation
    # args.cutoff_reag = 0.15 # cutoff of reagent for data augmentation
    # args.redo_epoch = 3
#################################################### delete
    if args.save_dir == None:
        raise ValueError('Model save directory must be given.')
    
    os.makedirs(args.save_dir, exist_ok=True)
    print('savel model path: '+args.save_dir)
    print_args_info(args)
    """ Load basic classes, training data and listwise model to trian """
    
    data_path = os.path.join(os.path.join(args.train_path, 'For_second_part_model'),  'Splitted_second_train_labels_processed.txt')
    
    solvent_class_path = os.path.join(os.path.join(args.train_path, 'label_processed'), 'class_names_solvent_labels_processed.pkl')
    reagent_class_path = os.path.join(os.path.join(args.train_path, 'label_processed'), 'class_names_reagent_labels_processed.pkl')
    solvent_classes = get_classes(solvent_class_path)
    reagent_classes = get_classes(reagent_class_path)
    print('data_path: '+data_path)
    data = get_data(data_path)
    CDS = create_ContextDataset_for_listwise(data, args, solvent_classes, reagent_classes)
    TDS = create_TemperatureDataset_for_regression(data, args, solvent_classes, reagent_classes)
    print('original number of training data: {}'.format(len(data)))
    print('processed number of rank traininng data: {}'.format(len(CDS)))
    print('number of temperature traininig data: {}'.format(len(TDS)))


    args.train_data_size = len(CDS)

    rxn_model = ReactionModel_LWTemp(args = args,len_solvent = len(solvent_classes), len_reagent = len(reagent_classes))
    print('\nModel framework:')
    print(rxn_model)
    rxn_model = rxn_model.to(args.device) # move to gpu
    # # Create data loaders
    train_data_loader_ranking = ContextDataLoader(
        dataset=CDS,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed
    )
    train_data_loader_temperature = TemperatureDataLoader(
        dataset=TDS,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed
    )
    print('length of ranking train dataloader: {}'.format(len(train_data_loader_ranking)))
    print('length of temperature train dataloader: {}'.format(len(train_data_loader_temperature)))
    
    # # # Get loss functions
    loss_func_rank = listNet_top_one
    loss_func_regression = nn.MSELoss(reduction = 'mean') # mean

    optimizer = build_optimizer_MTL_2(rxn_model, args)
    scheduler = build_lr_scheduler(optimizer, args)

    """ Load Multitask_Evaluator for fake data cutoff augmentation (or random augmentation, not recommemded) """
    Evaluator = MultiTask_Evaluator(solvent_classes, reagent_classes, cutoff_solv = args.cutoff_solv, cutoff_reag = args.cutoff_reag)
    Evaluator.load_model(args.checkpoint_path, args.device)
    
    """ Get validation data """
    print('Processing validation data...')
    val_path = os.path.join(os.path.join(args.train_path, 'For_second_part_model'),  'Splitted_second_validate_labels_processed.txt')
    val_data = get_data(val_path)
    Evaluator.reset_cutoff(cutoff_solv = 0.25, cutoff_reag = 0.25 ) # validation cutoff is different from the data augmentation cutoff
    for i in range(len(val_data)):
        val_fp = torch.Tensor(create_rxn_Morgan2FP_concatenate(val_data[i][1], val_data[i][2], fpsize=args.fpsize, radius=args.radius))
        val_data[i].extend([val_fp])
        val_contexts = Evaluator.make_input_rxn_conditionBYnames(rxn_fp = val_fp)
        val_data[i].extend([val_contexts])
    del val_fp
    torch.cuda.empty_cache()
    
    """ Run training """
    print('Start training...')
    Evaluator.reset_cutoff(cutoff_solv = args.cutoff_solv, cutoff_reag = args.cutoff_reag)
    for epoch in range(args.epochs):# args.epochs
        rxn_model.train()
        epoch_loss_rank = 0
        epoch_loss_temp = 0
        epoch_loss_total = 0
        
        rank_iterator = iter(train_data_loader_ranking)
        
        for batch_temp in train_data_loader_temperature:
            ##################### tempeature loss computation ################
            fps = batch_temp.morgan_fingerprint()
            fps = fps.to(args.device)
            
            solvent_features = batch_temp.solvents()
            solvent_features = solvent_features.to(args.device)
            
            reagent_features = batch_temp.reagents()
            reagent_features = reagent_features.to(args.device)
            
            targets = batch_temp.targets()
                        
            # Run model
            _, preds_temp = rxn_model(fp = fps, condition_solvent = solvent_features, condition_reagent = reagent_features)
            # Move tensors to correct device
            targets = targets.to(preds_temp.device)
            targets = targets.float()
            loss_temp = loss_func_regression(preds_temp, targets)
            epoch_loss_temp += loss_temp
            
            ##################### rank loss computation ######################
            try:
                batch_context = next(rank_iterator)
                ################################# do cutoff augmentation every <args.redo_epoch> epoch
                if epoch % args.redo_epoch == 0:
                    batch_context.cutoff_augmentation(args.num_fold, Evaluator)
                #################################
            except StopIteration:
                rank_iterator = iter(train_data_loader_ranking)
                batch_context = next(rank_iterator)

            fps = batch_context.morgan_fingerprint()
            fps = fps.to(args.device)
            
            solvent_features = batch_context.solvents()
            solvent_features = solvent_features.to(args.device)
            
            reagent_features = batch_context.reagents()
            reagent_features = reagent_features.to(args.device)
            
            targets = batch_context.targets()
            
            # Run model
            
            preds_rank, _ = rxn_model(fp = fps, condition_solvent = solvent_features, condition_reagent = reagent_features)
            
            # Move tensors to correct device
            targets = targets.to(preds_rank.device)
            targets = targets.float()
            loss_rank = loss_func_rank(preds_rank, targets)
            epoch_loss_rank += loss_rank
            
            ##################################################################
            loss_total = cal_multitask_loss(rxn_model, loss_rank, loss_temp)
            epoch_loss_total += loss_total
            if args.grad_clip:
                nn.utils.clip_grad_norm_(rxn_model.parameters(), args.grad_clip)
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            if isinstance(scheduler, NoamLR):
                scheduler.step()
 
        avg_rank_loss = epoch_loss_rank / len(train_data_loader_ranking)
        avg_temp_loss = epoch_loss_temp / len(train_data_loader_ranking)
        avg_total_loss = epoch_loss_total / len(train_data_loader_ranking)
        print('\nepoch: {}'.format(epoch+1))    
        print('avg_ListNet_top_one_loss: {:.5f}'.format(avg_rank_loss))
        print('avg_temperature_mse_loss: {:.5f}'.format(avg_temp_loss))
        print('avg_multitask_total_loss: {:.5f}'.format(avg_total_loss[0]))
        print('log variance of ranking task: {:.3f}'.format(rxn_model.log_var_rank[0]))
        print('log variance of temperature task: {:.3f}'.format(rxn_model.log_var_temp[0]))
        
        """ Evaluate every ten epochs """
        
        if (epoch+1) % 10 == 0:
            rxn_model.eval()
            # save model
            MODEL_FILE_NAME = 'rxn_model_relevance_listwise_morgan_epoch-{}.checkpoint'.format(epoch+1)
            save_rxn_model_checkpoint(os.path.join(args.save_dir, MODEL_FILE_NAME), rxn_model, args)
            # evaluate
            acc_list = []
            ######
            for i, rxn in enumerate(val_data):
                contexts = rxn[5]
                rxn_fp = rxn[4].view(1,-1)
                rxn_fp = rxn_fp.repeat(len(contexts), 1)

                rxn_fp = rxn_fp.unsqueeze(0).to(args.device)
                solvent_batch, reagent_batch = Evaluator.convert_contexts2tensor(contexts)
                solvent_batch = solvent_batch.unsqueeze(0).to(args.device)
                reagent_batch = reagent_batch.unsqueeze(0).to(args.device)
                scores, temp_ = rxn_model(rxn_fp, solvent_batch, reagent_batch)
                scores = F.softmax(scores, dim=1).view(-1)
                top_index = torch.argsort(scores, descending=True)[:20] # select topk = 20 context combination
                
                gold_answers = get_answer(rxn)
                top_contexts = [sort_string(contexts[j]) for j in top_index]
                
                id_ = compare_answer_and_combinations(gold_answers, top_contexts)                        
                acc_list.append(id_)

            evaluate_overall(acc_list)
        
