# -*- coding: utf-8 -*-
"""
MODEL_FILE_NAME = 'rxn_model_epoch.checkpoint'
"""
from logging import Logger
from argparse import ArgumentParser, Namespace
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange, tqdm
from torch.optim.lr_scheduler import ExponentialLR
from rxn_yield_context.train_multilabel.model_utils import Multitask_Multilabel
from rxn_yield_context.train_multilabel.args_train import TrainArgs_rxn
from rxn_yield_context.train_multilabel.data_utils import ReactionDataset, ReactionDatapoint, ReactionDataLoader
from rxn_yield_context.train_multilabel.nn_utils import NoamLR
from rxn_yield_context.train_multilabel.data_utils import AsymmetricLossOptimized
from rxn_yield_context.train_multilabel.train_utils import build_optimizer_MTL, build_lr_scheduler, save_rxn_model_checkpoint
import pickle

def get_rxn_data(input_path):
    f = open(input_path, 'r')
    data = f.readlines()
    f.close()
    for i in range(len(data)):
        data[i] = data[i].rstrip('\n').split('\t')
    data = list(zip(*data))
    rxn_id, rsmiles, psmiles, reagents, solvents = data
    return list(rsmiles), list(psmiles), list(reagents), list(solvents)

def get_classes(path):
    f = open(path, 'rb')
    dict_ = pickle.load(f)
    f.close()
    classes = sorted(dict_.items(), key=lambda d: d[1],reverse=True)
    classes = [x for x,y in classes]
    return classes

def create_target_with_classes(targets:str, classes:list):
    targets = targets.split('; ')
    vector = [x in targets for x in classes]
    return np.array(vector, dtype=float)

def create_ReactionDataset_MultiTask(data_path, solvent_classes, reagent_classes, args: TrainArgs_rxn) -> ReactionDataset:
    rsmiles_list, psmiles_list, reagent_targets, solvent_targets = get_rxn_data(data_path)
    assert len(rsmiles_list) == len(psmiles_list) == len(solvent_targets) == len(reagent_targets)
    RDP_list = [] # a list of ReactionDatapoint
    for i in range(len(rsmiles_list)):
        rsmiles = rsmiles_list[i]
        psmiles = psmiles_list[i]
        solvent = solvent_targets[i]
        reagent = reagent_targets[i]
        
        s = create_target_with_classes(solvent, solvent_classes)
        r = create_target_with_classes(reagent, reagent_classes)
        RDP = ReactionDatapoint(rsmiles, psmiles, fpsize = args.fpsize, radius=args.radius)
        RDP.set_solvent(s)
        RDP.set_reagent(r)
        RDP_list.append(RDP)

    return ReactionDataset(RDP_list)

def calculate_batch_accuracy(preds: torch.Tensor, targets: torch.Tensor, cutoff = 0.5) -> (float, float):
    assert preds.size() == targets.size()
    p = preds.clone()
    t = targets.clone()
    p = p.cpu().detach().numpy() # move data on gpt to cpu
    t = t.cpu().detach().numpy()
    acc_list = []
    acc_list_recall = []
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            p[i][j] = float(p[i][j] >= cutoff)
        acc_list.append(calculate_accuracy(p[i], t[i]))
        acc_list_recall.append(calculate_recall(p[i], t[i]))
    return (np.mean(acc_list), np.mean(acc_list_recall))

def calculate_batch_cutoff_accuracy(preds: torch.Tensor, targets: torch.Tensor, cutoff = 0.5) -> (float, float):
    """return one more variables: average number of predictions """
    assert preds.size() == targets.size()
    p = preds.clone()
    t = targets.clone()
    p = p.cpu().detach().numpy() # move data on gpt to cpu
    t = t.cpu().detach().numpy()
    acc_list = []
    recall_list = []
    number_p = []
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            p[i][j] = float(p[i][j] >= cutoff)
        acc_list.append(calculate_accuracy(p[i], t[i]))
        recall_list.append(calculate_recall(p[i], t[i]))
        number_p.append(number_preds(p[i]))
    return np.mean(acc_list), np.mean(recall_list), np.mean(number_p)

def number_preds(x: np.array):
    return len(x.nonzero()[0])

def calculate_accuracy(x: np.array, y: np.array):
    intersection = len(np.intersect1d(x.nonzero()[0], y.nonzero()[0]))
    union = len(np.union1d(x.nonzero()[0], y.nonzero()[0]))

    return intersection/union

def calculate_recall(x: np.array, y: np.array):
    denominator = len(y.nonzero()[0])
    numerator = len(np.intersect1d(x.nonzero()[0], y.nonzero()[0]))
    return numerator / denominator


def print_args_info(args):
    print('batch size:', args.batch_size)
    print('epoches:', args.epochs)
    print('data lengeth:', args.train_data_size)
    print('dropout probabilty:', args.dropout)
    print('weight decay for optmizer:', args.weight_decay)
    print('initial learning rate:', args.init_lr)
    print('max learning rate:', args.max_lr)
    print('final learning rate:', args.final_lr)
    print('warm up epochs:', args.warmup_epochs)
    print('Model save path: {}'.format(args.save_dir))
    print('Model device: {}'.format(args.device))
    print('fingerprint size: {}'.format(args.fpsize))
    print('fingerprint radius: {}'.format(args.radius))
    print('alpha weighing factor: {}'.format(args.alpha))
    print('gamma modulating factor: {}'.format(args.gamma))

    return 

if __name__ == '__main__':

    parser = TrainArgs_rxn()
    args = parser.parse_args()
    
    # args.train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # TODO
    # args.train_path = os.path.join(os.path.join(args.train_path, 'All_LCC_Data'), 'processed_data_temp') # TODO
    # args.save_dir = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'save_model'), 'MultiTask') # TODO
    

    train_data_path = os.path.join(os.path.join(args.train_path, 'For_first_part_model'), 'Splitted_first_train_labels_processed.txt')
    valid_data_path = os.path.join(os.path.join(args.train_path, 'For_first_part_model'), 'Splitted_first_validate_labels_processed.txt')
    
    solvent_class_path = os.path.join(os.path.join(args.train_path, 'label_processed'), 'class_names_solvent_labels_processed.pkl')
    solvent_classes = get_classes(solvent_class_path)
    reagent_class_path = os.path.join(os.path.join(args.train_path, 'label_processed'), 'class_names_reagent_labels_processed.pkl')
    reagent_classes = get_classes(reagent_class_path)  

    if args.save_dir == None:
        raise ValueError('Model save directory must be given.')
    os.makedirs(args.save_dir, exist_ok=True)
    args.device = torch.device('cuda')
    
    train_RDS = create_ReactionDataset_MultiTask(train_data_path, solvent_classes, reagent_classes, args) # TODO
    
    valid_RDS = create_ReactionDataset_MultiTask(valid_data_path, solvent_classes, reagent_classes, args) # TODO
    
    valid_fps = valid_RDS.morgan_fingerprint()
    valid_fps = torch.Tensor(valid_fps)
    valid_fps = valid_fps.to(args.device)
    valid_solvent = torch.Tensor(valid_RDS.solvents()).float().to(args.device)
    valid_reagent = torch.Tensor(valid_RDS.reagents()).float().to(args.device)
    
    del valid_RDS
    
    
    args.solvent_num_classes = len(solvent_classes)
    args.reagent_num_classes = len(reagent_classes)
    args.train_data_size = len(train_RDS)
    # args.dropout 
    # args.batch_size = 64
    rxn_model = Multitask_Multilabel(args = args)
    print(rxn_model)
    rxn_model = rxn_model.to(args.device) # move to gpu
    print_args_info(args)
    # # # Create data loaders
    train_data_loader = ReactionDataLoader(
        dataset=train_RDS,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
        morgan = True
    )

    
    # # # Get loss and metric functions

    # loss_func = AsymmetricLossOptimized()
    optimizer = build_optimizer_MTL(rxn_model, args)
    scheduler = build_lr_scheduler(optimizer, args)
    
    
    # # # Run training
    # # best_score = float('inf') if args.minimize_score else -float('inf')
    # # best_epoch, n_iter = 0, 0

    
    for epoch in trange(args.epochs):# TODO
        rxn_model.train()
        print('Learning Rate: '+ str(scheduler.get_lr()))
        solvent_acc_list_5 = []
        solvent_acc_list_4 = []
        solvent_acc_list_3 = []
        solvent_acc_list_2 = []
        solvent_acc_list_1 = []
        
        reagent_acc_list_5 = []
        reagent_acc_list_4 = []
        reagent_acc_list_3 = []
        reagent_acc_list_2 = []
        reagent_acc_list_1 = []
        loss_sum, iter_count = 0, 0
        for batch in train_data_loader:
            batch: ReactionDataset
            fps = batch.morgan_fingerprint()
            fps = torch.Tensor(fps)
            fps = fps.to(args.device)
            
            
            # targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])
            
            # Run model
            optimizer.zero_grad()
            preds = rxn_model(fps)
            
            target_solvent = torch.Tensor(batch.solvents()).float()
            target_reagent = torch.Tensor(batch.reagents()).float()
            # Move tensors to correct device
            target_solvent = target_solvent.to(preds[0].device)
            target_reagent = target_reagent.to(preds[0].device)
            

            loss = rxn_model.cal_loss(preds, [target_solvent, target_reagent])
            loss_sum += loss.item()*len(batch)
            iter_count += len(batch)
            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm_(rxn_model.parameters(), args.grad_clip)
            optimizer.step()
            if isinstance(scheduler, NoamLR):
                scheduler.step()
                
            
            if args.train_info:
                solvent_acc_list_5.append(calculate_batch_accuracy(preds[0], target_solvent, cutoff=0.5))
                solvent_acc_list_4.append(calculate_batch_accuracy(preds[0], target_solvent, cutoff=0.4))
                solvent_acc_list_3.append(calculate_batch_accuracy(preds[0], target_solvent, cutoff=0.3))
                solvent_acc_list_2.append(calculate_batch_accuracy(preds[0], target_solvent, cutoff=0.2))
                solvent_acc_list_1.append(calculate_batch_accuracy(preds[0], target_solvent, cutoff=0.1))
    
                
                reagent_acc_list_5.append(calculate_batch_accuracy(preds[1], target_reagent, cutoff=0.5))
                reagent_acc_list_4.append(calculate_batch_accuracy(preds[1], target_reagent, cutoff=0.4))
                reagent_acc_list_3.append(calculate_batch_accuracy(preds[1], target_reagent, cutoff=0.3))
                reagent_acc_list_2.append(calculate_batch_accuracy(preds[1], target_reagent, cutoff=0.2))
                reagent_acc_list_1.append(calculate_batch_accuracy(preds[1], target_reagent, cutoff=0.1))
        
        if args.train_info:    
            solvent_acc_list_5 = list(zip(*solvent_acc_list_5))
            solvent_acc_list_4 = list(zip(*solvent_acc_list_4))
            solvent_acc_list_3 = list(zip(*solvent_acc_list_3))
            solvent_acc_list_2 = list(zip(*solvent_acc_list_2))
            solvent_acc_list_1 = list(zip(*solvent_acc_list_1))
            reagent_acc_list_5 = list(zip(*reagent_acc_list_5))
            reagent_acc_list_4 = list(zip(*reagent_acc_list_4))
            reagent_acc_list_3 = list(zip(*reagent_acc_list_3))
            reagent_acc_list_2 = list(zip(*reagent_acc_list_2))
            reagent_acc_list_1 = list(zip(*reagent_acc_list_1))
            
            avg_loss = loss_sum / args.train_data_size
            print('-'*15+' Train '+'-'*15)
            print('\nepoch: {}'.format(epoch+1))    
            print('Overall two tasks of avg_BCE_loss: {:.5f}'.format(avg_loss))
            print('Solvent task 1:')
            print('cutoff = 0.5, avg. precision: {:.5f}, avg. recall: {:.5f}'.format(np.mean(solvent_acc_list_5[0][:-1]), np.mean(solvent_acc_list_5[1][:-1])))
            print('cutoff = 0.4, avg. precision: {:.5f}, avg. recall: {:.5f}'.format(np.mean(solvent_acc_list_4[0][:-1]), np.mean(solvent_acc_list_4[1][:-1])))
            print('cutoff = 0.3, avg. precision: {:.5f}, avg. recall: {:.5f}'.format(np.mean(solvent_acc_list_3[0][:-1]), np.mean(solvent_acc_list_3[1][:-1])))
            print('cutoff = 0.2, avg. precision: {:.5f}, avg. recall: {:.5f}'.format(np.mean(solvent_acc_list_2[0][:-1]), np.mean(solvent_acc_list_2[1][:-1])))
            print('cutoff = 0.1, avg. precision: {:.5f}, avg. recall: {:.5f}'.format(np.mean(solvent_acc_list_1[0][:-1]), np.mean(solvent_acc_list_1[1][:-1])))
            print('\nReagent task 2:')
            print('cutoff = 0.5, avg. precision: {:.5f}, avg. recall: {:.5f}'.format(np.mean(reagent_acc_list_5[0][:-1]), np.mean(reagent_acc_list_5[1][:-1])))
            print('cutoff = 0.4, avg. precision: {:.5f}, avg. recall: {:.5f}'.format(np.mean(reagent_acc_list_4[0][:-1]), np.mean(reagent_acc_list_4[1][:-1])))
            print('cutoff = 0.3, avg. precision: {:.5f}, avg. recall: {:.5f}'.format(np.mean(reagent_acc_list_3[0][:-1]), np.mean(reagent_acc_list_3[1][:-1])))
            print('cutoff = 0.2, avg. precision: {:.5f}, avg. recall: {:.5f}'.format(np.mean(reagent_acc_list_2[0][:-1]), np.mean(reagent_acc_list_2[1][:-1])))
            print('cutoff = 0.1, avg. precision: {:.5f}, avg. recall: {:.5f}'.format(np.mean(reagent_acc_list_1[0][:-1]), np.mean(reagent_acc_list_1[1][:-1])))
        
        
        ###################------- Start Validate ------######################
        rxn_model.eval()
        preds = rxn_model(valid_fps)
        val_loss = rxn_model.cal_loss(preds, [valid_solvent, valid_reagent])

        solvent_acc_tuple_5 = calculate_batch_cutoff_accuracy(preds[0], valid_solvent, cutoff = 0.5) # a tuple contains: (accuracy, recall, num of preds)avg.
        solvent_acc_tuple_4 = calculate_batch_cutoff_accuracy(preds[0], valid_solvent, cutoff = 0.4)
        solvent_acc_tuple_3 = calculate_batch_cutoff_accuracy(preds[0], valid_solvent, cutoff = 0.3)
        solvent_acc_tuple_2 = calculate_batch_cutoff_accuracy(preds[0], valid_solvent, cutoff = 0.2)
        solvent_acc_tuple_1 = calculate_batch_cutoff_accuracy(preds[0], valid_solvent, cutoff = 0.1)
        
        reagent_acc_tuple_5 = calculate_batch_cutoff_accuracy(preds[1], valid_reagent, cutoff = 0.5)
        reagent_acc_tuple_4 = calculate_batch_cutoff_accuracy(preds[1], valid_reagent, cutoff = 0.4)
        reagent_acc_tuple_3 = calculate_batch_cutoff_accuracy(preds[1], valid_reagent, cutoff = 0.3)
        reagent_acc_tuple_2 = calculate_batch_cutoff_accuracy(preds[1], valid_reagent, cutoff = 0.2)
        reagent_acc_tuple_1 = calculate_batch_cutoff_accuracy(preds[1], valid_reagent, cutoff = 0.1)  
        
        print('-'*15+' Validate '+'-'*15)
        print('\nepoch: {}'.format(epoch+1))
        print('Overall two tasks of validation avg_BCE_loss: {:.5f}'.format(val_loss.item()))
        print('Solvent task 1:')
        print('cutoff = 0.5, avg. precision: {:.5f}, avg. recall: {:.5f}, avg. number of preds: {:.2f}'.format(*solvent_acc_tuple_5))
        print('cutoff = 0.4, avg. precision: {:.5f}, avg. recall: {:.5f}, avg. number of preds: {:.2f}'.format(*solvent_acc_tuple_4))
        print('cutoff = 0.3, avg. precision: {:.5f}, avg. recall: {:.5f}, avg. number of preds: {:.2f}'.format(*solvent_acc_tuple_3))
        print('cutoff = 0.2, avg. precision: {:.5f}, avg. recall: {:.5f}, avg. number of preds: {:.2f}'.format(*solvent_acc_tuple_2))
        print('cutoff = 0.1, avg. precision: {:.5f}, avg. recall: {:.5f}, avg. number of preds: {:.2f}'.format(*solvent_acc_tuple_1))
        print('\nReagent task 2:')
        print('cutoff = 0.5, avg. precision: {:.5f}, avg. recall: {:.5f}, avg. number of preds: {:.2f}'.format(*reagent_acc_tuple_5))
        print('cutoff = 0.4, avg. precision: {:.5f}, avg. recall: {:.5f}, avg. number of preds: {:.2f}'.format(*reagent_acc_tuple_4))
        print('cutoff = 0.3, avg. precision: {:.5f}, avg. recall: {:.5f}, avg. number of preds: {:.2f}'.format(*reagent_acc_tuple_3))
        print('cutoff = 0.2, avg. precision: {:.5f}, avg. recall: {:.5f}, avg. number of preds: {:.2f}'.format(*reagent_acc_tuple_2))
        print('cutoff = 0.1, avg. precision: {:.5f}, avg. recall: {:.5f}, avg. number of preds: {:.2f}'.format(*reagent_acc_tuple_1))
        print('-'*20)
        
        ###################------- End Validate ------########################
        MODEL_FILE_NAME = 'multitask_model_epoch-{}.checkpoint'.format(epoch+1)
        save_rxn_model_checkpoint(os.path.join(args.save_dir, MODEL_FILE_NAME), rxn_model, args)
        
        std_s = torch.exp(rxn_model.log_var_s)**0.5
        std_r = torch.exp(rxn_model.log_var_r)**0.5
        print('log variance of solvent: ', std_s.item())
        print('log variance of reagent: ', std_r.item())
        print('-'*50+'\n')



