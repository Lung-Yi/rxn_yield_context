# -*- coding: utf-8 -*-
"""
Train the first model for candidate generation.
"""
from logging import Logger
from argparse import ArgumentParser, Namespace
import os
import sys
from typing import List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
import time
import wandb

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
        RDP = ReactionDatapoint(rsmiles, psmiles, fpsize = args.fpsize, radius=args.radius, fp_type = args.fp_type)
        RDP.set_solvent(s)
        RDP.set_reagent(r)
        RDP_list.append(RDP)

    return ReactionDataset(RDP_list)

def calculate_batch_cutoff_accuracy(preds: torch.Tensor, targets: torch.Tensor, cutoff = 0.5):
    """return one more variables: average number of predictions """
    assert preds.size() == targets.size()
    p = preds.clone()
    t = targets.clone()
    p = p.cpu().detach().numpy() # move data on gpu to cpu
    t = t.cpu().detach().numpy()
    acc_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    number_p = []
    p = (p >= cutoff).astype(int)

    for i in range(p.shape[0]):
        acc_list.append(accuracy_score(t[i], p[i]))
        precision_list.append(precision_score(t[i], p[i], zero_division = 0))
        recall_list.append(recall_score(t[i], p[i]))
        f1_list.append(f1_score(t[i], p[i], zero_division = 0))
        number_p.append(number_preds(p[i]))
    return np.mean(acc_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_list), np.mean(number_p)

def number_preds(x: np.array):
    return len(x.nonzero()[0])

# def calculate_accuracy(x: np.array, y: np.array):
#     intersection = len(np.intersect1d(x.nonzero()[0], y.nonzero()[0]))
#     union = len(np.union1d(x.nonzero()[0], y.nonzero()[0]))

#     return intersection/union

# def calculate_recall(x: np.array, y: np.array):
#     denominator = len(y.nonzero()[0])
#     numerator = len(np.intersect1d(x.nonzero()[0], y.nonzero()[0]))
#     return numerator / denominator


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

def login_wandb(args):
    hyperparameter_dict = {"init_lr": args.init_lr, "max_lr": args.max_lr, "final_lr": args.final_lr,
                           "warmup_epochs": args.warmup_epochs, "model_save_path": args.save_dir,
                           "fp_size": args.fpsize, "fp_radius": args.radius, "alpha": args.alpha,
                           "gamma": args.gamma, "dropout": args.dropout, "batch_size": args.batch_size,
                           "epochs": args.epochs, "activation": args.activation, "h1_size_rxn_fp": args.h1_size_rxn_fp,
                           "hidden_share_size": args.hidden_share_size, "hidden_reagent_size": args.hidden_reagent_size,
                           "hidden_solvent_size": args.hidden_solvent_size}
    wandb.init(project="rxn_yield_context_first_model", config=hyperparameter_dict)
    return

if __name__ == '__main__':

    parser = TrainArgs_rxn()
    args = parser.parse_args()
    login_wandb(args)

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
        seed=args.seed
    )

    
    # # # Get loss and metric functions

    optimizer = build_optimizer_MTL(rxn_model, args)
    scheduler = build_lr_scheduler(optimizer, args)
    
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
                solvent_acc_list_5.append(calculate_batch_cutoff_accuracy(preds[0], target_solvent, cutoff=0.5))
                solvent_acc_list_4.append(calculate_batch_cutoff_accuracy(preds[0], target_solvent, cutoff=0.4))
                solvent_acc_list_3.append(calculate_batch_cutoff_accuracy(preds[0], target_solvent, cutoff=0.3))
                solvent_acc_list_2.append(calculate_batch_cutoff_accuracy(preds[0], target_solvent, cutoff=0.2))
                solvent_acc_list_1.append(calculate_batch_cutoff_accuracy(preds[0], target_solvent, cutoff=0.1))
    
                
                reagent_acc_list_5.append(calculate_batch_cutoff_accuracy(preds[1], target_reagent, cutoff=0.5))
                reagent_acc_list_4.append(calculate_batch_cutoff_accuracy(preds[1], target_reagent, cutoff=0.4))
                reagent_acc_list_3.append(calculate_batch_cutoff_accuracy(preds[1], target_reagent, cutoff=0.3))
                reagent_acc_list_2.append(calculate_batch_cutoff_accuracy(preds[1], target_reagent, cutoff=0.2))
                reagent_acc_list_1.append(calculate_batch_cutoff_accuracy(preds[1], target_reagent, cutoff=0.1))
        
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
        
        if (epoch + 1) % args.valid_per_epoch == 0:
            ###################------- Start Validate ------######################
            rxn_model.eval()
            preds = rxn_model(valid_fps)
            val_losses = rxn_model.sep_loss(preds, [valid_solvent, valid_reagent])

            A = time.time()
            solvent_acc_tuple_5 = calculate_batch_cutoff_accuracy(preds[0], valid_solvent, cutoff = 0.5)
            solvent_acc_tuple_4 = calculate_batch_cutoff_accuracy(preds[0], valid_solvent, cutoff = 0.4)
            solvent_acc_tuple_3 = calculate_batch_cutoff_accuracy(preds[0], valid_solvent, cutoff = 0.3)
            solvent_acc_tuple_2 = calculate_batch_cutoff_accuracy(preds[0], valid_solvent, cutoff = 0.2)
            solvent_acc_tuple_1 = calculate_batch_cutoff_accuracy(preds[0], valid_solvent, cutoff = 0.1)
            
            reagent_acc_tuple_5 = calculate_batch_cutoff_accuracy(preds[1], valid_reagent, cutoff = 0.5)
            reagent_acc_tuple_4 = calculate_batch_cutoff_accuracy(preds[1], valid_reagent, cutoff = 0.4)
            reagent_acc_tuple_3 = calculate_batch_cutoff_accuracy(preds[1], valid_reagent, cutoff = 0.3)
            reagent_acc_tuple_2 = calculate_batch_cutoff_accuracy(preds[1], valid_reagent, cutoff = 0.2)
            reagent_acc_tuple_1 = calculate_batch_cutoff_accuracy(preds[1], valid_reagent, cutoff = 0.1)  
            
            print("\nTime:", time.time()-A)
            print('-'*15+' Validate '+'-'*15)
            print('\nepoch: {}'.format(epoch+1))
            print('Overall two tasks of validation avg_BCE_loss: {:.4f} and {:.4f}'.format(*val_losses))
            print('Solvent task 1:')
            print('cutoff = 0.5, acc: {:.5f}, precision: {:.5f}, recall: {:.5f}, f1-score: {:.5f}, number of preds: {:.2f}'.format(*solvent_acc_tuple_5))
            print('cutoff = 0.4, acc: {:.5f}, precision: {:.5f}, recall: {:.5f}, f1-score: {:.5f}, number of preds: {:.2f}'.format(*solvent_acc_tuple_4))
            print('cutoff = 0.3, acc: {:.5f}, precision: {:.5f}, recall: {:.5f}, f1-score: {:.5f}, number of preds: {:.2f}'.format(*solvent_acc_tuple_3))
            print('cutoff = 0.2, acc: {:.5f}, precision: {:.5f}, recall: {:.5f}, f1-score: {:.5f}, number of preds: {:.2f}'.format(*solvent_acc_tuple_2))
            print('cutoff = 0.1, acc: {:.5f}, precision: {:.5f}, recall: {:.5f}, f1-score: {:.5f}, number of preds: {:.2f}'.format(*solvent_acc_tuple_1))
            print('\nReagent task 2:')
            print('cutoff = 0.5, acc: {:.5f}, precision: {:.5f}, recall: {:.5f}, f1-score: {:.5f}, number of preds: {:.2f}'.format(*reagent_acc_tuple_5))
            print('cutoff = 0.4, acc: {:.5f}, precision: {:.5f}, recall: {:.5f}, f1-score: {:.5f}, number of preds: {:.2f}'.format(*reagent_acc_tuple_4))
            print('cutoff = 0.3, acc: {:.5f}, precision: {:.5f}, recall: {:.5f}, f1-score: {:.5f}, number of preds: {:.2f}'.format(*reagent_acc_tuple_3))
            print('cutoff = 0.2, acc: {:.5f}, precision: {:.5f}, recall: {:.5f}, f1-score: {:.5f}, number of preds: {:.2f}'.format(*reagent_acc_tuple_2))
            print('cutoff = 0.1, acc: {:.5f}, precision: {:.5f}, recall: {:.5f}, f1-score: {:.5f}, number of preds: {:.2f}'.format(*reagent_acc_tuple_1))
            print('-'*20)
            
            for ii, (acc, prec, recall, f1, num_pred) in enumerate([solvent_acc_tuple_1, solvent_acc_tuple_2, 
                                                          solvent_acc_tuple_3, solvent_acc_tuple_4, solvent_acc_tuple_5]):
                log_dict = {f"solvent 0.{ii+1} accuracy": acc, f"solvent 0.{ii+1} precision": prec, 
                            f"solvent 0.{ii+1} recall": recall, f"solvent 0.{ii+1} f1-score": f1, f"solvent 0.{ii+1} number preds":num_pred}
                wandb.log(log_dict)
            for ii, (acc, prec, recall, f1, num_pred) in enumerate([reagent_acc_tuple_1, reagent_acc_tuple_2, 
                                                          reagent_acc_tuple_3, reagent_acc_tuple_4, reagent_acc_tuple_5]):
                log_dict = {f"reagent 0.{ii+1} accuracy": acc, f"reagent 0.{ii+1} precision": prec, 
                            f"reagent 0.{ii+1} recall": recall, f"reagent 0.{ii+1} f1-score": f1, f"reagent 0.{ii+1} number preds":num_pred}
                wandb.log(log_dict)
            overall_f1score03 = solvent_acc_tuple_3[3] + reagent_acc_tuple_3[3]
            wandb.log({"sum 0.3 f1-score": overall_f1score03})
            log_dict = {"epoch":epoch, "log varaince solvent": rxn_model.log_var_s.item(), "log variance reagent": rxn_model.log_var_r.item()}
            wandb.log(log_dict)
            MODEL_FILE_NAME = 'multitask_model_epoch-{}.checkpoint'.format(epoch+1)
            save_rxn_model_checkpoint(os.path.join(args.save_dir, MODEL_FILE_NAME), rxn_model, args)
            ###################------- End Validate ------########################
        
        print('log variance of solvent: ', rxn_model.log_var_s.item())
        print('log variance of reagent: ', rxn_model.log_var_r.item())
        print('-'*50+'\n')



