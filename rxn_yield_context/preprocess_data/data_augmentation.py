# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:15:33 2020

@author: Lung-Yi
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from itertools import combinations
import random

from rxn_yield_context.train_multilabel.data_utils import get_classes, create_rxn_Morgan2FP_concatenate
from rxn_yield_context.train_multilabel.args_train import TrainArgs_rxn
import argparse

def sort_out_data(data):
    data = [line.strip('\n').split('\t') for line in data]
    sorted_data = []
    rxn_id = ''
    j = -1
    for i in range(len(data)):
        if rxn_id != data[i][0]:
            j += 1
            sorted_data.append([data[i][0], data[i][1], data[i][2], []])
            y_ = data[i][3]; r_ = remove_duplicated_records(data[i][4]);
            s_ = remove_duplicated_records(data[i][5]); t_ = data[i][6];
            sorted_data[j][3].append((y_, r_, s_, t_))
            rxn_id = data[i][0]
        else:
            if (r_ == remove_duplicated_records(data[i][4])) & (s_ == remove_duplicated_records(data[i][5])):
                pass
            else:
                y_ = data[i][3]; r_ = remove_duplicated_records(data[i][4]);
                s_ = remove_duplicated_records(data[i][5]); t_ = data[i][6];
                sorted_data[j][3].append((y_, r_, s_, t_))
    return sorted_data

def remove_duplicated_records(records):
    return '; '.join(list(dict.fromkeys(records.split('; '))))

def create_cutoff(preds, type_, cutoff=0.1):
    """Create cutoff function to select wanted candidates."""
    global solvent_classes
    global reagent_classes
    if type_ == 'solvent': 
        condition = solvent_classes
    elif type_ == 'reagent':
        condition = reagent_classes
    
    p = preds.clone()
    p = p.cpu().detach().numpy()
    candidates = []
    for i in range(p.shape[0]):
        if p[i] > cutoff:
            candidates.append((i, condition[i][0], p[i]))
            p[i] = 1
        else:
            p[i] = 0
    return p, candidates


def one_hot(solvent: str, reagent: str):
    """Convert the gold answer solvent ans reagent names to the one-hot feature vector. 
    :param solvent: solvent string.
    :param reagent: reagent string.
    """
    global solvent_classes
    global reagent_classes
    solvent = solvent.split('; ')
    vec_solv = np.array([float(x[0] in solvent) for x in solvent_classes])
    reagent = reagent.split('; ')
    vec_reag = np.array([float(x[0] in reagent) for x in reagent_classes])
    
    return np.concatenate((vec_solv, vec_reag), 0)
    
def get_multilabel_answer(rxn):
    """Get the condition gold answer in the dataset """
    context = rxn[3]
    context = list(zip(*context))
    reagent = context[1]
    solvent = context[2]
    solvent_ = []
    reagent_ = []
    for s in reagent: reagent_ += s.split('; ')
    for s in solvent: solvent_ += s.split('; ')
    reagent = list(dict.fromkeys(reagent_))
    solvent = list(dict.fromkeys(solvent_))
    return solvent, reagent

def create_fake_data(rxn, model, num_fake, cutoff_reag=0.1, cutoff_solv=0.1, fpsize=4096, radius=2):
    """Select multilabel predictions to make fake data. If no prediction, we random select other condition data. """
    global solvent_classes
    global reagent_classes
    
    gold_solv, gold_reag = get_multilabel_answer(rxn)
    rsmi = rxn[1]; psmi = rxn[2];
    rxn_fp = create_rxn_Morgan2FP_concatenate(rsmi, psmi, fpsize=fpsize, radius=radius)
    rxn_fp = torch.Tensor(rxn_fp)
    preds_solv, preds_reag = model(rxn_fp)
    # preds_solv = model_solvent(rxn_fp)
    # preds_reag = model_reagent(rxn_fp)

    p1, cand_solv = create_cutoff(preds_solv, 'solvent', cutoff_solv)
    p2, cand_reag = create_cutoff(preds_reag, 'reagent', cutoff_reag)
    
    false_reag = list(filter(lambda x: x[1] not in  gold_reag , cand_reag))
    false_solv = list(filter(lambda x: x[1] not in  gold_solv , cand_solv))
    
    if false_reag == []:
        false_reag = list(zip(*reagent_classes))[0]
        false_reag = list(filter(lambda x: x not in  gold_reag , false_reag))
    else:
        false_reag = list(zip(*false_reag))[1]
    
    if false_solv == []:
        false_solv = list(zip(*solvent_classes))[0]
        false_solv = list(filter(lambda x: x not in  gold_solv , false_solv))
    else:
        false_solv = list(zip(*false_solv))[1]

    
    fake_reag_list = list(combinations(false_reag, 1)) + list(combinations(false_reag, 2))
    fake_solv_list = list(combinations(false_solv, 1)) + list(combinations(false_solv, 2))
    
    if len(fake_solv_list) < num_fake:
        fake_solv = []
        idxs = np.random.randint(0, len(fake_solv_list), size=num_fake)
        for i in idxs: fake_solv.append(fake_solv_list[i])
    else:
        fake_solv = random.sample(fake_solv_list, num_fake)
        
    if len(fake_reag_list) < num_fake:
        fake_reag = []
        idxs = np.random.randint(0, len(fake_reag_list), size=num_fake)
        for i in idxs: fake_reag.append(fake_reag_list[i])
    else:
        fake_reag = random.sample(fake_reag_list, num_fake)
    
    fake_data = []
    for i in range(num_fake):
        f = (0. ,'; '.join(fake_reag[i]), '; '.join(fake_solv[i]))# data record: first reagent, and then solvent
        fake_data.append(f)
        
    return fake_data
    
if __name__ == '__main__':
    from rxn_yield_context.train_multilabel.model_utils import Multitask_Multilabel
    """
    Get args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_fold', type=int, default=6)
    parser.add_argument('--cutoff_reag', type=float, default=0.20)
    parser.add_argument('--cutoff_solv', type=float, default=0.15)
    parser.add_argument('--model_path', type=str, 
                        default='../save_model/MultiTask_test_hyp_6/multitask_model_epoch-38.checkpoint')
    parser.add_argument('--train_path', type=str, 
                        default='../All_LCC_Data/processed_data/05Final_for_second_part_model/Splitted_train_labels_processed.txt')
    parser.add_argument('--solvent_path', type=str, 
                        default='../All_LCC_Data/processed_data/05Final_for_second_part_model/class_names_solvent_labels_processed.pkl')
    parser.add_argument('--reagent_path', type=str, 
                        default='../All_LCC_Data/processed_data/05Final_for_second_part_model/class_names_reagent_labels_processed.pkl')    
    args = parser.parse_args()
    
    
    """
    Get the reaction data, the solvent classes and the reagent classes.
    """

    solvent_classes = get_classes(args.solvent_path)
    reagent_classes = get_classes(args.reagent_path)
    
    f = open(args.train_path)
    data = f.readlines()
    f.close()
    
    data_sorted = sort_out_data(data)
    
    """
    Call the multi-task multi-label predictive models.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    state = torch.load(args.model_path, map_location = device)
    args1 = TrainArgs_rxn()
    args1.from_dict(vars(state['args']), skip_unsettable=True)
    args1.device = device
    loaded_state_dict = state['state_dict']
    model = Multitask_Multilabel(args1)
    model.load_state_dict(loaded_state_dict)
    model.eval()
    
    
    """
    Test, input molecules
    """
    # rsmi = '[Li]C1=CC=CC=C1.CSC(=S=O)C1CCCCC1'
    # psmi = 'CS[C@H](C1CCCCC1)[S@](=O)C1=CC=CC=C1'
    # rsmi = 'CN(C)C1=CC=C(C=C1)C1=CC2=C(OC1=O)C=C(OC(C)=O)C=C2'
    # psmi = 'CN(C)C1=CC=C(C=C1)C1=CC2=C(OC1=O)C=C(O)C=C2'
    # rxn_fp = create_rxn_Morgan2FP_concatenate(rsmi, psmi, fpsize=args1.fpsize, radius=args1.radius)
    # rxn_fp = torch.Tensor(rxn_fp) # one-dimensional vector
    # # rxn_fp = rxn_fp.view(-1,32768)
    
    # preds_solv, preds_reag = model(rxn_fp)
    
    rxn = data_sorted[1]
    num_fake = len(rxn[3])*args.num_fold
    fake_data = create_fake_data(rxn, model, num_fake=num_fake, 
                             cutoff_reag=args.cutoff_reag, cutoff_solv=args.cutoff_solv, 
                             fpsize=args1.fpsize, radius=args1.radius)
    
    # p1, cand_solv = create_cutoff(preds_solv, 'solvent', 0.4)
    # p2, cand_reag = create_cutoff(preds_reag, 'reagent', 0.4)
    
    """
    Start creating fake data using training data and outputing augmented data.
    """
    # f = open(args.train_path + '.{}fold_newcutoff'.format(args.num_fold) + '.txt', 'w')
    
    # for i in tqdm(range(len(data_sorted))):
    #     rxn = data_sorted[i]
    #     line = rxn[0] + '\t' + rxn[1] + '\t' + rxn[2] + '\t'
    #     num_fake = len(rxn[3])
    #     if num_fake <= 10: num_fake *= args.num_fold # if num_fake <= 10, we make more 4 times fake data
    #     fake_data = create_fake_data(rxn, model, num_fake=num_fake, 
    #                                  cutoff_reag=args.cutoff_reag, cutoff_solv=args.cutoff_solv, 
    #                                  fpsize=args1.fpsize, radius=args1.radius)
    #     data_sorted[i][3] += fake_data
    #     for context in data_sorted[i][3]:
    #         # print(line + str(context[0]) + '\t' + context[1] + '\t' + context[2])
    #         f.write(line + str(context[0]) + '\t' + context[1] + '\t' + context[2] + '\n')
            
    # f.close()

    




