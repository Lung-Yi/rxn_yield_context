# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:59:51 2021

@author: Lung-Yi

This file evaluate the listwise ranking models performance.
"""

import os
import torch
import argparse
import numpy as np
from collections import Counter
from prettytable import PrettyTable
# import torch.nn.functional as F

from rxn_yield_context.train_multilabel.data_utils import get_classes, create_rxn_Morgan2FP_concatenate
# from rxn_yield_context.train_multilabel.model_utils import ReactionModel_Listwise
from rxn_yield_context.train_multilabel.args_train import TrainArgs_rxn
from rxn_yield_context.evaluate_model.eval_utils import get_answer, evaluate_overall, table_for_contexts, compare_all_answers
from rxn_yield_context.evaluate_model.eval_utils import MultiTask_Evaluator, Ranking_Evaluator, MetricsCalculator
from rxn_yield_context.preprocess_data import sort_out_data

def accuracy_within_range(temp_diffs, span):
    acc = [1 if (np.abs(t) <= span) else 0 for t in temp_diffs ]
    return sum(acc)/len(acc)

def in_topk(rank, topk):
    return True if rank <= topk else False

def update_hr_dict(hr_dict, ranks, topk=20):
    gold_length = len(ranks)
    hit = [r if r else float('inf') for r in list(zip(*ranks))[0]]
    if hr_dict.get(gold_length) == None:
        hr_dict[gold_length] = []
        update_hr_dict(hr_dict, ranks, topk)
    else:
        hit_length = len([r for r in hit if r <= topk])
        hr_dict[gold_length].append(hit_length)
    return hr_dict

def evaluate_hr_dict(hr_dict, max_show = 5):
    my_table = PrettyTable()
    my_table.field_names = ['number of contexts \ hit predictions'] + list(range(1, max_show+1))
    for key, value in sorted(hr_dict.items(), key=lambda x: x[0]):
        if key > max_show:
            break
        Denominator = len(value)
        results = dict(Counter(value))
        Numerator = sum(results.values())
        row = [str(key) + ' (' + str(Denominator)+')']
        for hit in list(range(1, max_show+1)):
            if hit <= key:
                try:
                    Numerator -= results[hit - 1]
                except:
                    Numerator -= 0
                text = '{:.2f}%'.format(Numerator / Denominator*100)
            else:
                text = '-'
            row.append(text)
        my_table.add_row(row)
    print(my_table)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir',type=str,
                        default='../All_LCC_Data/processed_data_temp')
    parser.add_argument('--multitask_model',type=str,
                        default='../save_model/MultiTask_temp_1/multitask_model_epoch-35.checkpoint')
    parser.add_argument('--listwise_model',type=str,
                        default='../save_model/MT_rank_temp_1/rxn_model_relevance_listwise_morgan_epoch-80.checkpoint')
    parser.add_argument('--cutoff_solvent',type=float,default=0.25)
    parser.add_argument('--cutoff_reagent',type=float,default=0.3)
    parser.add_argument('--verbose', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes'])) # Whether to print failed prediction information
    args_r = parser.parse_args()
    
    """
    Get the reaction data, the solvent classes and the reagent classes.
    """

    solvent_classes = get_classes(os.path.join(os.path.join(args_r.test_dir, 'label_processed'), 'class_names_solvent_labels_processed.pkl'))
    reagent_classes = get_classes(os.path.join(os.path.join(args_r.test_dir, 'label_processed'), 'class_names_reagent_labels_processed.pkl'))
    test_data = os.path.join(os.path.join(args_r.test_dir, 'For_second_part_model'), 'Splitted_second_test_labels_processed.txt')
    f = open(test_data, 'r')
    data = f.readlines()
    f.close()
    data = sort_out_data(data)

    
    """
    Call the multitask evaluator.
    """
    MT_Evaluator = MultiTask_Evaluator(solvent_classes, reagent_classes, cutoff_solv= args_r.cutoff_solvent, cutoff_reag = args_r.cutoff_reagent)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    MT_Evaluator.load_model(args_r.multitask_model, device = device)

    """
    Call the listwise ranking evaluator.
    """
    RK_Evaluator = Ranking_Evaluator(solvent_classes, reagent_classes)
    RK_Evaluator.load_model(args_r.listwise_model, device)
    
    
    """
    Start to evaluate the ranking of the different context combinations. 
    """
    metrics = MetricsCalculator()
    ranks_list = []
    acc_list = []
    fail_rank = []
    fail_multilabel = []
    mAP_list = []
    NDCG_list = []
    multiple_rxn_mAP = []
    multiple_rxn_NDCG = []
    temp_hit_diffs = []
    hit_ratio_dict = dict() # For evaluation of how many entries in the data can be predicted
    
    for i, rxn in enumerate(data[:]):

        rxn_fp = torch.Tensor(create_rxn_Morgan2FP_concatenate(rxn[1], rxn[2], fpsize = MT_Evaluator.args_MT.fpsize, radius = MT_Evaluator.args_MT.radius))
        rxn_fp = rxn_fp.to(device)
        """ Multitask predictions for solvent and reagent selection """
        input_solvents, input_reagents = MT_Evaluator.make_input_rxn_condition(rxn_fp)

        top_contexts = RK_Evaluator.rank_top_contexts(rxn_fp, input_solvents, input_reagents) # TODO : check
 
        gold_answers = get_answer(rxn)
        ranks, temp_hit_diff = compare_all_answers(gold_answers, top_contexts)
        temp_hit_diffs += temp_hit_diff
        ranks_list.append(ranks)
        mAP = metrics.mAP(ranks)
        mAP_list.append(mAP)
        NDCG = metrics.ndcg(ranks)
        NDCG_list.append(NDCG)
        hit_ratio_dict = update_hr_dict(hit_ratio_dict, ranks)
        
        if len(gold_answers) >= 2:
            multiple_rxn_mAP.append(mAP)
            multiple_rxn_NDCG.append(NDCG)
        
        id_ = min([r if r else float('inf') for r in list(zip(*ranks))[0]])
        if id_ == float('inf'):
            fail_multilabel.append(i)
            if args_r.verbose:
                print('Multilabel Failed: ',i)
                reaction_smiles = rxn[1]+'>>'+rxn[2]
                print(reaction_smiles)
                table_for_contexts(rxn, top_contexts[:20])
        elif id_ > 20:
            fail_rank.append(i)
            if args_r.verbose:
                print('Ranking Failed: ',i)
                reaction_smiles = rxn[1]+'>>'+rxn[2]
                print(reaction_smiles)
                table_for_contexts(rxn, top_contexts[:20])
        else:
            if args_r.verbose:
                print('Hit: ',i)
                reaction_smiles = rxn[1]+'>>'+rxn[2]
                print(reaction_smiles)
                table_for_contexts(rxn, top_contexts[:20])
        acc_list.append(id_)
    
    evaluate_overall(acc_list)
    RMSE_temp = np.sqrt(np.average(np.square(temp_hit_diffs)))
    MAE_temp = np.average(np.abs(temp_hit_diffs))
    print('-'*50)
    print('solvent cutoff: {}'.format(args_r.cutoff_solvent))
    print('reagent cutoff: {}'.format(args_r.cutoff_reagent))
    print('-'*50)
    print('For all {} reactions:'.format(len(ranks_list)))
    print('Prediction Failure because of multi-label selection accounts for {:.2f}%'.format(100*len(fail_multilabel)/len(data)))
    print('Prediction Failure because of ranking accounts for {:.2f}%'.format(100*len(fail_rank)/len(data)))
    print('Mean Average Precision: {:.4f}'.format(np.average(mAP_list)))
    print('Mean Normalized Discounted Cumulative Gain: {:.4f}'.format(np.average(NDCG_list)))
    print('-'*50)
    print('RMSE of temperature prediction: {:.1f}'.format(RMSE_temp))
    print('MAE of temperature prediction: {:.1f}'.format(MAE_temp))
    span1 = 10
    print(r'Temperature predictions fall within +-{} Celsius accuracy: {:.1f}%'.format(span1, 100*accuracy_within_range(temp_hit_diffs, span1)))
    span2 = 20
    print(r'Temperature predictions fall within +-{} Celsius accuracy: {:.1f}%'.format(span2, 100*accuracy_within_range(temp_hit_diffs, span2)))
    print('-'*50)
    print('For {} reactions that have multiple reaction conditions: '.format(len(multiple_rxn_NDCG)))
    print('Mean Average Precision: {:.4f}'.format(np.average(multiple_rxn_mAP)))
    print('Mean Normalized Discounted Cumulative Gain: {:.4f}'.format(np.average(multiple_rxn_NDCG)))
    print('-'*50)
    print('Given top {} predictions'.format(20))
    evaluate_hr_dict(hit_ratio_dict)
    
    