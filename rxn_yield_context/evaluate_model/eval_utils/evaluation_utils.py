# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:27:15 2020

@author: Lung-Yi
"""
import torch
import torch.nn.functional as F
import numpy as np
import random
from prettytable import PrettyTable
from itertools import combinations
from rxn_yield_context.train_multilabel.args_train import TrainArgs_rxn
from rxn_yield_context.train_multilabel.model_utils import ReactionModel_LWTemp, Multitask_Multilabel
from rxn_yield_context.train_multilabel.data_utils import Yield2Relevance, create_rxn_Morgan2FP_concatenate, get_classes

from tap import Tap
import pandas as pd
import os

class ReactionContextPredictor:
    def __init__(self, data_path, candidate_generation_model_path, ranking_model_path, cutoff_solv=0.3, cutoff_reag=0.3, device=None, verbose=True):
        self.verbose = verbose
        self.solvent_classes = get_classes(os.path.join(os.path.join(data_path, 'label_processed'), 'class_names_solvent_labels_processed.pkl'))
        self.reagent_classes = get_classes(os.path.join(os.path.join(data_path, 'label_processed'), 'class_names_reagent_labels_processed.pkl'))

        self.cutoff_solv = cutoff_solv
        self.cutoff_reag = cutoff_reag
        if device == None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.load_candidate_generation_model(candidate_generation_model_path)
        self.load_ranking_model(ranking_model_path)

    def load_candidate_generation_model(self, candidate_generation_model_path):
        MT_Evaluator = MultiTask_Evaluator(self.solvent_classes, self.reagent_classes, cutoff_solv= self.cutoff_solv, cutoff_reag = self.cutoff_reag)
        MT_Evaluator.load_model(candidate_generation_model_path, device=self.device)
        self.candidate_generation_model = MT_Evaluator
        self.fpsize = self.candidate_generation_model.args_MT.fpsize
        self.radius = self.candidate_generation_model.args_MT.radius
        return
    
    def load_ranking_model(self, ranking_model_path):
        RK_Evaluator = Ranking_Evaluator(self.solvent_classes, self.reagent_classes)
        RK_Evaluator.load_model(ranking_model_path, device=self.device)
        self.ranking_model = RK_Evaluator
        return
    
    def recommend_reaction_context(self, rxn_smiles_list: list, max_display: int=20) -> pd.DataFrame:
        separate_rxn_smiles_list = [rxn_smiles.split('>>') for rxn_smiles in rxn_smiles_list]
        # reactant_list, product_list = list(zip(*separate_rxn_smiles_list))
        rxn_fps = [torch.Tensor(create_rxn_Morgan2FP_concatenate(reac, prod, fpsize = self.fpsize, radius = self.radius)) for reac, prod in separate_rxn_smiles_list]
        results = []
        for rxn_smiles, rxn_fp in zip(rxn_smiles_list, rxn_fps):
            input_solvents, input_reagents = self.candidate_generation_model.make_input_rxn_condition(rxn_fp)
            top_contexts = self.ranking_model.rank_top_contexts(rxn_fp, input_solvents, input_reagents)
            pass
            result_data = self.table_for_one_reaction_contexts(rxn_smiles, top_contexts[:max_display])
            results.append(result_data)
        return results
        
    def table_for_one_reaction_contexts(self, rxn_smiles, top_contexts):
        my_table = PrettyTable()
        my_table.field_names = ['Yield / Rank', 'Reagent(s)', 'Solvent(s)', 'Temperature', 'Probability']
        predictions = [ [ 'Rank{}'.format(i+1), top_contexts[i][1], top_contexts[i][0], '{:.1f}'.format(top_contexts[i][2]), '{:.3f}'.format(top_contexts[i][3]) ] for i in range(len(top_contexts))]
        [my_table.add_row(row) for row in predictions]
        if self.verbose:
            print(rxn_smiles)
            print(my_table)
            print('\n')
        return pd.DataFrame(predictions, columns=my_table.field_names)


class MultiTask_Evaluator:
    """
    Used to evaluate the multi-label predictoins and to give the potential combinatoins for ranking model.
    """
    def __init__(self, solvent_classes, reagent_classes, cutoff_solv = 0.3, cutoff_reag = 0.3):
        self.solvent_classes = solvent_classes
        self.reagent_classes = reagent_classes
        self.model = None
        
        self.exc_solv_names = ['neat (no solvent)','not given', 'neat (no solvent, solid phase)']
        self.exc_reag_names = ['nan']
        self.get_exclusive_label_index()
        
        self.cutoff_solv = cutoff_solv
        self.cutoff_reag = cutoff_reag
        self.max_solv = 11
        self.max_reag = 11
        
        
    def load_model(self, model_path, device = 
                   torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        
        state_MT = torch.load(model_path, map_location= device)
        self.args_MT = TrainArgs_rxn()
        self.args_MT.from_dict(vars(state_MT['args']), skip_unsettable=True)
        self.args_MT.device = device
        
        loaded_state_dict_MT = state_MT['state_dict']
        self.model = Multitask_Multilabel(self.args_MT)
        self.model.to(device)
        self.model.load_state_dict(loaded_state_dict_MT)
        self.model.eval()
        
    def get_exclusive_label_index(self):
        """This function gets the index of exclusive condition such as (1)neat (no solvent), we won't use them 
        to enumerate combinations. """
        self.EXC_SOLV_INDEX = []
        self.EXC_REAG_INDEX = []
        for i in range(len(self.solvent_classes)):
            if self.solvent_classes[i][0] in self.exc_solv_names: self.EXC_SOLV_INDEX.append(i)
        for i in range(len(self.reagent_classes)):
            if self.reagent_classes[i][0] in self.exc_reag_names: self.EXC_REAG_INDEX.append(i)
            
    def print_evaluator_info(self):
        pass
    
    def reset_cutoff(self, cutoff_solv, cutoff_reag):
        self.cutoff_solv = cutoff_solv
        self.cutoff_reag = cutoff_reag
        
    def one_hot(self, context):
        """Convert the solvent and reagent names to the one-hot feature vector. """
        solvent, reagent = context
        solvent = solvent.split('; ')
        vec_solv = np.array([float(x[0] in solvent) for x in self.solvent_classes])
        reagent = reagent.split('; ')
        vec_reag = np.array([float(x[0] in reagent) for x in self.reagent_classes])
        return vec_solv, vec_reag
    
    def convert_contexts2tensor(self, contexts):
        solvent_batch = []
        reagent_batch = []
        for context in contexts:
            vec_solv, vec_reag = self.one_hot(context)
            solvent_batch.append(vec_solv)
            reagent_batch.append(vec_reag)        
        return torch.Tensor(solvent_batch), torch.Tensor(reagent_batch)    
    
    def create_cutoff(self, preds, class_name):
        """ Process one-dimensoin multilabel answer: """
        p = preds.clone()
        p = p.cpu().detach().numpy()
        # candidates = []
        if class_name == 'solvent':
            hard_selection_index = np.where(p > self.cutoff_solv)[0]
            if len(hard_selection_index) < 1:
                hard_selection_index = np.append(hard_selection_index, p.argsort()[-2:])
            
            new_p = np.zeros(len(self.solvent_classes))
            new_p[hard_selection_index] = 1
            candidates = [(i, self.solvent_classes[i][0], p[i]) for i in range(len(new_p)) if (new_p[i] == 1)]
            # for i in range(p.shape[0]):
            #     if p[i] > self.cutoff_solv:
            #         candidates.append((i, self.solvent_classes[i][0], p[i]))
            #         p[i] = 1
            #     else:
            #         p[i] = 0
        elif class_name == 'reagent':
            hard_selection_index = np.where(p > self.cutoff_reag)[0]
            if len(hard_selection_index) < 1:
                hard_selection_index = np.append(hard_selection_index, p.argsort()[-2:])
            
            new_p = np.zeros(len(self.reagent_classes))
            new_p[hard_selection_index] = 1
            candidates = [(i, self.reagent_classes[i][0], p[i]) for i in range(len(new_p)) if (new_p[i] == 1)]
            # for i in range(p.shape[0]):
            #     if p[i] > self.cutoff_reag:
            #         candidates.append((i, self.reagent_classes[i][0], p[i]))
            #         p[i] = 1
            #     else:
            #         p[i] = 0
        return new_p, candidates
    
    def enumerate_combinations(self, rxn_fp):
        preds_solv, preds_reag = self.model(rxn_fp)
        p1, candidates_solv = self.create_cutoff(preds_solv, class_name = 'solvent')
        p2, candidates_reag = self.create_cutoff(preds_reag, class_name = 'reagent')
        if len(candidates_solv) > self.max_solv:
            p1 = truncate_features(p1, candidates_solv, self.max_solv)
        if len(candidates_reag) > self.max_reag:
            p2 = truncate_features(p2, candidates_reag, self.max_reag)
        p1 = list(np.where(p1 == 1)[0])
        p2 = list(np.where(p2 == 1)[0])
        p1 = list(combinations([i for i in p1 if i not in self.EXC_SOLV_INDEX], 2)) + list(combinations(p1, 1))
        # list(combinations([i for i in p1 if i not in self.EXC_SOLV_INDEX], 3)) + \
        p2 = list(combinations([i for i in p2 if i not in self.EXC_REAG_INDEX], 3)) + \
             list(combinations([i for i in p2 if i not in self.EXC_REAG_INDEX], 2)) + list(combinations(p2, 1))
        
        enumerated_features = []
        for index_solv in p1:
            feature_solv = np.zeros(len(self.solvent_classes))
            for i in index_solv: feature_solv[i] = 1
            
            for index_reag in p2:
                feature_reag = np.zeros(len(self.reagent_classes))
                for i in index_reag: feature_reag[i] = 1
                enumerated_features.append((feature_solv, feature_reag))
        if enumerated_features == []:
            enumerated_features = [(np.zeros(len(self.solvent_classes)), np.zeros(len(self.reagent_classes)))]
        return enumerated_features

    def make_input_rxn_condition(self, rxn_fp):
        """ Enumerate all possible reaction conditions for the inputs of ranking model. """
        rxn_fp = rxn_fp.to(self.model.device)
        enumerated_features = self.enumerate_combinations(rxn_fp)
        enumerated_solvent, enumerated_reagent = list(zip(*enumerated_features))
        return torch.Tensor(enumerated_solvent), torch.Tensor(enumerated_reagent)
    
    def predict_context(self, rxn_fp, verbose = False):
        """ Given the reaction fingerprint, translate the predicted outputs to names. """
        rxn_fp = rxn_fp.to(self.model.device)
        preds_solv, preds_reag = self.model(rxn_fp)
        p1, candidates_solv = self.create_cutoff(preds_solv, class_name = 'solvent')
        p2, candidates_reag = self.create_cutoff(preds_reag, class_name = 'reagent')
        if verbose: # print the table of potential selections
            solv_table = PrettyTable()
            solv_table.field_names = ['Index', 'Solvent Name', 'Probabilty']
            [solv_table.add_row(row) for row in candidates_solv]
            print(solv_table)
            
            reag_table = PrettyTable()
            reag_table.field_names = ['Index', 'Reagent Name', 'Probabilty']
            [reag_table.add_row(row) for row in candidates_reag]
            print(reag_table)
        
        return candidates_solv, candidates_reag
    
    def make_input_rxn_conditionBYnames(self, rxn_fp):
        candidates_solv, candidates_reag = self.predict_context(rxn_fp, verbose = False)
        candidates_solv = truncate_candidates(candidates_solv, self.max_solv)
        candidates_reag = truncate_candidates(candidates_reag, self.max_reag)
        
        candidates_solv = [x[1] for x in candidates_solv]
        candidates_reag = [x[1] for x in candidates_reag]
        candidates_solv = list(combinations([x for x in candidates_solv if x not in self.exc_solv_names], 3)) + \
                          list(combinations([x for x in candidates_solv if x not in self.exc_solv_names], 2)) + list(combinations(candidates_solv, 1))
        candidates_reag = list(combinations([x for x in candidates_reag if x not in self.exc_reag_names], 3)) + \
                          list(combinations([x for x in candidates_reag if x not in self.exc_reag_names], 2)) + list(combinations(candidates_reag, 1))
        
        enumerated_contexts = []
        for solv in candidates_solv:
            for reag in candidates_reag:
                enumerated_contexts.append(('; '.join(solv), '; '.join(reag)))
        return enumerated_contexts
        
    def create_fake_data(self, rxn_fp, gold_solvent_index, gold_reagent_index, num_fake):
        """
        rxn_fp : one-dimension tensor
        gold_solvent_idnex : set() with gold index in it.
        gold_reagent_index : set() with golf index in it.
        num_fake : indicates how many fake data should be generated.
        """
        # _, reagents, solvents  = list(zip(*gold_contexts))
        # gold_reagent_pool = set(); gold_solvent_pool = set();
        # [gold_reagent_pool.update(x.split('; ')) for x in reagents]
        # [gold_solvent_pool.update(x.split('; ')) for x in solvents]
        
        candidates_solv, candidates_reag = self.predict_context(rxn_fp.squeeze(0), verbose = False)
        
        false_reag = list(filter(lambda x: x[0] not in  gold_reagent_index , candidates_reag))
        false_solv = list(filter(lambda x: x[0] not in  gold_solvent_index , candidates_solv))
        
        if false_reag == []:
            false_reag = [i for i in range(len(self.reagent_classes))]
            false_reag = list(filter(lambda x: x not in  gold_reagent_index , false_reag))
        else:
            false_reag = list(zip(*false_reag))[0]
        
        if false_solv == []:
            false_solv = [i for i in range(len(self.solvent_classes))]
            false_solv = list(filter(lambda x: x not in  gold_solvent_index , false_solv))
        else:
            false_solv = list(zip(*false_solv))[0]        
        
        upper_reag = 3 if (len(false_reag) >= 3) else 2 if (len(false_reag) == 2) else 1
        upper_solv = 2 if (len(false_solv) >= 2) else 1
        # upper_reag = 2 if (len(false_reag) >= 2) else 1
        # upper_solv = 2 if (len(false_solv) >= 2) else 1
        # false_reag, false_solv -> Both are list that contain the false condition index.
        fake_combinations = []
        for i in range(num_fake): # TODO: 改回來
            seed = random.randint(1,20)
            if seed%20 == 0:
                comb = [random.sample(list(gold_solvent_index), 1),
                        random.sample(false_reag, random.randint(1,upper_reag)),
                        1]
            elif seed%20 == 1:
                comb = [random.sample(false_solv, random.randint(1,upper_solv)),
                        random.sample(list(gold_reagent_index), 1),
                        1]
            else:
                comb = [random.sample(false_solv, random.randint(1,upper_solv)),
                        random.sample(false_reag, random.randint(1,upper_reag)),
                        0]
                
            fake_combinations.append(comb)
        
        return fake_combinations # combination [[(solvent index), (reagent index)], ...]
            
############################################################################################################################################################################

class Ranking_Evaluator():
    """ Used to evaluate and rank the different reactoin conditions. """
    def __init__(self, solvent_classes, reagent_classes):
        self.solvent_classes = solvent_classes
        self.reagent_classes = reagent_classes
        self.model = None
        self.device = None
        
    def load_model(self, model_path, device = 
                   torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.device = device
        state_LW = torch.load(model_path, map_location= device)
        self.args_LW = TrainArgs_rxn()
        self.args_LW.from_dict(vars(state_LW['args']), skip_unsettable=True)
        self.args_LW.device = device
        loaded_state_dict_LW = state_LW['state_dict']
        self.model = ReactionModel_LWTemp(self.args_LW, len_solvent=len(self.solvent_classes), len_reagent=len(self.reagent_classes))
        self.model.to(device)
        self.model.load_state_dict(loaded_state_dict_LW)
        self.model.eval()
    
    def convert_features2name(self, inputs, class_name):
        """"Convert binary vector to solvent names and reagent names. """
        input_features = inputs.clone().cpu().detach().numpy()
        context = []
        if class_name == 'solvent':
            classes = self.solvent_classes
        elif class_name == 'reagent':
            classes = self.reagent_classes
            
        for feature in input_features:
            text = ''
            for i in np.where(feature==1)[0]: text += classes[i][0]+'; '
            text = text.rstrip('; ')
            context.append(text)
        return context
    
    def make_contexts(self, input_solvents, input_reagents):
        """ Convert the binary tensor to the reaction condition names. """
        context_solv = self.convert_features2name(input_solvents, 'solvent')
        context_reag = self.convert_features2name(input_reagents, 'reagent')
        
        return list(zip(context_solv, context_reag))
    
    def predict_scores(self, rxn_fp, input_solvents, input_reagents):
        """ Input rxn_fp shape should be [1, N, fp_size*2] per query. """
        input_solvents = input_solvents.unsqueeze(0).to(self.device)
        input_reagents = input_reagents.unsqueeze(0).to(self.device)
        
        rxn_fp = rxn_fp.view(1,-1)
        rxn_fp = rxn_fp.repeat(input_solvents.shape[1], 1)
        rxn_fp = rxn_fp.unsqueeze(0).to(self.device)
        
        scores, temperatures = self.model(rxn_fp, input_solvents, input_reagents)
        scores = F.softmax(scores, dim=1).view(-1)
        temperatures = temperatures.view(-1)
        temperatures = [float(t.cpu().detach().numpy()) for t in temperatures]
        
        return scores, temperatures
    
    def rank_top_contexts(self, rxn_fp, input_solvents, input_reagents):
        contexts = self.make_contexts(input_solvents, input_reagents)
        scores, temperatures = self.predict_scores(rxn_fp, input_solvents, input_reagents)
        top_index = torch.argsort(scores, descending=True)
        top_contexts = [sort_string(contexts[j]) + [temperatures[j]] + [float(scores[j].detach().cpu())] for j in top_index] # TODO : check
        # temperatures = [[temperatures[j]] for j in top_index]
        # [ top_contexts[j].extend(temperatures[j]) for j in range(len(top_contexts)) ]
        
        return top_contexts#, temperatures


############################################################################################################################################################################

class MetricsCalculator():
    """ Include recommendation metrics such as MAP (mean average precision), NDCG (normalized discounted cumulative gain). """
    def __init__(self, topk = 20):
        self.topk = topk
        
    def calculate_dcg(self, scores):
        """ Input: list[relevances] , index-> rank """
        if scores == []:
            return 0.
        return np.sum(
            np.divide(scores, np.log2(np.arange(len(scores), dtype=np.float32) + 2)),
            dtype=np.float32)
    
    def ndcg(self, ranks):
        idcg = list(list(zip(*sorted(ranks, key = lambda x:x[1], reverse=True)))[1])[:self.topk]
        idcg = self.calculate_dcg(idcg)
        dcg_rank = [(rank if rank else float('inf'), relv) for rank, relv in ranks]
        dcg_rank = sorted(dcg_rank, key = lambda x:x[0], reverse=False)
        scores = [0]*self.topk
        for rank, relv in dcg_rank:
            if rank <= self.topk:
                scores[rank-1] = relv
        dcg = self.calculate_dcg(scores)
        return dcg/idcg
    
    def mAP(self, ranks):
        doc = min(len(ranks), self.topk)
        results = [(rank if rank else float('inf'), relv) for rank, relv in ranks]
        results = sorted(results, key = lambda x:x[0], reverse=False)
        scores = [0]*self.topk
        for rank, relv in results:
            if rank <= self.topk: scores[rank-1] = 1
        mAP = 0
        hit = 0
        for j, relv in enumerate(scores):
            if relv != 0:
                hit += 1
                mAP += hit/(j+1)
        return mAP/doc
        
############################################################################################################################################################################

def truncate_features(p, candidates, top=11):
    assert type(top) == int
    candidates = sorted(candidates, key = lambda x:x[2], reverse =True)[0:top]
    p_ = np.zeros(len(p))
    idx,_1, _2 = zip(*candidates)
    np.put(p_, idx, 1.0)
    return p_

def truncate_candidates(candidates, top=11):
    assert type(top) == int
    candidates = sorted(candidates, key = lambda x:x[2], reverse =True)[0:top]
    return candidates



def get_answer(rxn):
    """Get the condition gold answer in the dataset """
    answers = []
    for cond in rxn[3]:
        yield_ = cond[0]; reagent = cond[1]; solvent = cond[2]; temp = cond[3]
        if temp == 'None' : temp = 'None'
        relevance = Yield2Relevance(yield_)
        reagent = set(reagent.split('; '))
        solvent = set(solvent.split('; '))
        answers.append((solvent, reagent, relevance, temp))
    return answers

def get_answer_separate_water(rxn):
    """Get the condition gold answer in the dataset, if solvents have water => remove water, if only water => keep it"""
    answers = []
    for cond in rxn[3]:
        yield_ = cond[0]; reagent = cond[1]; solvent = cond[2]; temp = cond[3]
        if temp == 'None' : temp = 'None'
        relevance = Yield2Relevance(yield_)
        reagent = set(reagent.split('; '))
        solvent = set(solvent.split('; '))
        if ('water' in solvent) and (solvent != {'water'}):
            solvent.remove('water')
        
        answers.append((solvent, reagent, relevance, temp))
    return answers

def compare_answer_and_combinations(answers, context_combinations):
    for answer in answers:
        answer_s, answer_r, y_, t_ = answer
        for i, context in enumerate(context_combinations):
            solvent, reagent = context
            solvent = set(solvent.split('; '))
            reagent = set(reagent.split('; '))
            if (answer_s == solvent) & (answer_r == reagent):
                return i+1
    return None

def compare_all_answers(answers, context_combinations):
    ranks = []
    temp_hit_diff = []
    for answer in answers:
        answer_s, answer_r, relevance, temp_true = answer
        for hit, context in enumerate(context_combinations):
            solvent, reagent, temp_pred, _ = context
            solvent = set(solvent.split('; '))
            reagent = set(reagent.split('; '))
            if (answer_s == solvent) & (answer_r == reagent):
                ranks.append((hit+1, relevance))
                if temp_true == 'nan':
                    continue
                temp_hit_diff.append(float(temp_true) - temp_pred)
                break
        else:
            ranks.append((None, relevance))
    return ranks, temp_hit_diff

def evaluate_overall(acc_list, show=(1,3,5,10,15,20)):
    topk_dict = dict(zip(show,[0]*len(show)))
    length = len(acc_list)
    for rank in acc_list:
        if rank == None: continue
        for key in topk_dict.keys():
            if rank <= key:
                topk_dict[key] += 1
    acc_dict = {}
    for key, value in topk_dict.items():
        print("top accuracy@{} : {:.4f}".format(key, value/length))
        acc_dict.update({"top@{} accuracy".format(key):  value/length})
    return acc_dict

def table_for_contexts(rxn, top_contexts): # TODO: change!!!!
    my_table = PrettyTable()
    my_table.field_names = ['Yield / Rank', 'Reagent(s)', 'Solvent(s)', 'Temperature', 'Probability']
    
    gold_context = rxn[3]
    gold_context = sort_string(gold_context)
    for i in range(len(gold_context)):
        gold_context[i].append('N/A')
    predictions = [ [ 'Rank{}'.format(i+1), top_contexts[i][1], top_contexts[i][0], '{:.1f}'.format(top_contexts[i][2]), '{:.3f}'.format(top_contexts[i][3]) ] for i in range(len(top_contexts))]
    gold_context.extend(predictions)
    
    [my_table.add_row(row) for row in gold_context]
    print(my_table)
    
    
def sort_string(context):
    if type(context) == str:
        return '; '.join(sorted(context.split('; ')))
    else:
        return [sort_string(z) for z in context]

def remove_duplicated_records(records):
    return '; '.join(list(dict.fromkeys(records.split('; '))))



