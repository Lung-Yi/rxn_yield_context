# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:50:40 2020

@author: Lung-Yi
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import itertools
from rdkit import Chem
import math

def FilterYield(target, threshold = 95):
    try:
        target = float(target)
    except:
        return True
    if target >= threshold:
        return False
    else:
        return True

def CheckNaN(target):
    try:
        target = float(target)
    except:
        return False
    return math.isnan(target)

def CheckYield(target):
    '''function for having yield report'''
    try:
        target = float(target)
    except:
        return True
    return math.isnan(target)

def RemoveDuplicate(target):
    target = list(set(target.split('; ')))
    return target

def GetFrequencyDict(datas):
    Dict = dict({'nan':0})
    for data in datas:
        if CheckNaN(data):
            Dict['nan'] += 1
            continue
        data = RemoveDuplicate(data)
        for sub_data in data:
            if sub_data in Dict.keys():
                Dict[sub_data] += 1
            else:
                Dict.update({sub_data:1})
    return Dict

def make_hist(ax, x, bins=None, binlabels=None, width=0.85, extra_x=1, extra_y=4, 
              text_offset=0.3, title=r"Frequency diagram", 
              xlabel="Values", ylabel="Frequency"):
    if bins is None:
        xmax = max(x)+extra_x
        bins = range(xmax+1)
    if binlabels is None:
        if np.issubdtype(np.asarray(x).dtype, np.integer):
            binlabels = [str(bins[i]) if bins[i+1]-bins[i] == 1 else 
                         '{}-{}'.format(bins[i], bins[i+1]-1)
                         for i in range(len(bins)-1)]
        else:
            binlabels = [str(bins[i]) if bins[i+1]-bins[i] == 1 else 
                         '{}-{}'.format(*bins[i:i+2])
                         for i in range(len(bins)-1)]
        if bins[-1] == np.inf:
            binlabels[-1] = '{}+'.format(bins[-2])
    n, bins = np.histogram(x, bins=bins)
    patches = ax.bar(range(len(n)), n, align='center', width=width)
    ymax = max(n)+extra_y

    ax.set_xticks(range(len(binlabels)))
    ax.set_xticklabels(binlabels)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis='y')
    # http://stackoverflow.com/a/28720127/190597 (peeol)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # http://stackoverflow.com/a/11417222/190597 (gcalmettes)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    autolabel(patches, text_offset)

def autolabel(rects, shift=0.3):
    """
    http://matplotlib.org/1.2.1/examples/pylab_examples/barchart_demo.html
    """
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            plt.text(rect.get_x()+rect.get_width()/2., height+shift, '%d'%int(height),
                     ha='center', va='bottom')

def plot_frequency(Dict:dict, title_name = 'Frequency Plot', save_path= None):
    x = list(Dict.values())
    fig, ax = plt.subplots(figsize=(14,5), dpi = 800)
    # make_hist(ax, x)
    # make_hist(ax, [1,1,1,0,0,0], extra_y=1, text_offset=0.1)
    make_hist(ax, x, bins=list(range(0,10,2))+list(range(10,1010,200))+[np.inf], extra_y=6, title= title_name)
    # plt.show()
    if save_path is not None:
        fig.savefig(save_path)

def remove_invalid_smiles(data):
    rxn_id = list(data['Reaction ID'])
    start_ = 0
    result = []
    for i in range(len(rxn_id)):
        if rxn_id[i] != rxn_id[start_]:
            end_ = i
            result.append((start_, end_))
            start_ = end_
    rm_pair = []
    for a, b in result:
        if (Chem.MolFromSmiles(str(data['products'][a])) == None) or (Chem.MolFromSmiles(str(data['reactants'][a])) == None):
            rm_pair.append((a,b))
    if rm_pair:        
        rm_pair = list(itertools.chain.from_iterable([list(range(a,b)) for a,b in rm_pair]))
        data.drop(rm_pair, inplace=True)
        data = data.reset_index(drop=True)


def get_remove_list(Dict:dict, freq:int):
    rm = [key for key, value in Dict.items() if value < freq ]
    return rm

def check_rm(reag:str, rm:list):
    if str(reag) == 'nan':
        return False
    reag = reag.split('; ')
    for x in reag:
        if x in rm:
            return True
    return False

def train_validate_test_split_for_Reaxys_condition(data, train_percent=0.8, validate_percent=0.1, SEED=45):
    '''We have to split the data according to its Reaxys ID. '''
    rxn_id = list(data['Reaction ID'])
    start_ = 0
    result = []
    for i in range(len(rxn_id)):
        if rxn_id[i] != rxn_id[start_]:
            end_ = i
            result.append((start_, end_))
            start_ = end_
    result = shuffle(result, random_state=SEED)
    train_index = int(len(result)*train_percent)
    validate_index = int(len(result)*(train_percent + validate_percent))
    train_list = result[0:train_index]
    validate_list = result[train_index:validate_index]
    test_list = result[validate_index:]
    
    train_list = list(itertools.chain.from_iterable([list(range(a,b)) for a,b in train_list]))
    validate_list = list(itertools.chain.from_iterable([list(range(a,b)) for a,b in validate_list]))
    test_list = list(itertools.chain.from_iterable([list(range(a,b)) for a,b in test_list]))
    
    return data.loc[train_list], data.loc[validate_list], data.loc[test_list]

def write_DF2text_second_part(data, output_path):
    f = open(output_path, 'w')
    for i in range(len(data)):
        row = data.iloc[i]
        text = str(row['Reaction ID']+'\t'+row['reactants'])+'\t'+str(row['products'])+'\t'+str(row['Yield (numerical)'])+'\t' \
        +str(row['Reagent'])+'\t'+str(row['Solvent (Reaction Details)'])+'\t'+str(row['Temperature (Reaction Details) [C]'])+'\n'
        f.write(text)
    f.close()

def write_DF2text_first_part(data, output_path):
    rxn_id = list(data['Reaction ID'])
    start_ = 0
    result = []
    for i in range(len(rxn_id)):
        if rxn_id[i] != rxn_id[start_]:
            end_ = i
            result.append((start_, end_))
            start_ = end_
    f = open(output_path, 'w')
    for pair in result:
        data_ = data.loc[list(range(*pair))]
        rxn_id = str(data_['Reaction ID'][pair[0]])
        reactant = str(data_['reactants'][pair[0]])
        product = str(data_['products'][pair[0]])
        solvents = list(data_['Solvent (Reaction Details)'])
        reagents = list(data_['Reagent'])
        solvents = make_classes(solvents)
        reagents = make_classes(reagents)
        text = rxn_id+'\t'+reactant+'\t'+product+'\t'+reagents+'\t'+solvents+'\n'
        f.write(text)
    f.close()

def save_dict2pkl(Dict:dict, output_path):
    f = open(output_path, 'wb')
    pickle.dump(Dict, f)
    f.close()

def make_classes(k:list):
    a = []
    for conds in k:
        conds = str(conds) # consider the float type nan
        for cond in conds.split('; '):
            if cond not in a:
                a.append(cond)
    return '; '.join(a)

def string_average(l_):
    l_ = [float(x) for x in l_]
    return np.average(l_)

def highest_temperature(temp:str):
    if str(temp) == 'nan':
        return temp
    temp = str(temp)
    temps = temp.split('; ')
    temps = [string_average(x.split(' - ')) for x in temps]
    # temp = temp.split(' - ')
    temp = max(temps)
    return temp
    