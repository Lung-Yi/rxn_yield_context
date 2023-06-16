# -*- coding: utf-8 -*-
"""
Created on Sun May 16 14:49:37 2021

label process after finding the chemical names

@author: sun73
"""
import pickle
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir',type=str,
                        default='../All_LCC_Data/processed_data_temp')
    args = parser.parse_args()
    
    """ Solvent Process """
    solvent_emerge_file = os.path.join(os.path.join(args.target_dir, 'label_processed'), 'class_names_solvent_emerge.txt')
    f = open(solvent_emerge_file, 'r')
    emerge_solvent = f.readlines()
    f.close()
    
    solvent_dict_file = os.path.join(os.path.join(args.target_dir, 'unprocessed_class'), 'class_names_solvent.pkl')
    f = open(solvent_dict_file, 'rb')
    freq_dict_solvent = pickle.load(f)
    f.close()
    
    smiles_dict_solvent = dict()
    for line in emerge_solvent:
        name, smiles = line.strip('\n').split('\t')
        if smiles != 'None':
            if smiles_dict_solvent.get(smiles):
                smiles_dict_solvent[smiles].append((name, freq_dict_solvent[name]))
            else:
                smiles_dict_solvent[smiles] = [(name, freq_dict_solvent[name])]
    
    # organize smiles dictionary
    
    for key, value in smiles_dict_solvent.items():
        if len(value) > 1:
            name = sorted(value, key= lambda x:x[1], reverse=True)[0][0]
            count_sum = sum(list(zip(*value))[1])
            smiles_dict_solvent[key] = [(name, count_sum)]
    
    # prepane name dictionary for old_name -> new_name
    # and new canditate dictionary for multi-label
    name_dict_solvent = dict()
    new_freq_dict_solvent = dict()
    for line in emerge_solvent:
        name, smiles = line.strip('\n').split('\t')
        if smiles == 'None':
            name_dict_solvent[name] = name
            new_freq_dict_solvent[name] = freq_dict_solvent[name]
        else:
            new_name = smiles_dict_solvent[smiles][0][0]
            name_dict_solvent[name] = new_name
            if not new_freq_dict_solvent.get(new_name):
                new_freq_dict_solvent[new_name] = smiles_dict_solvent[smiles][0][1]
    
    new_freq_dict_solvent_path = os.path.join(os.path.join(args.target_dir, 'label_processed'), 'class_names_solvent_labels_processed.pkl')
    f = open(new_freq_dict_solvent_path, 'wb')
    pickle.dump(new_freq_dict_solvent, f)
    f.close()
    
    
    
    
    """ Reagent Process """
    reagent_emerge_file = os.path.join(os.path.join(args.target_dir, 'label_processed'), 'class_names_reagent_emerge.txt')
    f = open(reagent_emerge_file, 'r')
    emerge_reagent = f.readlines()
    f.close()
    
    
    reagent_dict_file = os.path.join(os.path.join(args.target_dir, 'unprocessed_class'), 'class_names_reagent.pkl')
    f = open(reagent_dict_file, 'rb')
    freq_dict_reagent = pickle.load(f)
    f.close()
    
    smiles_dict_reagent = dict()
    for line in emerge_reagent:
        name, smiles = line.strip('\n').split('\t')
        if smiles != 'None':
            if smiles_dict_reagent.get(smiles):
                smiles_dict_reagent[smiles].append((name, freq_dict_reagent[name]))
            else:
                smiles_dict_reagent[smiles] = [(name, freq_dict_reagent[name])]
    
    # organize smiles dictionary
    
    for key, value in smiles_dict_reagent.items():
        if len(value) > 1:
            name = sorted(value, key= lambda x:x[1], reverse=True)[0][0]
            count_sum = sum(list(zip(*value))[1])
            smiles_dict_reagent[key] = [(name, count_sum)]
    
    # prepane name dictionary for old_name -> new_name
    # and new canditate dictionary for multi-label
    name_dict_reagent = dict()
    new_freq_dict_reagent = dict()
    for line in emerge_reagent:
        name, smiles = line.strip('\n').split('\t')
        if smiles == 'None':
            name_dict_reagent[name] = name
            new_freq_dict_reagent[name] = freq_dict_reagent[name]
        else:
            new_name = smiles_dict_reagent[smiles][0][0]
            name_dict_reagent[name] = new_name
            if not new_freq_dict_reagent.get(new_name):
                new_freq_dict_reagent[new_name] = smiles_dict_reagent[smiles][0][1]
    
    
    new_freq_dict_reagent_path = os.path.join(os.path.join(args.target_dir, 'label_processed'), 'class_names_reagent_labels_processed.pkl')
    f = open(new_freq_dict_reagent_path, 'wb')
    pickle.dump(new_freq_dict_reagent, f)
    f.close()
    
    # Now process original splitted data
    """ First Part Data """
    first_data_path = os.path.join(args.target_dir, 'For_first_part_model')
    
    files = ['test', 'validate', 'train']
    for file in files:
        f = open(os.path.join(first_data_path,'Splitted_first_{}.txt'.format(file)), 'r')
        data = f.readlines()
        f.close()
        
        for i in range(len(data)):
            line = data[i]
            line = line.strip('\n').split('\t')
            r_ = line[3].split('; ')
            s_ = line[4].split('; ')
            r_ = [name_dict_reagent[name] for name in r_]
            s_ = [name_dict_solvent[name] for name in s_]
            line[3] = '; '.join(r_)
            line[4] = '; '.join(s_)
            data[i] = '\t'.join(line) + '\n'
        
        f = open(os.path.join(first_data_path, 'Splitted_first_{}_labels_processed.txt'.format(file)), 'w')
        f.writelines(data)
        f.close()
    
    """ Second Part Data """
    second_data_path = os.path.join(args.target_dir, 'For_second_part_model')
    
    files = ['test', 'validate', 'train']
    for file in files:
        f = open(os.path.join(second_data_path,'Splitted_second_{}.txt'.format(file)), 'r')
        data = f.readlines()
        f.close()
        
        for i in range(len(data)):
            line = data[i]
            line = line.strip('\n').split('\t')
            r_ = line[4].split('; ')
            s_ = line[5].split('; ')
            r_ = [name_dict_reagent[name] for name in r_]
            s_ = [name_dict_solvent[name] for name in s_]
            line[4] = '; '.join(r_)
            line[5] = '; '.join(s_)
            data[i] = '\t'.join(line) + '\n'
            
        f = open(os.path.join(second_data_path, 'Splitted_second_{}_labels_processed.txt'.format(file)), 'w')
        f.writelines(data)
        f.close()