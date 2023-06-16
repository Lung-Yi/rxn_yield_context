# -*- coding: utf-8 -*-

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir',type=str,
                        default='../All_LCC_Data/processed_data_12/label_processed')
    parser.add_argument('--criterion_reagent',type=str,
                        default='../../manually_modified_reagent.txt')
    parser.add_argument('--criterion_solvent',type=str,
                        default='../../manually_modified_solvent.txt')
    args = parser.parse_args()
    
    """ Reagent """
    f = open(args.criterion_reagent, 'r')
    r = f.readlines()
    reagent_dict = {line.split(' -> ')[0] : line.split(' -> ')[1].strip('\n') for line in r[:]}
    f.close()
    
    f = open(os.path.join(args.target_dir, 'class_names_reagent_emerge_backup.txt'), 'r')
    reagent_raw = f.readlines()
    f.close()
    
    g = open(os.path.join(args.target_dir, 'class_names_reagent_emerge.txt'), 'w')
    
    for line in reagent_raw:
        name, smiles = line.strip('\n').split('\t')
        if reagent_dict.get(name):
            new_smiles = reagent_dict.get(name)
        else:
            new_smiles = smiles
        g.write(name + '\t' + new_smiles + '\n')
        
    g.close()
    
    """ Solvent """
    f = open(args.criterion_solvent, 'r')
    r = f.readlines()
    solvent_dict = {line.split(' -> ')[0] : line.split(' -> ')[1].strip('\n') for line in r[:]}
    f.close()
    
    f = open(os.path.join(args.target_dir, 'class_names_solvent_emerge_backup.txt'), 'r')
    solvent_raw = f.readlines()
    f.close()
    
    g = open(os.path.join(args.target_dir, 'class_names_solvent_emerge.txt'), 'w')
    
    for line in solvent_raw:
        name, smiles = line.strip('\n').split('\t')
        if solvent_dict.get(name):
            new_smiles = solvent_dict.get(name)
        else:
            new_smiles = smiles
        g.write(name + '\t' + new_smiles + '\n')
        
    g.close()
    
    