# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:42:09 2021

@author: sun73
"""
from molvs import standardize_smiles
from rdkit import Chem
from tqdm import tqdm

import os
import argparse
import pubchempy as pcp
import time
import re
from chemspipy import ChemSpider
cs = ChemSpider('2CCdzprbsgWZAUh5N9yhJtsGOYNIoJiz')

def PCPconvert(name):
    s = pcp.get_compounds(name,'name')
    try:
        return s[0].canonical_smiles
    except:
        return 'None'

def ChemSpiderConvert(name):
    results = cs.search(name)
    time.sleep(0.3)
    if len(results) == 0:
        return 'None'
    return results[0].smiles

def SmilesWithoutIsotopesToSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
    for atom, isotope in atom_data:
        if isotope:
            atom.SetIsotope(0)
    smiles = Chem.MolToSmiles(mol)
    for atom, isotope in atom_data:
        if isotope:
            atom.SetIsotope(isotope)
    return smiles

def cation_valence_check(name):
    if '(l' in name:
        start = name.find('(l') + 1
        end = name.find('l)') + 1
        name = list(name)
        for i in range(start, end):
            if name[i] == 'l': name[i] = 'I'
            
        return ''.join(name)
    else:
        return name

def hydrate_check(name):
    words = name.split(' ')
    for word in words:
        if 'hydrate' in word:
            new_words = words.copy()
            new_words.remove(word)
            return ' '.join(new_words)
    return name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',type=str,
                        default='../All_LCC_Data/processed_data_temp/unprocessed_class')
    parser.add_argument('--output_dir',type=str,
                        default='../All_LCC_Data/processed_data_temp/label_processed')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    f = open(os.path.join(args.input_dir, 'class_names_reagent.txt'), 'r')
    names = f.readlines()
    f.close()
    
    g = open(os.path.join(args.input_dir, 'class_names_reagent_smiles.txt'), 'r')
    smiles_ = g.readlines()
    g.close()
    
    reagent_list = []
    z = open(os.path.join(args.output_dir, 'class_names_reagent_emerge_backup.txt'), 'w')
    dup = 0
    for i in tqdm(range(len(names))):
        name_ = names[i].strip('\n')
        name = cation_valence_check(name_)
        name = re.sub('\?', '-', name)
        name = hydrate_check(name)
        smiles = smiles_[i].strip('\n')
        if (smiles == '') or ('hydrate' in name_): # remove "hydrate" in the reagent name
            smiles = PCPconvert(name)
            if (('palladium' in name) and (('carbon' in name) or ('charcoal' in name))) or ('Pd/C' in name):
                smiles = 'C.[Pd]'
            elif smiles == 'None':
                smiles = ChemSpiderConvert(name)
        if smiles != 'None':
            try:
                smiles = SmilesWithoutIsotopesToSmiles(smiles)
                smiles = standardize_smiles(smiles)
            except:
                pass
            if smiles in reagent_list:
                print('\nDuplicate: ')
                print(name, smiles)
                dup += 1
            else:
                reagent_list.append(smiles)
                
        z.write(name_+'\t'+smiles+'\n')
    print('Total Duplicate Reagent Number: ',dup)
    z.close()
    
    f = open(os.path.join(args.input_dir, 'class_names_solvent.txt'), 'r')
    names = f.readlines()
    f.close()
    
    g = open(os.path.join(args.input_dir, 'class_names_solvent_smiles.txt'), 'r')
    smiles_ = g.readlines()
    g.close()
    
    print('--------------------------------------------')
    solvent_list = []
    dup = 0
    z = open(os.path.join(args.output_dir, 'class_names_solvent_emerge_backup.txt'), 'w')
    for i in tqdm(range(len(names))):
        name = names[i].strip('\n')
        name = re.sub('\?', '-', name)
        smiles = smiles_[i].strip('\n')
        if smiles == '': 
            smiles = PCPconvert(name)
            if smiles == 'None':
                smiles = ChemSpiderConvert(name)
        if smiles != 'None':
            try:
                smiles = SmilesWithoutIsotopesToSmiles(smiles)
                smiles = standardize_smiles(smiles)
            except:
                pass
            if smiles in solvent_list:
                print('Duplicate: ')
                print(name, smiles)
                dup += 1
            else:
                solvent_list.append(smiles)
        z.write(name+'\t'+smiles+'\n')
    z.close()
    print('Total Duplicate Solvent Number: ',dup)