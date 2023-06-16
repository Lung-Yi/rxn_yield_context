# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:27:15 2020

@author: Lung-Yi
"""
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem,DataStructs


def get_classes(path):
    f = open(path, 'rb')
    dict_ = pickle.load(f)
    f.close()
    classes = sorted(dict_.items(), key=lambda d: d[1],reverse=True)
    classes = [(x,y) for x,y in classes]
    return classes

def create_rxn_Morgan2FP_concatenate(rsmi, psmi, rxnfpsize=16384, pfpsize=16384, useFeatures=False, calculate_rfp=True, useChirality=True):
    # Similar as the above function but takes smiles separately and returns pfp and rfp separately

    rsmi = rsmi.encode('utf-8')
    psmi = psmi.encode('utf-8')
    try:
        mol = Chem.MolFromSmiles(rsmi)
    except Exception as e:
        print(e)
        return
    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(
            mol=mol, radius=2, nBits=rxnfpsize, useFeatures=useFeatures, useChirality=useChirality)
        fp = np.empty(rxnfpsize, dtype='float32')
        DataStructs.ConvertToNumpyArray(fp_bit, fp)
    except Exception as e:
        print("Cannot build reactant fp due to {}".format(e))
        return
    rfp = fp

    try:
        mol = Chem.MolFromSmiles(psmi)
    except Exception as e:
        return
    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(
            mol=mol, radius=2, nBits=pfpsize, useFeatures=useFeatures, useChirality=useChirality)
        fp = np.empty(pfpsize, dtype='float32')
        DataStructs.ConvertToNumpyArray(fp_bit, fp)
    except Exception as e:
        print("Cannot build product fp due to {}".format(e))
        return
    pfp = fp
    rxn_fp = pfp - rfp
    final_fp = np.concatenate((pfp, rxn_fp))
    return final_fp