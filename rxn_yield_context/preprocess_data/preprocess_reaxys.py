# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:24:17 2020

@author: Lung-Yi

Preprocess the reaxys data (.xlsx) 
"""

import os
import pandas as pd
import argparse

from rxn_yield_context.preprocess_data.preprocess import CheckNaN, CheckYield, GetFrequencyDict, plot_frequency
from rxn_yield_context.preprocess_data.preprocess import remove_invalid_smiles, train_validate_test_split_for_Reaxys_condition, write_DF2text_second_part, write_DF2text_first_part
from rxn_yield_context.preprocess_data.preprocess import save_dict2pkl, highest_temperature

def CheckSemicolon_count(target, count = 2):
    return target.count('; ') >= count

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

def remove_duplicated_records(records):
    return '; '.join(list(dict.fromkeys(records.split('; '))))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',type=str,
                        default='../../Data_From_Reaxys_Original_test')
    parser.add_argument('--output_dir',type=str,
                        default='../All_LCC_Data/processed_data_test')
    args = parser.parse_args()
    
    input_file_path = args.input_dir
    output_file_path = args.output_dir
    os.makedirs(output_file_path, exist_ok=True)
    
    input_files  = []
    for root, dirs, files in os.walk(input_file_path):
        for name in files:
            input_files.append(os.path.join(root, name))
    

    keep_index = ['Reaction ID', 'Reaction', 'Temperature (Reaction Details) [C]', 'Yield (numerical)',
                  'Reagent', 'Catalyst', 'Solvent (Reaction Details)']
    
    
    '''
    Concatenate all the data in the input_file_path directory. 
    '''
    data = []

    
    for input_file in input_files[:]:
        df1 = pd.read_excel(input_file)
        df1 = df1[keep_index]
        df1 = df1[:-3]
        data.append(df1)
    
    data = pd.concat(data, axis=0)
    data = data.reset_index(drop=True)

    print('Raw data information: ')
    data.info()
    print('-'*50 + '\n')
    
    '''
    Remove duplicate records from the different reaction files.
    '''
    records = set()
    duplicate_records = set()
    duplicate_index = []
    prev_id = 0
    for i in range(len(data)):
        reaxys_id = data['Reaction ID'][i]
        if reaxys_id not in records:
            records.add(reaxys_id)
            prev_id = reaxys_id
        else:
            if prev_id == reaxys_id:
                pass
            else:
                duplicate_records.add(reaxys_id)
                duplicate_index.append(i)
                
    data.drop(duplicate_index, inplace=True)
    data = data.reset_index(drop=True)
    print('Data information after removing duplicates: ')
    data.info()
    
    '''
    remove invalid smiles reaction in the data and data which has no solvent information.
    Also, we remove data that do not have yield report. And Temperature!
    '''
    # split reaction smiles to reactant and product and remove no smiles data
    data[['reactants', 'products']] = data['Reaction'].str.split('>>',expand=True)
    data = data.drop(['Reaction'], axis=1)
    
    # reactant
    indexNames = data[ [CheckNaN(data.iloc[i]['reactants']) for i in range(len(data))] ].index
    data.drop(indexNames, inplace=True)
    data = data.reset_index(drop=True)
    # product
    indexNames = data[ [CheckNaN(data.iloc[i]['products']) for i in range(len(data))] ].index
    data.drop(indexNames, inplace=True)
    data = data.reset_index(drop=True)
    # solvent
    indexNames = data[ [CheckNaN(data.iloc[i]['Solvent (Reaction Details)']) for i in range(len(data))] ].index
    data.drop(indexNames, inplace=True)
    data = data.reset_index(drop=True)
    # yield
    indexNames = data[ [CheckYield(data.iloc[i]['Yield (numerical)']) for i in range(len(data))] ].index
    data.drop(indexNames, inplace=True)
    data = data.reset_index(drop=True)
    # temperature: pick the highest temperature in the stages,
    # (Because we train the ranking and regression using two separate dataset)
    data['Temperature (Reaction Details) [C]'] = [highest_temperature(t) for t in data['Temperature (Reaction Details) [C]']]
    # indexNames = data[ [CheckNaN(data.iloc[i]['Temperature (Reaction Details) [C]']) for i in range(len(data))] ].index
    # data.drop(indexNames, inplace=True)
    # data = data.reset_index(drop=True)
        
    remove_invalid_smiles(data)
    data = data.reset_index(drop=True)
    
    print('After removing nan data: ')
    data.info()
    print('-'*50 + '\n')
    
    '''
    #2021/01/17 remove duplicated reagent name and solvent name in one reaction.
    '''
    for i in range(len(data)):
        data.loc[i,'Solvent (Reaction Details)'] = remove_duplicated_records(str(data.loc[i,'Solvent (Reaction Details)']))
        data.loc[i,'Reagent'] = remove_duplicated_records(str(data.loc[i,'Reagent']))
    
    '''
    #2021/02/23 move some solvent name to reagent dictionary, and move some reagent name to solvent dictionary.
    '''
    reagent_dict = GetFrequencyDict(data['Reagent'])
    solvent_dict = GetFrequencyDict(data['Solvent (Reaction Details)'])
    solvent_dict.pop('nan',None)
    
    for key, value in solvent_dict.copy().items():
        if reagent_dict.get(key):
            if value >= reagent_dict[key]:
                reagent_dict.pop(key, None)
            else:
                solvent_dict.pop(key, None)
                
    for i in range(len(data)):
        r_s = data.loc[i,'Reagent'].split('; ')
        s_s = data.loc[i,'Solvent (Reaction Details)'].split('; ')
        for r in r_s.copy():
            if r in solvent_dict.keys():
                r_s.remove(r)
                s_s.append(r)
        for s in s_s.copy():
            if s not in solvent_dict.keys():
                s_s.remove(s)
                r_s.append(s)
        if r_s == []:
            data.loc[i,'Reagent'] = 'nan'
        else:
            data.loc[i,'Reagent'] = '; '.join(r_s)
        if s_s == []:
            data.loc[i,'Solvent (Reaction Details)'] = 'nan'
        else:
            data.loc[i,'Solvent (Reaction Details)'] = '; '.join(s_s)
        
    
    indexNames = data[ [CheckNaN(data.iloc[i]['Solvent (Reaction Details)']) for i in range(len(data))] ].index
    data.drop(indexNames, inplace=True)
    data = data.reset_index(drop=True)
    
    print('After reassign reaction role, data information: ')
    data.info()
    print('-'*50 + '\n')
    '''
    #2021/02/23 remove duplicated reagent name and solvent name in one reaction. AGAIN!!!
    '''
    for i in range(len(data)):
        data.loc[i,'Solvent (Reaction Details)'] = remove_duplicated_records(str(data.loc[i,'Solvent (Reaction Details)']))
        data.loc[i,'Reagent'] = remove_duplicated_records(str(data.loc[i,'Reagent']))
    
    '''
    #2021/01/17 remove solvent, reagent number >2
    '''
    
    indexNames = data[ [CheckSemicolon_count(str(data.iloc[i]['Solvent (Reaction Details)'])) for i in range(len(data))] ].index
    data.drop(indexNames, inplace=True)
    data = data.reset_index(drop=True)
    indexNames = data[ [CheckSemicolon_count(str(data.iloc[i]['Reagent'])) for i in range(len(data))] ].index
    data.drop(indexNames, inplace=True)
    data = data.reset_index(drop=True)
    
    print('After remove the reaction with solvent or reagent number > 2, data information: ')
    data.info()
    print('-'*50 + '\n')
    '''
    #2021/01/17 remove duplicated reaction condition records.
    '''
    
    rxn_id = list(data['Reaction ID'])
    start_ = 0
    result = []
    memory = []
    indexNames = []
    for i in range(len(rxn_id)):
        if rxn_id[i] != rxn_id[start_]:
            start_ = i
            memory = [str(data.iloc[i]['Reagent'])+'+'+str(data.iloc[i]['Solvent (Reaction Details)'])]
        else:
            condition = str(data.iloc[i]['Reagent'])+'+'+str(data.iloc[i]['Solvent (Reaction Details)'])
            if condition not in memory:
                memory.append(condition)
            else:
                indexNames.append(i)
    
    data.drop(indexNames, inplace=True)
    data = data.reset_index(drop=True)

    
    # See the frequency dict of reagent & solvent, almost zero catalyst in Reaxys data:
    reagent_dict = GetFrequencyDict(data['Reagent'])
    plot_frequency(reagent_dict, 'Reagent', os.path.join(output_file_path, 'Reagent_plot_first.png'))
    solvent_dict = GetFrequencyDict(data['Solvent (Reaction Details)'])
    solvent_dict.pop('nan', None)
    plot_frequency(solvent_dict, 'Solvent', os.path.join(output_file_path, 'Solvent_plot_first.png'))
    catalyst_dict = GetFrequencyDict(data['Catalyst'])
    plot_frequency(catalyst_dict)
    
    
    '''
    After observing the distribution of reagent and solvent data, now we remove data that uses reagent and solvent whose frequency < 10.
    
    '''
    
    rm_reagent = get_remove_list(reagent_dict, 10)
    rm_solvent = get_remove_list(solvent_dict, 10)
    
    indexNames = data[ [check_rm(data.iloc[i]['Reagent'], rm_reagent) for i in range(len(data))] ].index
    data.drop(indexNames, inplace=True)
    data = data.reset_index(drop=True)
    indexNames = data[ [check_rm(data.iloc[i]['Solvent (Reaction Details)'], rm_solvent) for i in range(len(data))] ].index
    data.drop(indexNames, inplace=True)
    data = data.reset_index(drop=True)

    
    '''
    second remove < 5
    '''
    reagent_dict = GetFrequencyDict(data['Reagent'])
    solvent_dict = GetFrequencyDict(data['Solvent (Reaction Details)'])
    solvent_dict.pop('nan', None)
    rm_reagent = get_remove_list(reagent_dict, 5)
    rm_solvent = get_remove_list(solvent_dict, 5)
    
    indexNames = data[ [check_rm(data.iloc[i]['Reagent'], rm_reagent) for i in range(len(data))] ].index
    data.drop(indexNames, inplace=True)
    data = data.reset_index(drop=True)
    indexNames = data[ [check_rm(data.iloc[i]['Solvent (Reaction Details)'], rm_solvent) for i in range(len(data))] ].index
    data.drop(indexNames, inplace=True)
    data = data.reset_index(drop=True)
    
    
    # observe the frequency plot again
    
    reagent_dict = GetFrequencyDict(data['Reagent'])
    plot_frequency(reagent_dict, 'Reagent Frequency After Removal', os.path.join(output_file_path, 'Reagent_plot_second.png'))
    solvent_dict = GetFrequencyDict(data['Solvent (Reaction Details)'])
    solvent_dict.pop('nan', None)
    plot_frequency(solvent_dict, 'Solvent Frequency After Removal', os.path.join(output_file_path, 'Solvent_plot_second.png'))
    
    print('After removing data with frequency < 10: ')
    data.info()
    print('-'*50 + '\n')
    data.to_csv(os.path.join(output_file_path, 'processed_data.csv'))
    
    '''
    Now split and save the second part data
    (1) train, validate and test data (pd.Dataframe) -> .txt
    (2) solvent and reagent name classes (dict) -> .pkl
    '''
    
    train_data, validate_data, test_data = train_validate_test_split_for_Reaxys_condition(data)
    
    train_data = train_data.reset_index(drop=True)
    validate_data = validate_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    
    output_file_path_second_part = os.path.join(output_file_path, 'For_second_part_model')
    os.makedirs(output_file_path_second_part, exist_ok=True)
    
    write_DF2text_second_part(data, os.path.join(output_file_path_second_part, 'Splitted_second_{}.txt'.format('total')))
    write_DF2text_second_part(train_data, os.path.join(output_file_path_second_part, 'Splitted_second_{}.txt'.format('train')))
    write_DF2text_second_part(validate_data, os.path.join(output_file_path_second_part, 'Splitted_second_{}.txt'.format('validate')))
    write_DF2text_second_part(test_data, os.path.join(output_file_path_second_part, 'Splitted_second_{}.txt'.format('test')))
    
    output_file_path_class = os.path.join(output_file_path, 'unprocessed_class')
    os.makedirs(output_file_path_class, exist_ok=True)
    save_dict2pkl(reagent_dict, os.path.join(output_file_path_class, 'class_names_{}.pkl'.format('reagent')))
    save_dict2pkl(solvent_dict, os.path.join(output_file_path_class, 'class_names_{}.pkl'.format('solvent')))
    
    """
    Write the solvent and reagent dictionary into .txt files. => for label process
    """
    
    f = open(os.path.join(output_file_path_class, 'class_names_reagent.txt'), 'w')
    for key, value in reagent_dict.items():
        f.write(key + '\n')
    f.close()
    f = open(os.path.join(output_file_path_class, 'class_names_solvent.txt'), 'w')
    for key, value in solvent_dict.items():
        f.write(key + '\n')
    f.close()
    
    '''
    Now we have to prepare the data for the first part of rxn_yield_context model.
    Each reaction has its own canditate lists of solvent and reagent.
    '''
    output_file_path_first_part = os.path.join(output_file_path, 'For_first_part_model')
    os.makedirs(output_file_path_first_part, exist_ok=True)
    
    write_DF2text_first_part(data, os.path.join(output_file_path_first_part, 'Splitted_first_{}.txt'.format('total')))
    write_DF2text_first_part(train_data, os.path.join(output_file_path_first_part, 'Splitted_first_{}.txt'.format('train')))
    write_DF2text_first_part(validate_data, os.path.join(output_file_path_first_part, 'Splitted_first_{}.txt'.format('validate')))
    write_DF2text_first_part(test_data, os.path.join(output_file_path_first_part, 'Splitted_first_{}.txt'.format('test')))
    

    
    