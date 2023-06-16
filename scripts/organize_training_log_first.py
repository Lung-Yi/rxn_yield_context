import pandas as pd
import argparse
import os

def extract_five_lines(data_list):
    df = pd.DataFrame()
    for line in data_list:
        S = line.replace(',', ' ')
        S = S.split()
        cutoff = S[2]
        subdf = pd.DataFrame({'accuracy '+cutoff: [S[4]], 'precision '+cutoff: [S[6]],
                             'recall '+cutoff: [S[8]], 'f1-score '+cutoff: [S[10]], 'number preds '+cutoff:[S[-1]]})
        df = pd.concat([df, subdf], axis = 1)
    return df

parser = argparse.ArgumentParser()
parser.add_argument('--log_file', type = str, required = True)
args = parser.parse_args()

with open(args.log_file, 'r') as f:
    data = f.readlines()

solv_df = pd.DataFrame()
reag_df = pd.DataFrame()
for i, line in enumerate(data):
    if 'Solvent task' in line:
        solv_subdf = extract_five_lines(data[i+1:i+6])
        solv_df = pd.concat([solv_df, solv_subdf], axis=0)
    if 'Reagent task' in line:
        reag_subdf = extract_five_lines(data[i+1:i+6])
        reag_df = pd.concat([reag_df, reag_subdf], axis=0)

output_dir = os.path.dirname(args.log_file)
solv_saved_path = os.path.join(output_dir, "solvent_task.csv")
reag_saved_path = os.path.join(output_dir, "reagent_task.csv")

solv_df.to_csv(solv_saved_path)
reag_df.to_csv(reag_saved_path)
