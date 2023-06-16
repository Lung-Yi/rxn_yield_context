from rxn_yield_context.evaluate_model.eval_utils import ReactionContextPredictor
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_data_path',type=str,
                        default='data/reaxys_output_local')
    parser.add_argument('--input_data_path',type=str,
                        default='paper_examples.txt')
    parser.add_argument('--candidate_generation_model_path',type=str,
                        default='save_models/test_10R_first_local/multitask_model_epoch-80.checkpoint')
    parser.add_argument('--ranking_model_path',type=str,
                        default='save_models/test_10R_second_local_y/rxn_model_relevance_listwise_morgan_epoch-80.checkpoint')
    parser.add_argument('--cutoff_solvent',type=float,default=0.3)
    parser.add_argument('--cutoff_reagent',type=float,default=0.3)
    parser.add_argument('--verbose', default=True, type=lambda x: (str(x).lower() in ['false','0', 'no'])) # Whether to print failed prediction information
    args = parser.parse_args()

with open(args.input_data_path) as f:
    rxn_smiles_list = [rxn.strip() for rxn in f.readlines()]

rc_predictor = ReactionContextPredictor(args.class_data_path, args.candidate_generation_model_path, args.ranking_model_path, 
                                        cutoff_solv=args.cutoff_solvent, cutoff_reag=args.cutoff_reagent, verbose=args.verbose)
output_results = rc_predictor.recommend_reaction_context(rxn_smiles_list, max_display=20)
for i, result in enumerate(output_results):
    result.to_csv("example_results/{}.csv".format(i), index=False)