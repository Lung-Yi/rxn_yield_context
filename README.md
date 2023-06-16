# rxn_yield_context
# Preprocess data
cd rxn_yield_context/preprocess_data/

See the notes about preprocessing the data.

# Train the first model (multi-task multi-label classification model).
cd rxn_yield_context/train_multilabel/

# Use the trained model to predict
cd rxn_yield_context/evaluate_model

python evaluate_one_example.py

(needs to change the rxn_smile in the evaluate_one_example.py)


