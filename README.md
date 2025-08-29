# Creditworthiness Prediction

This project demonstrates a creditworthiness prediction pipeline using classification algorithms
(Logistic Regression, Decision Tree, Random Forest, SVM, and optional XGBoost). It includes
feature engineering, training, evaluation, and prediction scripts along with a synthetic sample
dataset for demonstration.

## What's included
- `data/credit_data_sample.csv`: synthetic dataset (features + target)
- `data/credit_sample_input.csv`: sample input without target for predictions
- `src/`:
  - `data_prep.py` - loading and feature engineering
  - `models.py` - model factory and save/load helpers
  - `train.py` - training script (saves models + preprocessor)
  - `predict.py` - predict from input CSV using saved model and preprocessor
  - `evaluate.py` - evaluation helpers
  - `utils.py` - utility printing
- `saved_models/` - where trained models are saved after training
- `requirements.txt` - dependencies

## Quick start
```bash
pip install -r requirements.txt
# Train models on the sample dataset
python src/train.py --csv data/credit_data_sample.csv --target target --out saved_models
# Predict using one of the saved models (paths printed during training)
python src/predict.py --model saved_models/credit_data_sample_logistic_regression.joblib \
                     --preproc saved_models/credit_data_sample_preprocessor.joblib \
                     --input data/credit_sample_input.csv
```

## Notes & Next steps
- Replace synthetic data with real credit bureau datasets for production.
- Add cross-validation and hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
- Be careful with fairness and bias; add fairness checks before deployment.
