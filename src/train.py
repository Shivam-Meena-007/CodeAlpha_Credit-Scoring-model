import os
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data_prep import load_data, feature_engineering, split_data
from src.models import get_models, save_model
import numpy as np

def train(csv_path, target='target', out_dir='saved_models', test_size=0.2):
    os.makedirs(out_dir, exist_ok=True)
    df = load_data(csv_path)
    df = feature_engineering(df)

    # Identify feature types
    numeric_feats = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target in numeric_feats:
        numeric_feats.remove(target)
    # simple categorical detection
    cat_feats = ['home_owner', 'married', 'education_level'] if set(['home_owner','married','education_level']).issubset(df.columns) else []

    X_train, X_test, y_train, y_test = split_data(df, target_col=target, test_size=test_size)

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_feats)], remainder='passthrough')

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    models = get_models()
    results = {}
    for name, model in models.items():
        print(f'Training {name} ...')
        try:
            model.fit(X_train_p, y_train)
        except Exception as e:
            print(f'Error training {name}:', e)
            continue
        preds = model.predict(X_test_p)
        probs = None
        try:
            probs = model.predict_proba(X_test_p)[:,1]
        except Exception:
            pass
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc = roc_auc_score(y_test, probs) if probs is not None else None

        results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc}
        print(f"{name}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, roc_auc={roc}")

        # save model and preprocessor
        base = os.path.splitext(os.path.basename(csv_path))[0]
        model_path = os.path.join(out_dir, f"{base}_{name}.joblib")
        preproc_path = os.path.join(out_dir, f"{base}_preprocessor.joblib")
        save_model(model, model_path)
        save_model(preprocessor, preproc_path)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to CSV dataset')
    parser.add_argument('--target', default='target')
    parser.add_argument('--out', default='saved_models')
    args = parser.parse_args()
    train(args.csv, target=args.target, out_dir=args.out)
