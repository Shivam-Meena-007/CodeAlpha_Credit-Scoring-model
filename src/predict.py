import argparse
import pandas as pd
import joblib
from src.data_prep import feature_engineering

def predict(model_path, preproc_path, input_csv):
    model = joblib.load(model_path)
    preproc = joblib.load(preproc_path)
    df = pd.read_csv(input_csv)
    df_proc = feature_engineering(df)
    Xp = preproc.transform(df_proc)
    preds = model.predict(Xp)
    out = df.copy()
    out['pred'] = preds
    try:
        probs = model.predict_proba(Xp)[:,1]
        out['prob'] = probs
    except Exception:
        pass
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--preproc', required=True)
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    res = predict(args.model, args.preproc, args.input)
    print(res.head())
