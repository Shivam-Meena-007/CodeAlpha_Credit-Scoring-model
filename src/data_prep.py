import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(csv_path):
    return pd.read_csv(csv_path)

def feature_engineering(df):
    df = df.copy()
    # common features
    df['debt_to_income'] = df['debt'] / (df['income'] + 1)
    df['savings_ratio'] = df['savings'] / (df['income'] + 1)
    df['expense_ratio'] = df['monthly_expenses'] / (df['income'] / 12 + 1)
    # fillna if any
    df = df.fillna(0)
    return df

def split_data(df, target_col='target', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
