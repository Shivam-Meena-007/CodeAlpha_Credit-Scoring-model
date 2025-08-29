import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def get_models(random_state=42):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "svm": SVC(probability=True, random_state=random_state)
    }
    if HAS_XGB:
        models['xgboost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    return models

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
