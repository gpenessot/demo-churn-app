"""
Entraîne le modèle de churn et le sauvegarde dans churn_model.pkl.
Lancer une fois : python train_model.py
"""
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
n = 2000

tenure = rng.uniform(1, 72, n)
monthly_charges = rng.uniform(20, 120, n)

churn_prob = (
    0.6 * (1 - tenure / 72)
    + 0.3 * (monthly_charges / 120)
    + rng.normal(0, 0.05, n)
)
y = (churn_prob > 0.45).astype(int)
X = np.column_stack([tenure, monthly_charges])

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(n_estimators=100, random_state=42)),
])
pipeline.fit(X, y)

joblib.dump(pipeline, "churn_model.pkl")
print("Modèle sauvegardé : churn_model.pkl")
print(f"Features : tenure, MonthlyCharges")
print(f"Accuracy : {pipeline.score(X, y):.2%}")
