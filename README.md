# Prédicteur de Churn Client

Application Streamlit de démonstration — prédit le risque de churn d'un client à partir de son ancienneté et de sa facture mensuelle.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://demo-churn-app.streamlit.app)

## Stack

- **Streamlit** — interface interactive
- **scikit-learn** — Gradient Boosting
- **joblib** — sérialisation du modèle

## Lancer en local

```bash
pip install -r requirements.txt
python train_model.py      # génère churn_model.pkl (une seule fois)
streamlit run app.py
```

## Déploiement Streamlit Cloud

1. Pusher ce repo sur GitHub (public)
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → sélectionner le repo → `main` → `app.py`
3. Cliquer **Deploy!**

> `churn_model.pkl` est dans `.gitignore` — Streamlit Cloud entraîne le modèle au démarrage via `train_model.py`.
