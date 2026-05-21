import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Prédicteur de Churn", page_icon="🎯")

@st.cache_resource
def load_model():
    if not Path("churn_model.pkl").exists():
        import train_model  # génère churn_model.pkl au premier démarrage
    return joblib.load("churn_model.pkl")

@st.cache_data
def get_risk_table():
    return pd.DataFrame({
        "Risque": ["Faible", "Élevé"],
        "Seuil de probabilité": ["< 50 %", "≥ 50 %"],
        "Action recommandée": ["Aucune action requise", "Contacter le client rapidement"],
    })

model = load_model()

st.title("Prédiction de Churn Client")
st.sidebar.header("Paramètres du Client")
tenure = st.sidebar.slider("Ancienneté (mois)", 1, 72, 12)
monthly_charges = st.sidebar.slider("Facture mensuelle (€)", 20, 120, 50)

if st.button("Lancer la Prédiction", type="primary"):
    if model is not None:
        input_data = pd.DataFrame([[tenure, monthly_charges]],
                                   columns=['tenure', 'MonthlyCharges'])
        prediction = model.predict_proba(input_data)[0][1]
        if prediction > 0.5:
            st.error(f"Risque de Churn Élevé : {prediction:.1%}")
        else:
            st.success(f"Risque de Churn Faible : {prediction:.1%}")

with st.expander("Grille d'interprétation"):
    st.dataframe(get_risk_table(), hide_index=True)
