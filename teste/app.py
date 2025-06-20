import streamlit as st
import pandas as pd
import joblib

# Carregar modelo
modelo = joblib.load("modelo_irrigacao.pkl")

st.title("💧 Previsão de Irrigação - FarmTech Solutions")
st.markdown("Este sistema decide se é necessário irrigar com base na umidade do solo.")

# Entrada manual
umidade = st.slider("Umidade do Solo (%)", min_value=0.0, max_value=100.0, step=0.1)

# Previsão
if st.button("Verificar Necessidade de Irrigação"):
    resultado = modelo.predict([[umidade]])[0]
    if resultado == 1:
        st.warning("🚨 Irrigação Necessária!")
    else:
        st.success("✅ Solo não precisa ser irrigado.")

# Exibir CSV original para contexto
df = pd.read_csv("dataset_umidade.csv")
st.subheader("📊 Base de Dados Simulada")
st.dataframe(df.head(20))
