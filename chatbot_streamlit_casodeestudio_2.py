# -*- coding: utf-8 -*-
"""
ChatBot Caso de Estudio (Seguros) - Streamlit
Carga directa desde GitHub.
"""

import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ------------------------------
# Configuración / Carga de archivo desde GitHub
# ------------------------------
GITHUB_CSV_URL = "https://raw.githubusercontent.com/usuario/repositorio/main/casodeestudio.csv"

try:
    df_raw = pd.read_csv(GITHUB_CSV_URL)
    st.success("✅ Base cargada correctamente desde GitHub")
except Exception as e:
    st.error(f"⚠️ No se pudo cargar el archivo desde GitHub: {e}")
    st.stop()

# ------------------------------
# Funciones auxiliares (formato, parsing)
# ------------------------------
# ... aquí puedes copiar todas las funciones helper del script original: money, pct, coalesce_pandas, parse_effective_to_date, prepare_df, etc.

# ------------------------------
# Preparar DataFrame
# ------------------------------
df = prepare_df(df_raw)

# ------------------------------
# Sidebar filtros (solo gráficas)
# ------------------------------
# ... copiar sección sidebar completa

# ------------------------------
# Funciones de FAQ y NLP
# ------------------------------
# ... copiar faq_generators, training_phrases, predict_intent, etc.

# ------------------------------
# Panel de gráficas
# ------------------------------
# ... copiar draw_dashboard, df_for_charts

# ------------------------------
# Interfaz Streamlit
# ------------------------------
st.title("🚗 ChatBot Caso de Estudio (Seguros) - GitHub CSV")
st.write("Los **filtros de la izquierda** modifican **solo las 3 gráficas**. Las respuestas de texto usan toda la base.")

# Historial
if "history" not in st.session_state:
    st.session_state["history"] = []

# Entrada
user_input = st.text_input("👤 ¿Qué deseas preguntar?")

if user_input:
    intent = predict_intent(user_input)
    if intent:
        response = faq_generators[intent](df)
    else:
        response = "❓ No entendí tu consulta, por favor intenta con otra formulación."

    # Guardar y mostrar
    save_interaction(user_input, response)
    st.session_state["history"].append((user_input, response))

# Mostrar historial
for user_msg, bot_msg in st.session_state["history"]:
    st.markdown(f"👤 **Tú:** {user_msg}")
    st.markdown(f"🤖 **Bot:** {bot_msg}")
    st.divider()

# Mostrar gráficas con filtros aplicados
df_charts = df_for_charts(df)
draw_dashboard(df_charts)