# -*- coding: utf-8 -*-
"""
ChatBot Caso de Estudio (Seguros) - Streamlit
Cargado desde CSV (GitHub o local)
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# =============================
# Configuración de archivos
# =============================
CSV_FILE_LOCAL = r"C:\Users\Sala_\Downloads\casodeestudio.csv"
CSV_FILE_GITHUB = "https://raw.githubusercontent.com/usuario/repositorio/main/casodeestudio.csv"  # Reemplazar URL
EXCEL_FILE = r"chatbot_history.xlsx"

try:
    if os.path.exists(CSV_FILE_LOCAL):
        df_raw = pd.read_csv(CSV_FILE_LOCAL)
    else:
        df_raw = pd.read_csv(CSV_FILE_GITHUB)
    st.success("✅ Base cargada correctamente")
    st.dataframe(df_raw.astype(str), use_container_width=True)
except Exception as e:
    st.error(f"⚠️ No se pudo cargar el CSV: {e}")
    st.stop()

# =============================
# Helpers de formato y parsing
# =============================

def money(x, currency="$", decimals=0):
    try:
        return f"{currency}{x:,.{decimals}f}"
    except Exception:
        return str(x)

def pct(x, decimals=1):
    try:
        return f"{x*100:.{decimals}f}%"
    except Exception:
        return str(x)

def coalesce_pandas(x, fallback):
    if x is None:
        return fallback
    if isinstance(x, (pd.DataFrame, pd.Series)) and x.empty:
        return fallback
    return x

def parse_effective_to_date(series: pd.Series) -> pd.Series:
    s = series.copy()
    num = pd.to_numeric(s, errors="coerce")
    dt = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    s_str = s.astype(str).str.strip()
    fmt_list = ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%y", "%Y/%m/%d"]
    for fmt in fmt_list:
        mask = dt.isna()
        if not mask.any():
            break
        dt.loc[mask] = pd.to_datetime(s_str.loc[mask], format=fmt, errors="coerce")
    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(s_str.loc[mask], errors="coerce")
    return dt

# =============================
# Limpieza principal
# =============================

def prepare_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df.columns = [c.strip() for c in df.columns]
    if "Effective To Date" in df.columns:
        df["Effective To Date"] = parse_effective_to_date(df["Effective To Date"])
        df["Effective_Month"] = df["Effective To Date"].dt.to_period("M").astype(str)

    numeric_cols = [
        "Customer Lifetime Value", "Income", "Monthly Premium Auto",
        "Months Since Last Claim", "Months Since Policy Inception",
        "Number of Open Complaints", "Number of Policies", "Total Claim Amount"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    cat_cols = ["Coverage", "Policy Type", "Policy", "Sales Channel",
                "Vehicle Class", "Vehicle Size", "Response", "Renew Offer Type", "State"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df

df = prepare_df(df_raw)

# =============================
# Sidebar para filtros de gráficas
# =============================
st.sidebar.header("🔍 Filtros (solo gráficas)")
state_sel = st.sidebar.multiselect("Estado", sorted(df["State"].dropna().unique()) if "State" in df.columns else [])
channel_sel = st.sidebar.multiselect("Canal de Venta", sorted(df["Sales Channel"].dropna().unique()) if "Sales Channel" in df.columns else [])
month_sel = st.sidebar.multiselect("Mes de Vigencia", sorted(df["Effective_Month"].dropna().unique()) if "Effective_Month" in df.columns else [])
vehicle_sel = st.sidebar.multiselect("Clase de Vehículo", sorted(df["Vehicle Class"].dropna().unique()) if "Vehicle Class" in df.columns else [])

def df_for_charts(base: pd.DataFrame) -> pd.DataFrame:
    dfx = base.copy()
    if "State" in dfx.columns and state_sel:
        dfx = dfx[dfx["State"].isin(state_sel)]
    if "Sales Channel" in dfx.columns and channel_sel:
        dfx = dfx[dfx["Sales Channel"].isin(channel_sel)]
    if "Effective_Month" in dfx.columns and month_sel:
        dfx = dfx[dfx["Effective_Month"].isin(month_sel)]
    if "Vehicle Class" in dfx.columns and vehicle_sel:
        dfx = dfx[dfx["Vehicle Class"].isin(vehicle_sel)]
    return dfx

# =============================
# Guardar interacciones
# =============================
def save_interaction(user_msg, bot_response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame({"timestamp": [timestamp], "usuario": [user_msg], "bot": [bot_response]})
    if os.path.exists(EXCEL_FILE):
        try:
            df_existing = pd.read_excel(EXCEL_FILE)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_final = df_new
    else:
        df_final = df_new
    df_final.to_excel(EXCEL_FILE, index=False)

# =============================
# NLP (TF-IDF + KNN)
# =============================
training_phrases = {
    "coberturas": ["oportunidades por tipo de cobertura","cómo mejorar el mix de coberturas","migrar clientes a premium o extended","qué cobertura genera mejor margen"],
    "vigencia": ["picos de altas por mes","qué meses conviene hacer campañas","oportunidades de retención por vigencia","cohortes por fecha de inicio"],
    "pago_mensual": ["clientes con prima alta para ajuste","dónde ofrecer add-ons por prima","asequibilidad de la prima","mejorar pricing mensual"],
    "clv": ["segmentación por CLV","priorizar clientes de alto valor","oportunidades de up-sell por CLV","cómo mejorar el CLV"],
    "num_polizas": ["oportunidades de cross sell","clientes con una sola póliza","bundling de productos","aumentar share of wallet"],
    "reclamos": ["hotspots de siniestros","dónde bajar severidad de reclamos","frecuencia y severidad por estado","ajustes de deducible y precio"],
    "quejas": ["reducir quejas abiertas","mejorar experiencia por canal","oportunidades para bajar TAT","priorizar acciones de servicio"],
    "canales": ["qué canal vende mejor con buen margen","canales con mejor conversión","dónde invertir en ventas","desempeño por canal"],
    "vehiculo": ["pricing por clase de vehículo","segmentos de mayor riesgo","oportunidades por tamaño del vehículo","ajustar tarifas por vehículo"],
    "renovacion": ["mejor oferta de renovación","tasa de aceptación por oferta","A/B test de renovación","cómo subir la renovación"]
}

X, y = [], []
for intent, phrases in training_phrases.items():
    for phrase in phrases:
        X.append(phrase)
        y.append(intent)

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
model = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X_vec)

def predict_intent(user_input: str):
    user_vec = vectorizer.transform([user_input])
    dist, idx = model.kneighbors(user_vec)
    intent = y[idx[0][0]]
    confidence = 1 - dist[0][0]
    return intent if confidence >= 0.5 else None

# =============================
# Dashboard
# =============================
def draw_dashboard(df_filtered: pd.DataFrame):
    st.header("📊 Panel (3 gráficas) — Segmentado por filtros")
    if df_filtered.empty:
        st.info("No hay datos con los filtros seleccionados.")
        return

    st.subheader("1) Prima mensual promedio por cobertura")
    if "Monthly Premium Auto" in df_filtered.columns and "Coverage" in df_filtered.columns:
        g1 = df_filtered.groupby("Coverage", observed=True)["Monthly Premium Auto"].mean().sort_values(ascending=False)
        st.bar_chart(g1)

    st.subheader("2) Total de reclamos por estado (Top 5)")
    if "Total Claim Amount" in df_filtered.columns and "State" in df_filtered.columns:
        g2 = df_filtered.groupby("State", observed=True)["Total Claim Amount"].sum().sort_values(ascending=False).head(5)
        st.bar_chart(g2)

    st.subheader("3) Tasa de aceptación por tipo de oferta de renovación")
    if "Response" in df_filtered.columns and "Renew Offer Type" in df_filtered.columns:
        temp = df_filtered.copy()
        temp["accepted"] = np.where(temp["Response"].astype(str).str.strip().str.lower() == "yes", 1, 0)
        g3 = temp.groupby("Renew Offer Type", observed=True)["accepted"].mean().sort_values(ascending=False)
        st.bar_chart(g3)

# =============================
# Interfaz Streamlit
# =============================
st.title("🚗 ChatBot Caso de Estudio (Seguros)")
st.write("Los **filtros de la izquierda** modifican **solo las 3 gráficas**. Las respuestas de texto usan toda la base.")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_input("👤 ¿Qué deseas preguntar?")

if user_input:
    intent = predict_intent(user_input)
    if intent:
        response = f"Respuesta simulada para el intent: {intent}"  # Aquí se puede integrar faq_generators
    else:
        response = "❓ No entendí tu consulta, por favor intenta con otra formulación."

    save_interaction(user_input, response)
    st.session_state["history"].append((user_input, response))

for user_msg, bot_msg in st.session_state["history"]:
    st.markdown(f"👤 **Tú:** {user_msg}")
    st.markdown(f"🤖 **Bot:** {bot_msg}")
    st.divider()

df_charts = df_for_charts(df)
draw_dashboard(df_charts)