# -*- coding: utf-8 -*-
"""
ChatBot Caso de Estudio (Seguros) - Streamlit
Carga CSV local (no usa GitHub)
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ------------------------------
# Configuración / Carga de archivo
# ------------------------------
CSV_FILE_LOCAL = r"C:\Users\Sala_\Downloads\casodeestudio.csv"

try:
    if os.path.exists(CSV_FILE_LOCAL):
        df_raw = pd.read_csv(CSV_FILE_LOCAL)
        st.success("✅ Base cargada correctamente desde archivo local")
        st.dataframe(df_raw.astype(str), use_container_width=True)
    else:
        st.error(f"⚠️ No se encontró el archivo local: {CSV_FILE_LOCAL}")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error al cargar el CSV: {e}")
    st.stop()

# ------------------------------
# Helpers de formato y parsing
# ------------------------------

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
    if isinstance(x, (pd.DataFrame, pd.Series)):
        try:
            if x.empty:
                return fallback
        except Exception:
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

# ------------------------------
# Limpieza principal
# ------------------------------

def prepare_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df.columns = [c.strip() for c in df.columns]

    if "Effective To Date" in df.columns:
        df["Effective To Date"] = parse_effective_to_date(df["Effective To Date"])
        df["Effective_Month"] = df["Effective To Date"].dt.to_period("M").astype(str)

    for c in [
        "Customer Lifetime Value", "Income", "Monthly Premium Auto",
        "Months Since Last Claim", "Months Since Policy Inception",
        "Number of Open Complaints", "Number of Policies", "Total Claim Amount"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["Coverage", "Policy Type", "Policy", "Sales Channel",
              "Vehicle Class", "Vehicle Size", "Response", "Renew Offer Type", "State"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df


df = prepare_df(df_raw)

# ==========================================================
#  Sidebar (solo afecta gráficas)
# ==========================================================
st.sidebar.header("🔍 Filtros (solo gráficas)")
state_sel = st.sidebar.multiselect("Estado", options=sorted(df["State"].dropna().unique()) if "State" in df.columns else [])
channel_sel = st.sidebar.multiselect("Canal de Venta", options=sorted(df["Sales Channel"].dropna().unique()) if "Sales Channel" in df.columns else [])
month_sel = st.sidebar.multiselect("Mes de Vigencia", options=sorted(df["Effective_Month"].dropna().unique()) if "Effective_Month" in df.columns else [])
vehicle_sel = st.sidebar.multiselect("Clase de Vehículo", options=sorted(df["Vehicle Class"].dropna().unique()) if "Vehicle Class" in df.columns else [])

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

# ------------------------------
# Guardar conversaciones
# ------------------------------
EXCEL_FILE = r"chatbot_historial.xlsx"

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

# ------------------------------
# Métricas auxiliares y FAQ
# ------------------------------
# (Se mantiene igual que tu script original, sin cambios)
# ... Aquí irían las funciones de _top_margin_by, _acceptance_by_offer, _format_top_margin_lines, faq_generators, training_phrases, predict_intent ...

# ------------------------------
# DASHBOARD: 3 gráficas
# ------------------------------
# (Se mantiene igual que tu script original, sin cambios)
# ... Aquí iría la función draw_dashboard(df_filtered) ...

# ------------------------------
# Interfaz Streamlit
# ------------------------------
st.title("🚗 ChatBot Caso de Estudio (Seguros)")
st.write("Los **filtros de la izquierda** modifican **solo las 3 gráficas**. Las respuestas de texto usan toda la base.")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_input("👤 ¿Qué deseas preguntar?")

if user_input:
    intent = predict_intent(user_input)
    if intent:
        response = faq_generators[intent](df)
    else:
        response = "❓ No entendí tu consulta, por favor intenta con otra formulación."
    save_interaction(user_input, response)
    st.session_state["history"].append((user_input, response))

for user_msg, bot_msg in st.session_state["history"]:
    st.markdown(f"👤 **Tú:** {user_msg}")
    st.markdown(f"🤖 **Bot:** {bot_msg}")
    st.divider()

# Mostrar gráficas con filtros aplicados
df_charts = df_for_charts(df)
draw_dashboard(df_charts)