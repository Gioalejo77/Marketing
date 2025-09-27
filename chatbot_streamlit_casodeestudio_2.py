# -*- coding: utf-8 -*-
"""
ChatBot Caso de Estudio (Seguros) - Streamlit
Carga CSV mediante upload (evita rutas locales o GitHub)
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ===================== Configuración =====================
st.set_page_config(page_title="ChatBot Caso Seguros", page_icon="🚗", layout="wide")

# ===================== Carga de CSV =====================
st.sidebar.header("Carga de archivo CSV")
uploaded_file = st.sidebar.file_uploader("Selecciona tu archivo CSV", type="csv")

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success("✅ CSV cargado correctamente")
        st.dataframe(df_raw.astype(str), use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ No se pudo cargar el CSV: {e}")
        st.stop()
else:
    st.warning("⚠️ No se ha cargado ningún CSV.")
    st.stop()

# ===================== Helpers de formato =====================
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

# ===================== Limpieza principal =====================
def prepare_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df.columns = [c.strip() for c in df.columns]

    # Fechas
    if "Effective To Date" in df.columns:
        df["Effective To Date"] = parse_effective_to_date(df["Effective To Date"])
        df["Effective_Month"] = df["Effective To Date"].dt.to_period("M").astype(str)

    # Numéricos clave
    for c in [
        "Customer Lifetime Value", "Income", "Monthly Premium Auto",
        "Months Since Last Claim", "Months Since Policy Inception",
        "Number of Open Complaints", "Number of Policies", "Total Claim Amount"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Categóricos frecuentes
    for c in ["Coverage", "Policy Type", "Policy", "Sales Channel",
              "Vehicle Class", "Vehicle Size", "Response", "Renew Offer Type", "State"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

df = prepare_df(df_raw)

# ===================== Sidebar filtros =====================
st.sidebar.header("🔍 Filtros (solo gráficas)")
state_sel = st.sidebar.multiselect(
    "Estado",
    options=sorted(df["State"].dropna().unique()) if "State" in df.columns else []
)
channel_sel = st.sidebar.multiselect(
    "Canal de Venta",
    options=sorted(df["Sales Channel"].dropna().unique()) if "Sales Channel" in df.columns else []
)
month_sel = st.sidebar.multiselect(
    "Mes de Vigencia",
    options=sorted(df["Effective_Month"].dropna().unique()) if "Effective_Month" in df.columns else []
)
vehicle_sel = st.sidebar.multiselect(
    "Clase de Vehículo",
    options=sorted(df["Vehicle Class"].dropna().unique()) if "Vehicle Class" in df.columns else []
)

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

# ===================== Métricas y FAQ =====================
def _top_margin_by(df_in, group_col, top_k=3):
    if group_col not in df_in.columns:
        return None
    tmp = df_in.groupby(group_col, dropna=False).agg(
        primas=("Monthly Premium Auto", "sum"),
        reclamos=("Total Claim Amount", "sum"),
        n=("Customer", "count")
    )
    tmp["margen"] = tmp["primas"] - tmp["reclamos"]
    tmp = tmp.sort_values("margen", ascending=False)
    return tmp.head(top_k)

def _acceptance_by_offer(df_in):
    if "Renew Offer Type" not in df_in.columns or "Response" not in df_in.columns:
        return None
    tmp = df_in.assign(accepted=(df_in["Response"].astype(str).str.strip().str.lower() == "yes").astype(int))
    acc = tmp.groupby("Renew Offer Type", observed=True)["accepted"].mean().sort_values(ascending=False)
    return acc

def _format_top_margin_lines(top: pd.DataFrame) -> str:
    if not isinstance(top, pd.DataFrame) or top.empty:
        return "Sin datos suficientes."
    lines = []
    for idx, row in top.iterrows():
        n_val = int(row["n"]) if pd.notna(row["n"]) else 0
        lines.append(
            f"- {idx}: margen {money(row['margen'],'$',0)} "
            f"(primas {money(row['primas'],'$',0)}, reclamos {money(row['reclamos'],'$',0)}, n={n_val})"
        )
    return "\n".join(lines)

# ===================== Entrenamiento NLP =====================
training_phrases = {
    "coberturas": ["oportunidades por tipo de cobertura", "qué cobertura genera mejor margen"],
    "vigencia": ["picos de altas por mes", "cohortes por fecha de inicio"],
    "pago_mensual": ["clientes con prima alta para ajuste", "mejorar pricing mensual"],
    "clv": ["segmentación por CLV", "cómo mejorar el CLV"],
    "num_polizas": ["oportunidades de cross sell", "clientes con una sola póliza"],
    "reclamos": ["hotspots de siniestros", "frecuencia y severidad por estado"],
    "quejas": ["reducir quejas abiertas", "priorizar acciones de servicio"],
    "canales": ["qué canal vende mejor con buen margen", "desempeño por canal"],
    "vehiculo": ["pricing por clase de vehículo", "ajustar tarifas por vehículo"],
    "renovacion": ["mejor oferta de renovación", "tasa de aceptación por oferta"]
}

X, y_list = [], []
for intent, phrases in training_phrases.items():
    for phrase in phrases:
        X.append(phrase)
        y_list.append(intent)

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
model = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X_vec)

def predict_intent(user_input: str):
    user_vec = vectorizer.transform([user_input])
    dist, idx = model.kneighbors(user_vec)
    intent = y_list[idx[0][0]]
    confidence = 1 - dist[0][0]
    return intent if confidence >= 0.5 else None

# ===================== Dashboard =====================
def draw_dashboard(df_filtered: pd.DataFrame):
    st.header("📊 Panel (3 gráficas) — Segmentado por filtros")

    if df_filtered.empty:
        st.info("No hay datos con los filtros seleccionados.")
        return

    # 1) Prima mensual promedio por cobertura
    st.subheader("1) Prima mensual promedio por cobertura")
    if "Monthly Premium Auto" in df_filtered.columns and "Coverage" in df_filtered.columns:
        g1 = df_filtered.groupby("Coverage", observed=True)["Monthly Premium Auto"].mean().sort_values(ascending=False)
        st.bar_chart(g1)
    else:
        st.info("No se encontraron columnas 'Monthly Premium Auto' y/o 'Coverage'.")

    # 2) Total de reclamos por estado (Top 5)
    st.subheader("2) Total de reclamos por estado (Top 5)")
    if "Total Claim Amount" in df_filtered.columns and "State" in df_filtered.columns:
        g2 = df_filtered.groupby("State", observed=True)["Total Claim Amount"].sum().sort_values(ascending=False).head(5)
        st.bar_chart(g2)
    else:
        st.info("No se encontraron columnas 'Total Claim Amount' y/o 'State'.")

    # 3) Tasa de aceptación por tipo de oferta de renovación
    st.subheader("3) Tasa de aceptación por tipo de oferta de renovación")
    if "Response" in df_filtered.columns and "Renew Offer Type" in df_filtered.columns:
        temp = df_filtered.copy()
        temp["accepted"] = np.where(temp["Response"].astype(str).str.strip().str.lower() == "yes", 1, 0)
        g3 = temp.groupby("Renew Offer Type", observed=True)["accepted"].mean()
        st.bar_chart(g3)
    else:
        st.info("No se encontraron columnas 'Response' y/o 'Renew Offer Type'.")

# ===================== Interfaz =====================
st.title("🚗 ChatBot Caso de Estudio (Seguros)")
st.write("Los **filtros de la izquierda** modifican **solo las 3 gráficas**. Las respuestas de texto usan toda la base.")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_input("👤 ¿Qué deseas preguntar?")

if user_input:
    intent = predict_intent(user_input)
    response = f"🤖 Respuesta simulada para intent '{intent}'" if intent else "❓ No entendí tu consulta."
    st.session_state["history"].append((user_input, response))

# Mostrar historial
for user_msg, bot_msg in st.session_state["history"]:
    st.markdown(f"👤 **Tú:** {user_msg}")
    st.markdown(f"🤖 **Bot:** {bot_msg}")
    st.divider()

# Mostrar las 3 gráficas
df_charts = df_for_charts(df)
draw_dashboard(df_charts)
