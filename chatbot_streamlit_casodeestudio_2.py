# -*- coding: utf-8 -*-
"""
ChatBot Caso de Estudio (Seguros) - Streamlit
Carga CSV desde GitHub
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ------------------------------
# Configuración / Carga de CSV desde GitHub
# ------------------------------
CSV_FILE_GITHUB = "https://raw.githubusercontent.com/Gioalejo77/Marketing/main/casodeestudio.csv"

try:
    df_raw = pd.read_csv(CSV_FILE_GITHUB)
    st.success("✅ CSV cargado correctamente desde GitHub")
    st.dataframe(df_raw.head(), use_container_width=True)
except Exception as e:
    st.error(f"⚠️ No se pudo cargar el CSV desde GitHub: {e}")
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

# ==========================================================
# Sidebar (filtros para gráficas)
# ==========================================================
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

# ------------------------------
# Funciones de métricas
# ------------------------------
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

# ------------------------------
# FAQ dinámico
# ------------------------------
faq_generators = {
    "coberturas": lambda df_in: (
        (lambda top:
            "🛡️ **Coberturas con mejor margen estimado (prima - reclamos)**:\n"
            + _format_top_margin_lines(top)
        )(coalesce_pandas(_top_margin_by(df_in, "Coverage"), pd.DataFrame()))
    ),
    # Agregar otros intents según tu script original...
}

# ------------------------------
# Training phrases y NLP
# ------------------------------
training_phrases = {
    "coberturas": [
        "oportunidades por tipo de cobertura",
        "cómo mejorar el mix de coberturas",
        "migrar clientes a premium o extended",
        "qué cobertura genera mejor margen"
    ]
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

# ------------------------------
# Dashboard (3 gráficas)
# ------------------------------
def draw_dashboard(df_filtered: pd.DataFrame):
    st.header("📊 Panel (3 gráficas) — Segmentado por filtros")
    if df_filtered.empty:
        st.info("No hay datos con los filtros seleccionados.")
        return
    # 1) Prima mensual promedio por cobertura
    if "Monthly Premium Auto" in df_filtered.columns and "Coverage" in df_filtered.columns:
        g1 = df_filtered.groupby("Coverage", observed=True)["Monthly Premium Auto"].mean().sort_values(ascending=False).round(0)
        st.bar_chart(g1)
    # 2) Total de reclamos por estado
    if "Total Claim Amount" in df_filtered.columns and "State" in df_filtered.columns:
        g2 = df_filtered.groupby("State", observed=True)["Total Claim Amount"].sum().sort_values(ascending=False).head(5)
        st.bar_chart(g2)
    # 3) Tasa de aceptación por oferta
    if "Response" in df_filtered.columns and "Renew Offer Type" in df_filtered.columns:
        temp = df_filtered.copy()
        temp["accepted"] = np.where(temp["Response"].astype(str).str.strip().str.lower() == "yes", 1, 0)
        g3 = temp.groupby("Renew Offer Type", observed=True)["accepted"].mean().sort_values(ascending=False)
        st.bar_chart(g3)

# ------------------------------
# Interfaz Streamlit
# ------------------------------
st.title("🚗 ChatBot Caso de Estudio (Seguros)")
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
    st.session_state["history"].append((user_input, response))

# Mostrar historial
for user_msg, bot_msg in st.session_state["history"]:
    st.markdown(f"👤 **Tú:** {user_msg}")
    st.markdown(f"🤖 **Bot:** {bot_msg}")
    st.divider()

# Mostrar gráficas con filtros
df_charts = df_for_charts(df)
draw_dashboard(df_charts)
