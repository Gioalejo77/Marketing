# -*- coding: utf-8 -*-
"""
ChatBot Caso de Estudio (Seguros) - Streamlit
Incluye:
- Parser robusto de fechas
- Fix "truth value ambiguous"
- FAQ dinámico con cifras
- Filtros en sidebar para 3 gráficas interactivas
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ===============================
# Configuración / Selección de archivo CSV
# ===============================
st.title("🚗 ChatBot Caso de Estudio (Seguros)")
st.write("Selecciona tu archivo CSV de caso de estudio")

csv_file = st.file_uploader("📂 Cargar CSV", type=["csv"])

if csv_file is not None:
    df_raw = pd.read_csv(csv_file)
    st.success("✅ Base cargada correctamente")
    st.dataframe(df_raw.astype(str), use_container_width=True)
else:
    st.warning("⚠️ No se ha cargado ningún archivo CSV.")
    st.stop()

# ===============================
# Helpers de formato y parsing
# ===============================
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

# ===============================
# Limpieza principal
# ===============================
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

# ===============================
# Preparar DataFrame
# ===============================
df = prepare_df(df_raw)

# ===============================
# Sidebar filtros para gráficas
# ===============================
st.sidebar.header("🔍 Filtros (solo gráficas)")
state_sel = st.sidebar.multiselect("Estado", options=sorted(df["State"].dropna().unique()) if "State" in df.columns else [])
channel_sel = st.sidebar.multiselect("Canal de Venta", options=sorted(df["Sales Channel"].dropna().unique()) if "Sales Channel" in df.columns else [])
month_sel = st.sidebar.multiselect("Mes de Vigencia", options=sorted(df["Effective_Month"].dropna().unique()) if "Effective_Month" in df.columns else [])
vehicle_sel = st.sidebar.multiselect("Clase de Vehículo", options=sorted(df["Vehicle Class"].dropna().unique()) if "Vehicle Class" in df.columns else [])

# ===============================
# Función para aplicar filtros solo a gráficas
# ===============================
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

# ===============================
# Mostrar 3 gráficas
# ===============================
def draw_dashboard(df_filtered: pd.DataFrame):
    st.header("📊 Panel (3 gráficas) — Segmentado por filtros")

    if df_filtered.empty:
        st.info("No hay datos con los filtros seleccionados.")
        return

    # 1) Prima mensual promedio por cobertura
    st.subheader("1) Prima mensual promedio por cobertura")
    if "Monthly Premium Auto" in df_filtered.columns and "Coverage" in df_filtered.columns:
        g1 = df_filtered.groupby("Coverage")["Monthly Premium Auto"].mean().sort_values(ascending=False)
        st.bar_chart(g1)

    # 2) Total de reclamos por estado (Top 5)
    st.subheader("2) Total de reclamos por estado (Top 5)")
    if "Total Claim Amount" in df_filtered.columns and "State" in df_filtered.columns:
        g2 = df_filtered.groupby("State")["Total Claim Amount"].sum().sort_values(ascending=False).head(5)
        st.bar_chart(g2)

    # 3) Tasa de aceptación por tipo de oferta de renovación
    st.subheader("3) Tasa de aceptación por tipo de oferta de renovación")
    if "Response" in df_filtered.columns and "Renew Offer Type" in df_filtered.columns:
        temp = df_filtered.copy()
        temp["accepted"] = np.where(temp["Response"].astype(str).str.strip().str.lower() == "yes", 1, 0)
        g3 = temp.groupby("Renew Offer Type")["accepted"].mean().sort_values(ascending=False)
        st.bar_chart(g3)

# ===============================
# Ejecutar dashboard con filtros
# ===============================
df_charts = df_for_charts(df)
draw_dashboard(df_charts)