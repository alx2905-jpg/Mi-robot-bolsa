import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime, timedelta

# ==============================================================================
# 1. CONFIGURACION GENERAL Y TEMA NEUTRO OSCURO
# ==============================================================================
st.set_page_config(
    page_title="OPTIONBOT PRO • Centro Cuantitativo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_app = [
    "<style>",
    ".stApp { background-color: #060b12; color: #b8d0e8; font-family: 'Courier New', monospace; }",
    "div[data-testid='stMetric'] { background-color: #0b1220; border: 1px solid #162236; padding: 8px 12px; border-radius: 6px; }",
    "div[data-testid='stMetricLabel'] { color: #5c7c99 !important; font-size: 11px !important; }",
    "div[data-testid='stMetricValue'] { color: #e8f4ff !important; font-size: 18px !important; }",
    ".stTabs [data-baseweb='tab-list'] { background-color: #0b1220; border-bottom: 1px solid #162236; gap: 6px; flex-wrap: wrap; }",
    ".stTabs [data-baseweb='tab'] { color: #8ab8d8; font-family: 'Courier New', monospace; padding: 8px 14px; }",
    ".stTabs [aria-selected='true'] { color: #00c8ff !important; border-bottom: 2px solid #00c8ff !important; background-color: #0e1828 !important; }",
    "</style>",
]
st.markdown("".join(css_app), unsafe_allow_html=True)

if "posiciones_ibkr" not in st.session_state:
    st.session_state.posiciones_ibkr = [
        {"Símbolo": "NU", "Tipo": "CALL", "Strike": 12.5, "Vencimiento": "2026-12-18", "Cant.": 2, "Costo Avg ($)": 0.85, "Precio Act. ($)": 1.10, "P&L ($)": 50.0},
        {"Símbolo": "SPY", "Tipo": "CALL", "Strike": 590.0, "Vencimiento": "2026-12-18", "Cant.": 1, "Costo Avg ($)": 4.50, "Precio Act. ($)": 5.80, "P&L ($)": 130.0},
    ]

st.sidebar.header("⚙️ Control de Activo")
ticker = st.sidebar.text_input("Buscar Ticker (ej. NU, DIS, TSLA, SPY, MELI):", value="NU").strip().upper() or "NU"
intervalo = st.sidebar.selectbox("Temporalidad:", ["1d", "1h", "15m", "5m"], index=0)
presupuesto_max = st.sidebar.number_input("Presupuesto Máx. Contrato (USD):", min_value=50, max_value=10000, value=300, step=50)
umbral_confianza = st.sidebar.slider("Confianza Mínima IA (%):", min_value=50, max_value=99, value=97)

col_logo, col_status = st.columns([4, 1])
with col_logo:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;'>"
        "<h1 style='margin:0;color:#e8f4ff;font-size:28px;'>⚡ OPTIONBOT PRO • IBKR & QUANT ENGINE</h1>"
        f"<span style='background:#0D0D0D;color:#FFFFFF;border:1px solid #3F3F46;padding:4px 14px;border-radius:6px;font-weight:700;font-size:
