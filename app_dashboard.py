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
import html

# ==============================================================================
# 1. CONFIGURACION GENERAL Y TEMA VISUAL
# ==============================================================================
st.set_page_config(
    page_title="OPTIONBOT PRO • Centro Cuantitativo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
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
    "</style>"
]
st.markdown("".join(css_app), unsafe_allow_html=True)

if "posiciones_ibkr" not in st.session_state:
    st.session_state.posiciones_ibkr = [
        {
            "Símbolo": "NU", "Tipo": "CALL", "Strike": 12.5,
            "Vencimiento": "2026-12-18", "Cant.": 2,
            "Costo Avg ($)": 0.85, "Precio Act. ($)": 1.10, "P&L ($)": 50.0
        },
        {
            "Símbolo": "SPY", "Tipo": "CALL", "Strike": 590.0,
            "Vencimiento": "2026-12-18", "Cant.": 1,
            "Costo Avg ($)": 4.50, "Precio Act. ($)": 5.80, "P&L ($)": 130.0
        },
    ]

col_logo, col_status = st.columns([4, 1])
with col_logo:
    st.title("⚡ OPTIONBOT PRO • IBKR & QUANT ENGINE")
    st.caption("TradingView Avanzado • Monte Carlo Histórico + Top 5 Opciones • Informe Corporativo 24/7")
with col_status:
    bot_activo = st.toggle("🟢 BOT MONITOR", value=True)

st.sidebar.header("⚙️ Control de Activo")
ticker = st.sidebar.text_input("Buscar Ticker (ej. NU, DIS, TSLA, SPY, MELI):", value="NU").strip().upper()
if not ticker:
    ticker = "NU"

intervalo = st.sidebar.selectbox("Temporalidad:", ["1d", "1h", "15m", "5m"], index=0)
presupuesto_max = st.sidebar.number_input("Presupuesto Máx. Contrato (USD):", min_value=50, max_value=10000, value=300, step=50)
umbral_confianza = st.sidebar.slider("Confianza Mínima IA (%):", min_value=50, max_value=99, value=97)

# ==============================================================================
# 2. PALETAS CORPORATIVAS POR TICKER Y UTILIDADES
# ==============================================================================
BRAND_THEMES = {
    "NU":   {"navy": "#11052C", "accent": "#820AD1", "soft": "#F3E8FF"},
    "DIS":  {"navy": "#040714", "accent": "#0063E5", "soft": "#E6F0FF"},
    "TSLA": {"navy": "#171A20", "accent": "#E31937", "soft": "#FDE8EB"},
    "NVDA": {"navy": "#0B1408", "accent": "#76B900", "soft": "#EFF8E2"},
    "AAPL": {"navy": "#1D1D1F", "accent": "#0071E3", "soft": "#E8F3FF"},
    "AMZN": {"navy": "#131921", "accent": "#FF9900", "soft": "#FFF5E5"},
    "MSFT": {"navy": "#101820", "accent": "#00A4EF", "soft": "#E5F6FE"},
    "META": {"navy": "#0A131F", "accent": "#0668E1", "soft": "#E7F0FD"},
    "GOOGL":{"navy": "#101828", "accent": "#4285F4", "soft": "#E8F0FE"},
    "NFLX": {"navy": "#141414", "accent": "#E50914", "soft": "#FDE7E8"},
    "MELI": {"navy": "#2D3277", "accent": "#D4AF37", "soft": "#FFFBE6"},
    "SPY":  {"navy": "#0B192C", "accent": "#1E6091", "soft": "#E8F1F8"},
    "QQQ":  {"navy": "#0A1128", "accent": "#0096C7", "soft": "#E6F6FC"},
    "AMD":  {"navy": "#121212", "accent": "#ED1C24", "soft": "#FDE8E9"},
    "PLTR": {"navy": "#101418", "accent": "#2B59C3", "soft": "#EAEFFB"},
}

def obtener_colores_marca(symbol):
    if symbol in BRAND_THEMES:
        return BRAND_THEMES[symbol]
    paletas = [
        {"navy": "#0F172A", "accent": "#2563EB", "soft": "#EFF6FF"},
        {"navy": "#111827", "accent": "#059669", "soft": "#ECFDF5"},
        {"navy": "#1E1B4B", "accent": "#7C3AED", "soft": "#F5F3FF"},
        {"navy": "#1F2937", "accent": "#D97706", "soft": "#FFFBEB"},
    ]
    return paletas[sum(ord(c) for c in symbol) % len(paletas)]

def _normalizar_ohlcv(df):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["Close"])

def _safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val):
            return default
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default

def fmt_num
