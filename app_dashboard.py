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
    ".stApp { background-color: #060b12; color: #b8d0e8;",
    " font-family: 'Courier New', monospace; }",
    "div[data-testid='stMetric'] { background-color: #0b1220;",
    " border: 1px solid #162236; padding: 8px 12px; border-radius: 6px; }",
    "div[data-testid='stMetricLabel'] { color: #5c7c99 !important;",
    " font-size: 11px !important; }",
    "div[data-testid='stMetricValue'] { color: #e8f4ff !important;",
    " font-size: 18px !important; }",
    ".stTabs [data-baseweb='tab-list'] { background-color: #0b1220;",
    " border-bottom: 1px solid #162236; gap: 6px; flex-wrap: wrap; }",
    ".stTabs [data-baseweb='tab'] { color: #8ab8d8;",
    " font-family: 'Courier New', monospace; padding: 8px 14px; }",
    ".stTabs [aria-selected='true'] { color: #00c8ff !important;",
    " border-bottom: 2px solid #00c8ff !important;",
    " background-color: #0e1828 !important; }",
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
    st.caption(
        "TradingView Avanzado • Monte Carlo Histórico + Top 5 Opciones"
        " • Informe Corporativo 24/7"
    )
with col_status:
    bot_activo = st.toggle("🟢 BOT MONITOR", value=True)

st.sidebar.header("⚙️ Control de Activo")
ticker = st.sidebar.text_input(
    "Buscar Ticker (ej. NU, DIS, TSLA, SPY, MELI):", value="NU"
).strip().upper()
if not ticker:
    ticker = "NU"

intervalo = st.sidebar.selectbox("Temporalidad:", ["1d", "1h", "15m", "5m"], index=0)
presupuesto_max = st.sidebar.number_input(
    "Presupuesto Máx. Contrato (USD):",
    min_value=50, max_value=10000, value=300, step=50
)
umbral_confianza = st.sidebar.slider(
    "Confianza Mínima IA (%):", min_value=50, max_value=99, value=97
)

# ==============================================================================
# 2. PALETAS CORPORATIVAS POR TICKER Y UTILIDADES BLINDADAS
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
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    try:
        idx = pd.to_datetime(out.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        out.index = idx
    except Exception:
        pass
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Close" not in out.columns:
        return pd.DataFrame()
    return out.dropna(subset=["Close"])

def _descargar_ohlcv_resiliente(symbol, period, interval):
    try:
        df1 = yf.download(
            symbol, period=period, interval=interval,
            progress=False, auto_adjust=True
        )
        norm1 = _normalizar_ohlcv(df1)
        if not norm1.empty:
            return norm1
    except Exception:
        pass
    try:
        df2 = yf.Ticker(symbol).history(period=period, interval=interval)
        return _normalizar_ohlcv(df2)
    except Exception:
        return pd.DataFrame()

def _safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val):
            return default
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default

def fmt_num(val, prefix="$", suffix=""):
    try:
        if val is None or pd.isna(val):
            return "N/D"
        val = float(val)
        if not np.isfinite(val):
            return "N/D"
        abs_v = abs(val)
        if abs_v >= 1e12:
            return f"{prefix}{val/1e12:.2f}T{suffix}"
        if abs_v >= 1e9:
            return f"{prefix}{val/1e9:.2f}B{suffix}"
        if abs_v >= 1e6:
            return f"{prefix}{val/1e6:.1f}M{suffix}"
        return f"{prefix}{val:,.2f}{suffix}"
    except Exception:
        return "N/D"

def calcular_griegas_completas(S, K, T, sigma, tipo="call", r=0.045):
    S = _safe_float(S)
    K = _safe_float(K)
    T = max(_safe_float(T), 1.0 / 365.0)
    sigma = max(_safe_float(sigma), 0.001)
    if S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    try:
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        pdf_d1 = norm.pdf(d1)
        if tipo == "call":
            delta = norm.cdf(d1)
            theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
        else:
            delta = norm.cdf(d1) - 1.0
            theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0
        gamma = pdf_d1 / (S * sigma * sqrt_T)
        vega = S * pdf_d1 * sqrt_T / 100.0
        return {
            "delta": round(float(delta), 3),
            "gamma": round(float(gamma), 4),
            "theta": round(float(theta), 3),
            "vega": round(float(vega), 3),
            "rho": round(float(rho), 4)
        }
    except Exception:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

def obtener_prima_segura(row):
    bid = _safe_float(row.get("bid"), 0.0)
    ask = _safe_float(row.get("ask"), 0.0)
    last = _safe_float(row.get("lastPrice"), 0.0)
    if bid > 0 and ask > 0:
        return round((bid + ask) / 2.0, 2)
    if last > 0:
        return round(last, 2)
    if ask > 0:
        return round(ask, 2)
    if bid > 0:
        return round(bid, 2)
    return 0.0

# ==============================================================================
# 3. CARGA RAPIDA DE PRECIOS, MODELO IA, OPCIONES Y FUNDAMENTALES
# ==============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def cargar_datos_y_modelo(symbol, inter):
    df_diario = _descargar_ohlcv_resiliente(symbol, "2y", "1d")
    if df_diario.empty or len(df_diario) < 35:
        return None, None, 0.50

    df_diario["Ret_1d"] = df_diario["Close"].pct_change()
    df_diario["EMA_21"] = df_diario["Close"].ewm(span=21, adjust=False).mean()
    df_diario["Dist_EMA21"] = (
        (df_diario["Close"] - df_diario["EMA_21"])
        / df_diario["EMA_21"].replace(0, np.nan)
    )

    delta = df_diario["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    df_diario["RSI"] = 100.0 - (100.0 / (1.0 + (gain / loss.replace(0, np.nan))))
    df_diario["Vol_20d"] = df_diario["Ret_1d"].rolling(20).std()
    vol_mean = df_diario["Volume"].rolling(20).mean().replace(0, np.nan)
    df_diario["Vol_Rel"] = (df_diario["Volume"] / vol_mean).fillna(1.0)
    df_diario["Target"] = (
        df_diario["Close"].shift(-5) / df_diario["Close"] - 1.0 >= 0.02
    ).astype(int)

    features = ["Ret_1d", "Dist_EMA21", "RSI", "Vol_20d", "Vol_Rel"]
    df_feat = (
        df_diario[features]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
    )
    df_clean = df_feat.copy()
    df_clean["Target"] = df_diario["Target"]
    df_clean = df_clean.iloc[:-5].dropna()

    corte = int(len(df_clean) * 0.8)
    y_train = df_clean["Target"].iloc[:corte]
    if (
        len(df_clean) >= 35
        and y_train.nunique() > 1
        and int(y_train.value_counts().min()) >= 3
    ):
        try:
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
            modelo = CalibratedClassifierCV(rf, method="isotonic", cv=3)
            modelo.fit(df_clean[features].iloc[:corte], y_train)
            prob_call = float(modelo.predict_proba(df_feat.iloc[[-1]])[:, 1][0])
        except Exception:
            prob_call = float(y_train.mean())
    else:
        prob_call = float(y_train.mean()) if len(y_train) > 0 else 0.50

    if inter == "1d":
        df_velas = df_diario.tail(80).copy()
    else:
        periodo_velas = "1mo" if inter == "1h" else "5d"
        df_velas = _descargar_ohlcv_resiliente(symbol, periodo_velas, inter).tail(80)
        if df_velas.empty:
            df_velas = df_diario.tail(80).copy()

    df_velas["EMA9"] = df_velas["Close"].ewm(span=9, adjust=False).mean()
    df_velas["EMA21"] = df_velas["Close"].ewm(span=21, adjust=False
