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
from datetime import datetime

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
    "</style>",
]
st.markdown("".join(css_app), unsafe_allow_html=True)

if "posiciones_ibkr" not in st.session_state:
    st.session_state.posiciones_ibkr = [
        {
            "Símbolo": "NU", "Tipo": "CALL", "Strike": 12.5,
            "Vencimiento": "2026-12-18", "Cant.": 2,
            "Costo Avg ($)": 0.85, "Precio Act. ($)": 1.10, "P&L ($)": 50.0,
        },
        {
            "Símbolo": "SPY", "Tipo": "CALL", "Strike": 590.0,
            "Vencimiento": "2026-12-18", "Cant.": 1,
            "Costo Avg ($)": 4.50, "Precio Act. ($)": 5.80, "P&L ($)": 130.0,
        },
    ]

st.sidebar.header("⚙️ Control de Activo")
ticker_raw = st.sidebar.text_input(
    "Buscar Ticker (ej. NU, DIS, TSLA, SPY, MELI):", value="NU"
)
ticker = ticker_raw.strip().upper() or "NU"
intervalo = st.sidebar.selectbox(
    "Temporalidad:", ["1d", "1h", "15m", "5m"], index=0
)
presupuesto_max = st.sidebar.number_input(
    "Presupuesto Máx. Contrato (USD):",
    min_value=50, max_value=10000, value=300, step=50,
)
umbral_confianza = st.sidebar.slider(
    "Confianza Mínima IA (%):", min_value=50, max_value=99, value=97
)

col_logo, col_status = st.columns([4, 1])
with col_logo:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;'>"
        "<h1 style='margin:0;color:#e8f4ff;font-size:28px;'>"
        "⚡ OPTIONBOT PRO • IBKR & QUANT ENGINE</h1>"
        "<span style='background:#0D0D0D;color:#FFFFFF;"
        "border:1px solid #3F3F46;padding:4px 14px;border-radius:6px;"
        f"font-weight:700;font-size:15px;'>{ticker}</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "TradingView Avanzado • Monte Carlo Histórico + Top 5 Opciones"
        " • Cadena CALL/PUT • IBKR"
    )
with col_status:
    bot_activo = st.toggle("🟢 BOT MONITOR", value=True)

# ==============================================================================
# 2. UTILIDADES NUMERICAS, TRADINGVIEW Y GRIEGAS
# ==============================================================================
def widget_tradingview_html(symbol, tv_interval="D", height=540):
    partes = [
        "<!DOCTYPE html><html><body style='margin:0;padding:0;background:#060b12;'>",
        "<div class='tradingview-widget-container' style='height:"
        + str(height) + "px;width:100%'>",
        "<div class='tradingview-widget-container__widget'"
        " style='height:calc(100% - 32px);width:100%'></div>",
        "<script type='text/javascript'"
        " src='https://s3.tradingview.com/external-embedding/"
        "embed-widget-advanced-chart.js' async>",
        '{"allow_symbol_change":true,"calendar":false,"details":false,',
        '"hide_side_toolbar":false,"hide_top_toolbar":false,"hide_legend":false,',
        '"hide_volume":false,"hotlist":false,"interval":"' + tv_interval + '",',
        '"locale":"es","save_image":true,"style":"1","symbol":"' + symbol + '",',
        '"theme":"dark","timezone":"America/Bogota","backgroundColor":"#060b12",',
        '"gridColor":"rgba(255,255,255,0.06)","withdateranges":true,',
        '"studies":["STD;EMA","STD;RSI"],"autosize":true}',
        "</script></div></body></html>",
    ]
    return "".join(partes)

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
    return out.dropna(subset=["Close"]) if "Close" in out.columns else pd.DataFrame()

def _descargar_ohlcv_resiliente(symbol, period, interval):
    try:
        raw = yf.download(
            symbol, period=period, interval=interval,
            progress=False, auto_adjust=True,
        )
        norm1 = _normalizar_ohlcv(raw)
        if not norm1.empty:
            return norm1
    except Exception:
        pass
    try:
        hist = yf.Ticker(symbol).history(period=period, interval=interval)
        return _normalizar_ohlcv(hist)
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

def calcular_griegas_completas(S, K, T, sigma, tipo="call", r=0.045):
    S, K = _safe_float(S), _safe_float(K)
    T = max(_safe_float(T), 1.0 / 365.0)
    sigma = max(_safe_float(sigma), 0.001)
    if S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    try:
        sq = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sq)
        d2 = d1 - sigma * sq
        pdf = norm.pdf(d1)
        if tipo == "call":
            delta = norm.cdf(d1)
            theta = (
                -(S * pdf * sigma) / (2 * sq)
                - r * K * np.exp(-r * T) * norm.cdf(d2)
            ) / 365.0
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
        else:
            delta = norm.cdf(d1) - 1.0
            theta = (
                -(S * pdf * sigma) / (2 * sq)
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
            ) / 365.0
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0
        return {
            "delta": round(float(delta), 3),
            "gamma": round(float(pdf / (S * sigma * sq)), 4),
            "theta": round(float(theta), 3),
            "vega": round(float(S * pdf * sq / 100.0), 3),
            "rho": round(float(rho), 4),
        }
    except Exception:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

def obtener_prima_segura(row):
    bid = _safe_float(row.get("bid"), 0.0)
    ask = _safe_float(row.get("ask"), 0.0)
    last = _safe_float(row.get("lastPrice"), 0.0)
    if bid > 0 and ask > 0:
        return round((bid + ask) / 2.0, 2)
    return round(last, 2) if last > 0 else round(ask or bid, 2)

# ==============================================================================
# 3. CARGA DE DATOS, MODELO IA Y CADENA DE OPCIONES
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
        p_v = "1mo" if inter == "1h" else "5d"
        df_velas = _descargar_ohlcv_resiliente(symbol, p_v, inter).tail(80)
        if df_velas.empty:
            df_velas = df_diario.tail(80).copy()

    df_velas["EMA9"] = df_velas["Close"].ewm(span=9, adjust=False).mean()
    df_velas["EMA21"] = df_velas["Close"].ewm(span=21, adjust=False).mean()
    d_v = df_velas["Close"].diff()
    g_v = (d_v.where(d_v > 0, 0.0)).rolling(14).mean()
    l_v = (-d_v.where(d_v < 0, 0.
