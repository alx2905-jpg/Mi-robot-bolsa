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
import io

# ==============================================================================
# 1. TEMA VISUAL OPTIONBOT PRO (DARK TERMINAL)
# ==============================================================================
st.set_page_config(page_title="OPTIONBOT PRO • Centro Cuantitativo", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #060b12; color: #b8d0e8; font-family: 'Courier New', monospace; }
    div[data-testid="stMetric"] {
        background-color: #0b1220;
        border: 1px solid #162236;
        padding: 8px 12px;
        border-radius: 6px;
    }
    div[data-testid="stMetricLabel"] { color: #3a5570 !important; font-size: 11px !important; }
    div[data-testid="stMetricValue"] { color: #e8f4ff !important; font-size: 18px !important; }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0b1220;
        border-bottom: 1px solid #162236;
        gap: 6px;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8ab8d8;
        font-family: 'Courier New', monospace;
        padding: 8px 14px;
    }
    .stTabs [aria-selected="true"] {
        color: #00c8ff !important;
        border-bottom: 2px solid #00c8ff !important;
        background-color: #0e1828 !important;
    }
</style>
""", unsafe_allow_html=True)

if "posiciones_ibkr" not in st.session_state:
    st.session_state.posiciones_ibkr = [
        {"Símbolo": "NU", "Tipo": "CALL", "Strike": 12.5, "Vencimiento": "2026-12-18", "Cant.": 2, "Costo Avg ($)": 0.85, "Precio Act. ($)": 1.10, "P&L ($)": 50.0},
        {"Símbolo": "SPY", "Tipo": "CALL", "Strike": 590.0, "Vencimiento": "2026-12-18", "Cant.": 1, "Costo Avg ($)": 4.50, "Precio Act. ($)": 5.80, "P&L ($)": 130.0},
    ]

# --- HEADER ---
col_logo, col_status = st.columns([4, 1])
with col_logo:
    st.title("⚡ OPTIONBOT PRO • IBKR & QUANT ENGINE")
    st.caption("TradingView con Herramientas de Dibujo • Monte Carlo Histórico + Top 5 Opciones • Informe Corporativo 24/7")
with col_status:
    bot_activo = st.toggle("🟢 BOT MONITOR", value=True)

# --- BARRA LATERAL: UN SOLO CAMPO PARA BUSCAR EL TICKER ---
st.sidebar.header("⚙️ Control de Activo")
ticker = st.sidebar.text_input("Buscar Ticker (ej. NU, DIS, TSLA, SPY, MELI):", value="NU").strip().upper()
if not ticker:
    ticker = "NU"

intervalo = st.sidebar.selectbox("Temporalidad:", ["1d", "1h", "15m", "5m"], index=0)
presupuesto_max = st.sidebar.number_input("Presupuesto Máx. Contrato (USD):", min_value=50, max_value=10000, value=300, step=50)
umbral_confianza = st.sidebar.slider("Confianza Mínima IA (%):", min_value=50, max_value=99, value=97)

# ==============================================================================
# 2. PALETA DE COLORES POR TICKER Y FUNCIONES CUANTITATIVAS
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
    # Genera un tono corporativo consistente para cualquier otro ticker
    paletas = [
        {"navy": "#0F172A", "accent": "#2563EB", "soft": "#EFF6FF"},
        {"navy": "#111827", "accent": "#059669", "soft": "#ECFDF5"},
        {"navy": "#1E1B4B", "accent": "#7C3AED", "soft": "#F5F3FF"},
        {"navy": "#1F2937", "accent": "#D97706", "soft": "#FFFBEB"},
    ]
    return paletas[sum(ord(c) for c in symbol) % len(paletas)]

def fmt_num(val, prefix="$", suffix=""):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/D"
        val = float(val)
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
    if T <= 0 or sigma <= 0 or np.isnan(sigma):
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    if tipo == "call":
        delta = norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100
    return {
        "delta": round(delta, 3), "gamma": round(gamma, 4),
        "theta": round(theta, 3), "vega": round(vega, 3), "rho": round(rho, 4)
    }

def obtener_prima_segura(row):
    bid = row.get('bid', 0) or 0
    ask = row.get('ask', 0) or 0
    last = row.get('lastPrice', 0) or 0
    if bid > 0 and ask > 0:
        return round((bid + ask) / 2.0, 2)
    if last > 0:
        return round(float(last), 2)
    return round(float(ask if ask > 0 else bid), 2)

@st.cache_data(ttl=300)
def cargar_datos_y_modelo(symbol, inter):
    activo = yf.Ticker(symbol)
    df_diario = activo.history(period="2y", interval="1d")
    if df_diario.empty or len(df_diario) < 40:
        return None, None, None, {}, [], []

    df_diario['Ret_1d'] = df_diario['Close'].pct_change()
    df_diario['EMA_21'] = df_diario['Close'].ewm(span=21, adjust=False).mean()
    df_diario['Dist_EMA21'] = (df_diario['Close'] - df_diario['EMA_21']) / df_diario['EMA_21']

    delta = df_diario['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df_diario['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df_diario['Vol_20d'] = df_diario['Ret_1d'].rolling(20).std()
    vol_mean = df_diario['Volume'].rolling(20).mean().replace(0, np.nan)
    df_diario['Vol_Rel'] = (df_diario['Volume'] / vol_mean).fillna(1.0)
    df_diario['Target'] = (df_diario['Close'].shift(-5) / df_diario['Close'] - 1 >= 0.02).astype(int)

    features = ['Ret_1d', 'Dist_EMA21', 'RSI', 'Vol_20d', 'Vol_Rel']
    df_feat = df_diario[features].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    df_clean = df_feat.copy()
    df_clean['Target'] = df_diario['Target']
    df_clean = df_clean.iloc[:-5].dropna()

    corte = int(len(df_clean) * 0.8)
    y_train = df_clean['Target'].iloc[:corte]
    if len(df_clean) >= 30 and y_train.nunique() > 1:
        modelo = CalibratedClassifierCV(RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42), method='isotonic', cv=3)
        modelo.fit(df_clean[features].iloc[:corte], y_train)
        prob_call = float(modelo.predict_proba(df_feat.iloc[[-1]])[:, 1][0])
    else:
        prob_call = 0.50

    periodo_velas = "6mo" if inter == "1d" else "1mo" if inter == "1h" else "5d"
    df_velas = activo.history(period=periodo_velas, interval=inter).tail(80)
    if df_velas.empty:
        df_velas = df_diario.tail(80).copy()

    df_velas['EMA9'] = df_velas['Close'].ewm(span=9, adjust=False).mean()
    df_velas['EMA21'] = df_velas['Close'].ewm(span=21, adjust=False).mean()

    d_v = df_velas['Close'].diff()
    g_v = (d_v.where(d_v > 0, 0)).rolling(14).mean()
    l_v = (-d_v.where(d_v < 0, 0)).rolling(14).mean()
    df_velas['RSI'] = 100 - (100 / (1 + (g_v / l_v)))

    ema12 = df_velas['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_velas['Close'].ewm(span=26, adjust=False).mean()
    df_velas['MACD'] = ema12 - ema26
    df_velas['Signal'] = df_velas['MACD'].ewm(span=9, adjust=False).mean()
    df_velas['Hist'] = df_velas['MACD'] - df_velas['Signal']

    info = {}
    try:
        raw_info = activo.info
        if isinstance(raw_info, dict):
            info = raw_info
    except Exception:
        info = {}

    noticias_limpias = []
    try:
        raw_news = activo.news or []
        for n in raw_news[:3]:
            c = n.get("content", n) if isinstance(n, dict) else {}
            titulo = html.escape(str(c.get("title") or f"Actualización de mercado {symbol}"))
            resumen = html.escape(str(c.get("summary") or c.get("description") or "Reporte corporativo reciente."))
            prov_obj = c.get("provider")
            prov = html.escape(str(prov_obj.get("displayName") if isinstance(prov_obj, dict) else "Mercado"))
            link_obj = c.get("canonicalUrl") or c.get("clickThroughUrl")
            url = link_obj.get("url") if isinstance(link_obj, dict) else f"https://finance.yahoo.com/quote/{symbol}"
            noticias_limpias.append({"titulo": titulo, "resumen": resumen[:220], "prov": prov, "url": url})
    except Exception:
        noticias_limpias = []

    insiders_limpios = []
    try:
        df_ins = activo.insider_transactions
        if df_ins is not None and not df_ins.empty:
            for _, row in df_ins.head(4).iterrows():
                insiders_limpios.append({
                    "nombre": html.escape(str(row.get("Insider", "Directivo"))),
                    "trans": html.escape(str(row.get("Text", row.get("Transaction", "Operación SEC")))),
                    "shares": f"{int(row.get('Shares', 0)):,}" if pd.notna(row.get('Shares')) else "N/D",
                    "valor": fmt_num(row.get("Value"))
                })
    except Exception:
        insiders_limpios = []

    return df_diario, df_velas, prob_call, info, noticias_limpias, insiders_limpios

def widget_tradingview_html(symbol, tv_interval="D", theme="light", height=520):
    bg = "#ffffff" if theme == "light" else "#060b12"
    grid = "rgba(46, 46, 46, 0.06)" if theme == "light" else "rgba(255, 255, 255, 0.06)"
    return f"""<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:{bg};">
<div class="tradingview-widget-container" style="height:{height}px;width:100%">
  <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {{
    "allow_symbol_change": true,
    "calendar": false,
    "details": false,
    "hide_side_toolbar": false,
    "hide_top_toolbar": false,
    "hide_legend": false,
    "hide_volume": false,
    "hotlist": false,
    "interval": "{tv_interval}",
    "locale": "es",
    "save_image": true,
    "style": "1",
    "symbol": "{symbol}",
    "theme": "{theme}",
    "timezone": "America/Bogota",
    "backgroundColor": "{bg}",
    "gridColor": "{grid}",
    "watchlist": [],
    "withdateranges": true,
    "compareSymbols": [],
    "studies": ["STD;EMA", "STD;RSI"],
    "autosize": true
  }}
  </script>
</div>
</body>
</html>"""

PLANTILLA_HTML_REPORTE = '''<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <style>
    :root {
      --navy: __COLOR_NAVY__; --accent: __COLOR_ACCENT__; --accent-soft: __COLOR_SOFT__; --light: #F7F7F7;
      --card-bg: #FFFFFF; --text-dark: #1A1A1A; --text-muted: #666666;
      --green: #00875A; --green-bg: #E3FCEF; --red: #E53E3E; --red-bg: #FFEBE6;
      --gray-tag: #718096; --gray-tag-bg: #EDF2F7; --border-color: #E2E8F0;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      background-color: var(--light); color: var(--text-dark); line-height: 1.5; padding-bottom: 40px;
    }
    .hero { background-color: var(--navy); color: #FFFFFF; border-left: 14px solid var(--accent); padding: 28px 24px; margin-bottom: 20px; }
    .hero-container { max-width: 1200px; margin: 0 auto; }
    .hero-header { display: flex; align-items: center; gap: 18px; margin-bottom: 18px; }
    .logo-box {
      width: 82px; height: 82px; background-color: #FFFFFF; border-radius: 14px;
      display: flex; align-items: center; justify-content: center; flex-shrink: 0;
      padding: 8px; font-weight: 800; color: var(--navy); font-size: 20px;
    }
    .logo-box img { max-width: 100%; max-height: 100%; object-fit: contain; }
    .hero-title-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .hero-title-area h1 { font-family: Georgia, serif; font-size: 32px; font-weight: 700; color: #FFFFFF; }
    .ticker-badge {
      border: 1.5px solid var(--accent); color: #FFFFFF; padding: 4px 12px;
      border-radius: 20px; font-size: 14px; font-weight: 700; background: var(--accent);
    }
    .hero-subtitle { color: #CBD5E0; font-size: 14px; margin-top: 5px; }
    .stat-strip { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-top: 16px; }
    .stat-card { background-color: #FFFFFF; color: var(--text-dark); padding: 13px; border-radius: 10px; border-left: 4px solid var(--accent); }
    .stat-label { font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; margin-bottom: 4px; }
    .stat-value { font-size: 20px; font-weight: 700; color: var(--navy); }
    .stat-sub { font-size: 12px; margin-top: 3px; font-weight: 600; }
    .stat-sub.positive { color: var(--green); }
    .stat-sub.negative { color: var(--red); }
    .container { max-width: 1200px; margin: 0 auto; padding: 0 16px; }
    .dragsec { background: var(--card-bg); border-radius: 12px; padding: 20px; margin-bottom: 18px; border: 1px solid var(--border-color); }
    .dragsec-head { display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid var(--light); padding-bottom: 10px; margin-bottom: 16px; }
    .dragsec-head h2 { font-family: Georgia, serif; font-size: 20px; color: var(--navy); font-weight: 700; }
    .drag-handle { background: var(--accent-soft); border: 1px solid var(--accent); border-radius: 6px; padding: 4px 10px; font-size: 12px; color: var(--accent); font-weight: 700; }
    .about-card { font-size: 14.5px; line-height: 1.65; color: #2D3748; }
    .highlight { color: var
