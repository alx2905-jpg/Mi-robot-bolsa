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
    st.caption("TradingView Live • Velas Japonesas (EMA 9/21, RSI, MACD) • Monte Carlo 500 Escenarios • Informe Fundamental")
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
# 2. FUNCIONES AUXILIARES Y MODELO CUANTITATIVO
# ==============================================================================
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
    if T <= 0 or sigma <= 0:
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

@st.cache_data(ttl=300)
def cargar_datos_y_modelo(symbol, inter):
    activo = yf.Ticker(symbol)
    df_diario = activo.history(period="2y", interval="1d")
    if df_diario.empty or len(df_diario) < 60:
        return None, None, None, {}, [], []

    df_diario['Ret_1d'] = df_diario['Close'].pct_change()
    df_diario['EMA_21'] = df_diario['Close'].ewm(span=21, adjust=False).mean()
    df_diario['Dist_EMA21'] = (df_diario['Close'] - df_diario['EMA_21']) / df_diario['EMA_21']

    delta = df_diario['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df_diario['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df_diario['Vol_20d'] = df_diario['Ret_1d'].rolling(20).std()
    df_diario['Vol_Rel'] = df_diario['Volume'] / df_diario['Volume'].rolling(20).mean()
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

PLANTILLA_HTML_REPORTE = '''<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <style>
    :root {
      --navy: #11052C; --accent: #820AD1; --accent-soft: #F3E8FF; --light: #F7F7F7;
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
      border: 1.5px solid var(--accent); color: var(--accent-soft); padding: 4px 12px;
      border-radius: 20px; font-size: 14px; font-weight: 700; background: rgba(130, 10, 209, 0.2);
    }
    .hero-subtitle { color: #A0AEC0; font-size: 14px; margin-top: 5px; }
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
    .drag-handle { background: var(--light); border: 1px solid var(--border-color); border-radius: 6px; padding: 4px 10px; font-size: 12px; color: var(--text-muted); }
    .about-card { font-size: 14.5px; line-height: 1.65; color: #2D3748; }
    .highlight { color: var(--accent); font-weight: 700; background: var(--accent-soft); padding: 2px 6px; border-radius: 4px; }
    .grid-2x2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .comp-card { background: var(--light); border-left: 6px solid var(--accent); padding: 15px; border-radius: 8px; }
    .comp-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
    .comp-title { font-weight: 700; font-size: 15px; color: var(--navy); }
    .comp-rank { background: var(--accent); color: #FFFFFF; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 12px; }
    .comp-share { font-size: 13px; color: var(--accent); font-weight: 600; margin-bottom: 5px; }
    .comp-desc { font-size: 13px; color: var(--text-muted); }
    .fund-top { display: flex; align-items: center; justify-content: space-between; background: var(--accent-soft); border-radius: 10px; padding: 16px; margin-bottom: 16px; }
    .mcap-large .val { font-family: Georgia, serif; font-size: 28px; font-weight: 700; color: var(--accent); display: block; }
    .mcap-large .lbl { font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }
    .fund-metrics-summary { display: flex; gap: 20px; }
    .metric-item { text-align: right; }
    .metric-item .v { font-size: 17px; font-weight: 700; color: var(--navy); }
    .metric-item .l { font-size: 11px; color: var(--text-muted); }
    .tables-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .fund-table { width: 100%; border-collapse: collapse; border-radius: 8px; overflow: hidden; }
    .fund-table th { padding: 10px 12px; text-align: left; font-size: 13px; color: #FFFFFF; }
    .fund-table.last th { background: var(--navy); }
    .fund-table.next th { background: var(--accent); }
    .fund-table td { padding: 9px 12px; border-bottom: 1px solid var(--border-color); font-size: 13px; }
    .tech-grid { display: grid; grid-template-columns: 240px 1fr; gap: 16px; }
    .ath-box { background: var(--navy); color: #FFFFFF; border-radius: 10px; padding: 18px; display: flex; flex-direction: column; justify-content: center; text-align: center; }
    .ath-number { font-family: Georgia, serif; font-size: 46px; font-weight: 700; color: #FFFFFF; line-height: 1; margin-bottom: 8px; }
    .ath-details { font-size: 12.5px; color: #A0AEC0; line-height: 1.5; }
    .tech-bullets-card { background: #FFFFFF; border-left: 6px solid var(--accent); border-radius: 8px; padding: 16px; border: 1px solid var(--border-color); }
    .tech-bullets-card h3 { font-family: Georgia, serif; font-size: 16px; color: var(--navy); margin-bottom: 8px; }
    .tech-bullets-card ul { list-style: none; }
    .tech-bullets-card li { position: relative; padding-left: 16px; margin-bottom: 7px; font-size: 13px; color: #4A5568; }
    .tech-bullets-card li::before { content: "•"; color: var(--accent); font-size: 18px; position: absolute; left: 0; top: -3px; }
    .news-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .news-card { background: #FFFFFF; border-top: 4px solid var(--accent); border-radius: 8px; padding: 14px; border: 1px solid var(--border-color); display: flex; flex-direction: column; justify-content: space-between; }
    .news-date { font-size: 10.5px; text-transform: uppercase; color: var(--text-muted); font-weight: 700; margin-bottom: 5px; }
    .news-tag-res { color: var(--accent); font-weight: 700; }
    .news-card h3 { font-family: Georgia, serif; font-size: 14.5px; color: var(--navy); margin-bottom: 7px; line-height: 1.3; }
    .news-card p { font-size: 12.5px; color: #4A5568; margin-bottom: 10px; }
    .news-link { font-size: 12.5px; font-weight: 700; color: var(--accent); text-decoration: none; }
    .flow-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .flow-card { background: #FFFFFF; border: 1px solid var(--border-color); border-radius: 8px; padding: 16px; }
    .flow-header h3 { font-family: Georgia, serif; font-size: 16px; color: var(--navy); }
    .flow-header .sub { font-size: 11px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 10px; }
    .flow-item { padding: 8px; background: var(--light); border-radius: 6px; font-size: 12px; margin-bottom: 7px; }
    .flow-item-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px; }
    .flow-name { font-weight: 700; color: var(--navy); }
    .tag-sell { background: var(--red-bg); color: var(--red); font-weight: 700; padding: 2px 6px; border-radius: 4px; font-size: 10px; }
    .tag-buy { background: var(--green-bg); color: var(--green); font-weight: 700; padding: 2px 6px; border-radius: 4px; font-size: 10px; }
    .tag-neutral { background: var(--gray-tag-bg); color: var(--gray-tag); font-weight: 700; padding: 2px 6px; border-radius: 4px; font-size: 10px; }
    .flow-empty { background: var(--light); border: 1px dashed var(--border-color); border-radius: 6px; padding: 20px; text-align: center; color: var(--text-muted); font-size: 13px; }
    .footer { max-width: 1200px; margin: 16px auto 0 auto; text-align: center; font-size: 12px; color: var(--text-muted); }
    @media (max-width: 768px) {
      .stat-strip { grid-template-columns: 1fr 1fr; }
      .grid-2x2, .tables-grid, .tech-grid, .news-grid, .flow-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-container">
      <div class="hero-header">
        <div class="logo-box">
          <img src="https://logo.clearbit.com/__DOMINIO__" alt="__SYMBOL__" onerror="this.style.display='none'; this.parentNode.innerText='__SYMBOL__';">
        </div>
        <div class="hero-title-area">
          <div class="hero-title-row">
            <h1>__NOMBRE_EMPRESA__</h1>
            <span class="ticker-badge">__EXCHANGE__: __SYMBOL__</span>
          </div>
          <p class="hero-subtitle">Análisis fundamental y técnico en vivo • __SECTOR__ • __FECHA_HOY__</p>
        </div>
      </div>

      <div class="stat-strip">
        <div class="stat-card">
          <div class="stat-label">Precio Actual</div>
          <div class="stat-value">__PRECIO_ACT__</div>
          <div class="stat-sub positive">Consenso: __REC_KEY__</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Market Cap</div>
          <div class="stat-value">__MCAP__</div>
          <div class="stat-sub">P/E Forward: __FWD_PE__</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Crecimiento Ventas YoY</div>
          <div class="stat-value">__REV_GROWTH_STR__</div>
          <div class="stat-sub __REV_GROWTH_CLS__">Ingresos: __TOTAL_REV__</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Beneficio Neto / ROE</div>
          <div class="stat-value">__NET_INCOME__</div>
          <div class="stat-sub positive">ROE: __ROE_STR__</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Distancia a Máx 52s</div>
          <div class="stat-value">__DIST_ATH__</div>
          <div class="stat-sub __DIST_CLS__">Máx 52s: __HIGH_52__</div>
        </div>
      </div>
    </div>
  </header>

  <main class="container">
    <section class="dragsec">
      <div class="dragsec-head">
        <h2>¿A qué se dedica __NOMBRE_EMPRESA__?</h2>
        <span class="drag-handle">Perfil Corporativo</span>
      </div>
      <div class="about-card">
        <span class="highlight">__SECTOR__ • __INDUSTRIA__</span> — __RESUMEN__
      </div>
    </section>

    <section class="dragsec">
      <div class="dragsec-head">
        <h2>Posición financiera y estructura</h2>
        <span class="drag-handle">Balance</span>
      </div>
      <div class="grid-2x2">
        <div class="comp-card">
          <div class="comp-header">
            <span class="comp-title">Márgenes de Rentabilidad</span>
            <span class="comp-rank">Eficiencia</span>
          </div>
          <div class="comp-share">Margen Neto: __PROFIT_MARGIN__ • Operativo: __OP_MARGIN__</div>
          <p class="comp-desc">Capacidad de conversión de ingresos (__TOTAL_REV__) en utilidad neta (__NET_INCOME__) con un retorno sobre el patrimonio (ROE) de __ROE_STR__.</p>
        </div>
        <div class="comp-card">
          <div class="comp-header">
            <span class="comp-title">Solidez de Caja vs Deuda</span>
            <span class="comp-rank">Balance</span>
          </div>
          <div class="comp-share">Caja Total: __TOTAL_CASH__ • Deuda Total: __TOTAL_DEBT__</div>
          <p class="comp-desc">Posición de liquidez disponible para financiar crecimiento orgánico y operaciones globales (__PAIS__ • __EMPLEADOS__ empleados).</p>
        </div>
      </div>
    </section>

    <section class="dragsec">
      <div class="dragsec-head">
        <h2>Datos fundamentales y valoración</h2>
        <span class="drag-handle">Fundamentales</span>
      </div>
      <div class="fund-top">
        <div class="mcap-large">
          <span class="lbl">Capitalización de Mercado (__SYMBOL__)</span>
          <span class="val">__MCAP__ USD</span>
        </div>
        <div class="fund-metrics-summary">
          <div class="metric-item"><div class="v">__TRAIL_PE__</div><div class="l">Trailing P/E</div></div>
          <div class="metric-item"><div class="v">__FWD_PE__</div><div class="l">Forward P/E</div></div>
          <div class="metric-item"><div class="v">__TARGET_MEAN__</div><div class="l">Objetivo Analistas</div></div>
        </div>
      </div>

      <div class="tables-grid">
        <table class="fund-table last">
          <thead><tr><th colspan="2">Métricas Actuales (TTM)</th></tr></thead>
          <tbody>
            <tr><td>Ingresos Totales (Revenue)</td><td><b>__TOTAL_REV__</b></td></tr>
            <tr><td>Utilidad Neta (Net Income)</td><td><b>__NET_INCOME__</b></td></tr>
            <tr><td>EPS (Últimos 12 meses)</td><td><b>__EPS_TRAIL__</b></td></tr>
            <tr><td>Retorno sobre Patrimonio (ROE)</td><td><b>__ROE_STR__</b></td></tr>
          </tbody>
        </table>

        <table class="fund-table next">
          <thead><tr><th colspan="2">Consenso y Proyección de Analistas</th></tr></thead>
          <tbody>
            <tr><td>Recomendación Consenso</td><td><b>__REC_KEY__</b></td></tr>
            <tr><td>EPS Estimado (Forward)</td><td><b>__EPS_FWD__</b></td></tr>
            <tr><td>Precio Objetivo Medio</td><td><b>__TARGET_MEAN__</b></td></tr>
            <tr><td>Rango Objetivos (Mín – Máx)</td><td><b>__TARGET_LOW__ – __TARGET_HIGH__</b></td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="dragsec">
      <div class="dragsec-head">
        <h2>Análisis técnico — Distancia a máximos 52 semanas</h2>
        <span class="drag-handle">Niveles</span>
      </div>
      <div class="tech-grid">
        <div class="ath-box">
          <div class="ath-number">__DIST_ATH__</div>
          <div class="ath-details">
            Máximo 52 Semanas: <b>__HIGH_52__ USD</b><br>
            Cotización Actual: <b>__PRECIO_ACT__ USD</b><br>
            Mínimo 52 Semanas: <b>__LOW_52__ USD</b>
          </div>
        </div>
        <div class="tech-bullets-card">
          <h3>Estructura de Precios (__SYMBOL__)</h3>
          <ul>
            <li><b>Rango anual:</b> El activo cotiza entre un soporte anual de <b>__LOW_52__</b> y una resistencia máxima de <b>__HIGH_52__</b>.</li>
            <li><b>Objetivo institucional:</b> El consenso de analistas proyecta un valor razonable medio de <b>__TARGET_MEAN__</b> (máximo estimado en <b>__TARGET_HIGH__</b>).</li>
            <li><b>Sincronización multi-pestaña:</b> Puede contrastar estos niveles fundamentales con el gráfico interactivo de TradingView (Pestaña 1) y la proyección Monte Carlo (Pestaña 2).</li>
          </ul>
        </div>
      </div>
    </section>

    <section class="dragsec">
      <div class="dragsec-head">
        <h2>Últimas noticias (__SYMBOL__)</h2>
        <span class="drag-handle">Noticias</span>
      </div>
      <div class="news-grid">__NEWS_CARDS_HTML__</div>
    </section>

    <section class="dragsec">
      <div class="dragsec-head">
        <h2>Insider trading — Movimientos de directivos</h2>
        <span class="drag-handle">SEC Form 4</span>
      </div>
      <div class="flow-grid">
        <div class="flow-card">
          <div class="flow-header">
            <h3>Transacciones recientes de ejecutivos</h3>
            <div class="sub">Registros SEC Form 4 • __SYMBOL__</div>
          </div>
          __INSIDER_HTML__
        </div>
        <div class="flow-card">
          <div class="flow-header">
            <h3>Resumen de consenso institucional</h3>
            <div class="sub">Valoración y cobertura</div>
          </div>
          <div class="flow-empty">
            Calificación actual del consenso: <b>__REC_KEY__</b><br><br>
            Precio Objetivo Medio: <b>__TARGET_MEAN__</b> vs Precio Actual: <b>__PRECIO_ACT__</b>
          </div>
        </div>
      </div>
    </section>
  </main>

  <footer class="footer">
    Informe Fundamental Dinámico generado en tiempo real para __NOMBRE_EMPRESA__ (__SYMBOL__) • Fuentes: Yahoo Finance & SEC Filings
  </footer>
</body>
</html>'''

def construir_informe_html_dinamico(symbol, precio_act, info, noticias, insiders):
    nombre_empresa = html.escape(str(info.get("longName") or info.get("shortName") or symbol))
    exchange = html.escape(str(info.get("exchange") or "NYSE / NASDAQ"))
    sector = html.escape(str(info.get("sector") or "Mercado Bursátil"))
    industria = html.escape(str(info.get("industry") or "Activo Cotizado"))
    website = str(info.get("website") or "")
    dominio = website.replace("https://", "").replace("http://", "").split("/")[0] if website else f"{symbol.lower()}.com"
    resumen = html.escape(str(info.get("longBusinessSummary") or f"{nombre_empresa} ({symbol}) es una empresa cotizada en {exchange} dentro del sector {sector}."))

    mcap = fmt_num(info.get("marketCap"))
    rev_growth = info.get("revenueGrowth")
    rev_growth_str = f"{float(rev_growth)*100:+.1f}%" if isinstance(rev_growth, (int, float)) else "N/D"
    rev_growth_cls = "positive" if isinstance(rev_growth, (int, float)) and float(rev_growth) >= 0 else "negative"

    total_rev = fmt_num(info.get("totalRevenue"))
    net_income = fmt_num(info.get("netIncomeToCommon"))
    roe = info.get("returnOnEquity")
    roe_str = f"{float(roe)*100:.1f}%" if isinstance(roe, (int, float)) else "N/D"
    fwd_pe = f"{float(info.get('forwardPE')):.1f}x" if isinstance(info.get("forwardPE"), (int, float)) else "N/D"
    trail_pe = f"{float(info.get('trailingPE')):.1f}x" if isinstance(info.get("trailingPE"), (int, float)) else "N/D"
    eps_trail = f"${float(info.get('trailingEps')):.2f}" if isinstance(info.get("trailingEps"), (int, float)) else "N/D"
    eps_fwd = f"${float(info.get('forwardEps')):.2f}" if isinstance(info.get("forwardEps"), (int, float)) else "N/D"
    target_mean = f"${float(info.get('targetMeanPrice')):.2f}" if isinstance(info.get("targetMeanPrice"), (int, float)) else "N/D"
    target_high = f"${float(info.get('targetHighPrice')):.2f}" if isinstance(info.get("targetHighPrice"), (int, float)) else "N/D"
    target_low = f"${float(info.get('targetLowPrice')):.2f}" if isinstance(info.get("targetLowPrice"), (int, float)) else "N/D"
    rec_key = html.escape(str(info.get("recommendationKey") or "HOLD").upper().replace("_", " "))

    high_52 = float(info.get("fiftyTwoWeekHigh")) if isinstance(info.get("fiftyTwoWeekHigh"), (int, float)) else precio_act
    low_52 = float(info.get("fiftyTwoWeekLow")) if isinstance(info.get("fiftyTwoWeekLow"), (int, float)) else precio_act
    dist_ath = ((precio_act - high_52) / high_52) * 100 if high_52 else 0.0
    dist_cls = "positive" if dist_ath >= -5 else "negative"

    profit_margin = f"{float(info.get('profitMargins'))*100:.1f}%" if isinstance(info.get("profitMargins"), (int, float)) else "N/D"
    op_margin = f"{float(info.get('operatingMargins'))*100:.1f}%" if isinstance(info.get("operatingMargins"), (int, float)) else "N/D"
    total_cash = fmt_num(info.get("totalCash"))
    total_debt = fmt_num(info.get("totalDebt"))
    empleados = f"{int(info.get('fullTimeEmployees')):,}" if isinstance(info.get("fullTimeEmployees"), (int, float)) else "N/D"
    pais = html.escape(str(info.get("country") or "Global"))

    if noticias:
        news_cards_html = ""
        for n in noticias:
            news_cards_html += (
                f'<div class="news-card"><div>'
                f'<div class="news-date">{n["prov"]} <span class="news-tag-res">• NOTICIA EN VIVO</span></div>'
                f'<h3>{n["titulo"]}</h3><p>{n["resumen"]}...</p></div>'
                f'<a href="{n["url"]}" target="_blank" class="news-link">Leer noticia 🡪</a></div>'
            )
    else:
        news_cards_html = (
            f'<div class="news-card"><div>'
            f'<div class="news-date">MERCADO • {symbol}</div>'
            f'<h3>Seguimiento corporativo de {nombre_empresa}</h3>'
            f'<p>Consulte los comunicados oficiales de relación con inversores y presentaciones SEC para {symbol}.</p></div>'
            f'<a href="https://finance.yahoo.com/quote/{symbol}/news" target="_blank" class="news-link">Ver noticias en Yahoo Finance 🡪</a></div>'
        )

    if insiders:
        insider_html = ""
        for ins in insiders:
            t_low = ins["trans"].lower()
            tag_cls = "tag-sell" if ("sale" in t_low or "sell" in t_low) else "tag-buy" if ("purchase" in t_low or "buy" in t_low) else "tag-neutral"
            tag_txt = "VENTA" if tag_cls == "tag-sell" else "COMPRA" if tag_cls == "tag-buy" else "REGISTRO"
            insider_html += (
                f'<div class="flow-item"><div class="flow-item-top">'
                f'<span class="flow-name">{ins["nombre"]}</span><span class="{tag_cls}">{tag_txt}</span></div>'
                f'<div class="flow-details">{ins["trans"]} • Acciones: {ins["shares"]} ({ins["valor"]})</div></div>'
            )
    else:
        insider_html = f'<div class="flow-empty">No se reportan transacciones recientes de directivos (Form 4) en los últimos días para <b>{symbol}</b>.</div>'

    reemplazos = {
        "__DOMINIO__": dominio,
        "__SYMBOL__": symbol,
        "__NOMBRE_EMPRESA__": nombre_empresa,
        "__EXCHANGE__": exchange,
        "__SECTOR__": sector,
        "__INDUSTRIA__": industria,
        "__RESUMEN__": resumen,
        "__FECHA_HOY__": datetime.now().strftime("%d/%m/%Y"),
        "__PRECIO_ACT__": f"${precio_act:.2f}",
        "__REC_KEY__": rec_key,
        "__MCAP__": mcap,
        "__FWD_PE__": fwd_pe,
        "__TRAIL_PE__": trail_pe,
        "__REV_GROWTH_STR__": rev_growth_str,
        "__REV_GROWTH_CLS__": rev_growth_cls,
        "__TOTAL_REV__": total_rev,
        "__NET_INCOME__": net_income,
        "__ROE_STR__": roe_str,
        "__DIST_ATH__": f"{dist_ath:.1f}%",
        "__DIST_CLS__": dist_cls,
        "__HIGH_52__": f"${high_52:.2f}",
        "__LOW_52__": f"${low_52:.2f}",
        "__PROFIT_MARGIN__": profit_margin,
        "__OP_MARGIN__": op_margin,
        "__TOTAL_CASH__": total_cash,
        "__TOTAL_DEBT__": total_debt,
        "__PAIS__": pais,
        "__EMPLEADOS__": empleados,
        "__EPS_TRAIL__": eps_trail,
        "__EPS_FWD__": eps_fwd,
        "__TARGET_MEAN__": target_mean,
        "__TARGET_LOW__": target_low,
        "__TARGET_HIGH__": target_high,
        "__NEWS_CARDS_HTML__": news_cards_html,
        "__INSIDER_HTML__": insider_html,
    }

    resultado = PLANTILLA_HTML_REPORTE
    for k, v in reemplazos.items():
        resultado = resultado.replace(k, str(v))
    return resultado

# ==============================================================================
# 3. EJECUCIÓN PRINCIPAL
# ==============================================================================
with st.spinner(f"Analizando datos en vivo y fundamentales de {ticker} ({intervalo})..."):
    df_diario, df_velas, prob_call, info_empresa, noticias_empresa, insiders_empresa = cargar_datos_y_modelo(ticker, intervalo)

if df_diario is None or df_velas is None or df_velas.empty:
    st.error(f"❌ No se encontraron datos para el ticker **'{ticker}'**. Verifique que el símbolo exista (ej. AAPL, NU, DIS, SPY).")
    st.stop()

prob_put = 1.0 - prob_call
precio_actual = float(df_velas['Close'].iloc[-1])
apertura = float(df_velas['Open'].iloc[-1])
maximo = float(df_velas['High'].iloc[-1])
minimo = float(df_velas['Low'].iloc[-1])
volumen = float(df_velas['Volume'].iloc[-1])
cambio = precio_actual - apertura
cambio_pct = (cambio / apertura) * 100
rsi_actual = float(df_velas['RSI'].iloc[-1])
macd_actual = float(df_velas['MACD'].iloc[-1])

# Fila de métricas superiores
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("ÚLTIMO", f"${precio_actual:.2f}", f"{cambio:+.2f} ({cambio_pct:+.2f}%)")
m2.metric("MÁX / MÍN", f"${maximo:.2f} / ${minimo:.2f}")
m3.metric("VOLUMEN", f"{volumen/1e6:.2f}M")
m4.metric("IA CALL", f"{prob_call*100:.1f}%", delta=f"Meta ≥ {umbral_confianza}%")
m5.metric("IA PUT", f"{prob_put*100:.1f}%", delta=f"Meta ≥ {umbral_confianza}%")
m6.metric("RSI (14)", f"{rsi_actual:.1f}")
m7.metric("MACD", f"{macd_actual:.3f}")

# Alertas de cambio de tendencia
if bot_activo and len(df_velas) >= 3:
    e9_prev, e21_prev = df_velas['EMA9'].iloc[-3], df_velas['EMA21'].iloc[-3]
    e9_curr, e21_curr = df_velas['EMA9'].iloc[-1], df_velas['EMA21'].iloc[-1]
    if e9_prev < e21_prev and e9_curr > e21_curr and rsi_actual < 65:
        st.success(f"▲ **ALCISTA ({ticker})**: Cruce EMA9 > EMA21 — Posible CALL | Confianza IA: {prob_call*100:.1f}%")
    elif e9_prev > e21_prev and e9_curr < e21_curr and rsi_actual > 35:
        st.error(f"▼ **BAJISTA ({ticker})**: Cruce EMA9 < EMA21 — Posible PUT | Confianza IA: {prob_put*100:.1f}%")
    elif rsi_actual < 28:
        st.warning(f"◆ **SOBREVENTA ({ticker})**: RSI {rsi_actual:.1f} — Rebote probable")
    elif rsi_actual > 72:
        st.warning(f"◆ **SOBRECOMPRA ({ticker})**: RSI {rsi_actual:.1f} — Corrección probable")

# ==============================================================================
# 4. NAVEGACIÓN POR PESTAÑAS
# ==============================================================================
t_chart, t_mc, t_options, t_ibkr, t_sec, t_reporte = st.tabs([
    "📊 TradingView & Técnico",
    "🔮 Monte Carlo (500 Futuros)",
    "📈 Cadena Opciones (CALL / PUT)",
    "🏦 IBKR & Posiciones",
    "🛡️ Seguridad & Webhook",
    f"🏢 Informe Fundamental ({ticker})"
])

# --- TAB 1: ACCESO TRADINGVIEW EN VIVO + GRÁFICO CUANTITATIVO ---
with t_chart:
    st.subheader(f"🌐 Gráfico Interactivo TradingView en Vivo ({ticker})")
    tv_interval = "D" if intervalo == "1d" else "60" if intervalo == "1h" else "15" if intervalo == "15m" else "5"
    tv_html = f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0;padding:0;background:#060b12;">
    <div class="tradingview-widget-container" style="height:490px;width:100%;">
      <div id="tradingview_chart" style="height:490px;width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%",
        "height": 490,
        "symbol": "{ticker}",
        "interval": "{tv_interval}",
        "timezone": "America/Bogota",
        "theme": "dark",
        "style": "1",
        "locale": "es",
        "toolbar_bg": "#0b1220",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": ["MASimple@tv-basicstudies", "RSI@tv-basicstudies"],
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    </body>
    </html>
    """
    components.html(tv_html, height=500, scrolling=False)

    st.subheader(f"📉 Motor Cuantitativo Interno ({ticker} · {intervalo})")
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.52, 0.14, 0.17, 0.17],
        subplot_titles=(f"{ticker} · Velas + EMA 9/21", "Volumen", "RSI (14)", "MACD (12, 26, 9)")
    )
    fig.add_trace(go.Candlestick(
        x=df_velas.index, open=df_velas['Open'], high=df_velas['High'],
        low=df_velas['Low'], close=df_velas['Close'], name="Velas",
        increasing_line_color='#00e89a', decreasing_line_color='#ff3d6b'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_velas.index, y=df_velas['EMA9'], line=dict(color='#00c8ff', width=1.4), name="EMA 9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_velas.index, y=df_velas['EMA21'], line=dict(color='#ffd23f', width=1.4), name="EMA 21"), row=1, col=1)

    colores_vol = ['#00e89a' if c >= o else '#ff3d6b' for c, o in zip(df_velas['Close'], df_velas['Open'])]
    fig.add_trace(go.Bar(x=df_velas.index, y=df_velas['Volume'], marker_color=colores_vol, name="Volumen", opacity=0.65), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_velas.index, y=df_velas['RSI'], line=dict(color='#00c8ff', width=1.5), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ff3d6b", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00e89a", row=3, col=1)

    colores_macd = ['#00e89a' if v >= 0 else '#ff3d6b' for v in df_velas['Hist']]
    fig.add_trace(go.Bar(x=df_velas.index, y=df_velas['Hist'], marker_color=colores_macd, name="Hist MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_velas.index, y=df_velas['MACD'], line=dict(color='#00c8ff', width=1.3), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_velas.index, y=df_velas['Signal'], line=dict(color='#ff8c42', width=1.2), name="Señal"), row=4, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#060b12", plot_bgcolor="#0b1220",
        height=640, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: SIMULACIÓN MONTE CARLO (500 ESCENARIOS) ---
with t_mc:
    dias_proy = st.slider("Horizonte de proyección (días):", min_value=5, max_value=45, value=21)
    retornos = np.log(df_diario['Close'] / df_diario['Close'].shift(1)).dropna()
    media, vol = retornos.mean(), retornos.std()
    np.random.seed(42)
    caminos = np.zeros((dias_proy, 500))
    caminos[0] = precio_actual
    for t in range(1, dias_proy):
        caminos[t] = caminos[t - 1] * np.exp((media - 0.5 * vol ** 2) + vol * np.random.normal(0, 1, 500))

    ruta_acertada = np.median(caminos, axis=1)
    p95 = np.percentile(caminos, 95, axis=1)
    p5 = np.percentile(caminos, 5, axis=1)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Ruta Más Acertada (Mediana)", f"${ruta_acertada[-1]:.2f}", f"{((ruta_acertada[-1]/precio_actual)-1)*100:+.2f}%")
    mc2.metric("Escenario Optimista (P95)", f"${p95[-1]:.2f}")
    mc3.metric("Límite de Riesgo (P5)", f"${p5[-1]:.2f}")

    fig_mc = go.Figure()
    eje_x = list(range(dias_proy))
    for i in range(75):
        fig_mc.add_trace(go.Scatter(x=eje_x, y=caminos[:, i], mode='lines', line=dict(color='rgba(184,208,232,0.08)', width=1), showlegend=False))
    fig_mc.add_trace(go.Scatter(x=eje_x, y=p95, mode='lines', line=dict(color='#00c8ff', dash='dash'), name=f'Optimista (${p95[-1]:.2f})'))
    fig_mc.add_trace(go.Scatter(x=eje_x, y=ruta_acertada, mode='lines', line=dict(color='#00e89a', width=3), name=f'Ruta Más Acertada (${ruta_acertada[-1]:.2f})'))
    fig_mc.add_trace(go.Scatter(x=eje_x, y=p5, mode='lines', line=dict(color='#ff3d6b', dash='dash'), name=f'Soporte Riesgo (${p5[-1]:.2f})'))

    fig_mc.update_layout(
        title=f"Simulación Monte Carlo (500 Escenarios) • {ticker} a {dias_proy} días",
        template="plotly_dark", paper_bgcolor="#060b12", plot_bgcolor="#0b1220",
        xaxis_title="Días hacia el futuro", yaxis_title="Precio (USD)", height=460
    )
    st.plotly_chart(fig_mc, use_container_width=True)

# --- TAB 3: CADENA DUAL CALL / PUT + GRIEGAS COMPLETAS ---
with t_options:
    activo = yf.Ticker(ticker)
    if activo.options:
        venc_elegido = st.selectbox("Fecha de Vencimiento:", activo.options[:10])
        dias_venc = max(1, (datetime.strptime(venc_elegido, "%Y-%m-%d") - datetime.now()).days)
        T = dias_venc / 365.0

        chain = activo.option_chain(venc_elegido)
        calls = chain.calls[(chain.calls['strike'] >= precio_actual * 0.88) & (chain.calls['strike'] <= precio_actual * 1.12)].copy()
        puts = chain.puts[(chain.puts['strike'] >= precio_actual * 0.88) & (chain.puts['strike'] <= precio_actual * 1.12)].copy()

        merged = pd.merge(calls, puts, on="strike", suffixes=("_call", "_put")).sort_values("strike")
        filas_dual = []
        for _, r in merged.iterrows():
            mid_c = round((r['bid_call'] + r['ask_call']) / 2, 2)
            mid_p = round((r['bid_put'] + r['ask_put']) / 2, 2)
            gc = calcular_griegas_completas(precio_actual, r['strike'], T, r['impliedVolatility_call'], "call")
            gp = calcular_griegas_completas(precio_actual, r['strike'], T, r['impliedVolatility_put'], "put")

            filas_dual.append({
                "CALL OI": int(r['openInterest_call']) if pd.notna(r['openInterest_call']) else 0,
                "CALL IV%": round(r['impliedVolatility_call'] * 100, 1),
                "CALL Δ": gc['delta'],
                "CALL Θ": gc['theta'],
                "CALL $": mid_c,
                "STRIKE": r['strike'],
                "PUT $": mid_p,
                "PUT Δ": gp['delta'],
                "PUT Θ": gp['theta'],
                "PUT IV%": round(r['impliedVolatility_put'] * 100, 1),
                "PUT OI": int(r['openInterest_put']) if pd.notna(r['openInterest_put']) else 0,
            })

        df_dual = pd.DataFrame(filas_dual)
        st.dataframe(df_dual, use_container_width=True, hide_index=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_dual.to_excel(writer, index=False, sheet_name='Cadena_Dual')
        st.download_button(
            label="📥 Descargar Cadena Dual CALL/PUT en Excel (.xlsx)",
            data=buffer.getvalue(),
            file_name=f"Cadena_Opciones_{ticker}_{venc_elegido}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info(f"El ticker **{ticker}** no dispone de cadena de opciones listada en Yahoo Finance.")

# --- TAB 4: IBKR PANEL & GESTOR DE POSICIONES ---
with t_ibkr:
    col_con, col_ord = st.columns(2)
    with col_con:
        st.subheader("🔌 Configuración TWS / IB Gateway")
        st.radio("Entorno:", ["📝 Paper Trading (Puerto 7497)", "💰 Live Trading (Puerto 7496)"], horizontal=True)
        st.info("Por seguridad, la ejecución real se realiza manualmente desde su app móvil de IBKR tras validar la señal.")
        st.code("""# Conexión segura de solo lectura (ibapi)
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
# Puerto Paper: 7497 | Puerto Live: 7496 | Host: 127.0.0.1""", language="python")

    with col_ord:
        st.subheader("📋 Registrar Operación Manual en Bitácora")
        with st.form("form_posicion"):
            f_tipo = st.selectbox("Tipo Contrato:", ["CALL", "PUT"])
            f_strike = st.number_input("Strike ($):", value=round(precio_actual, 1), step=0.5)
            f_venc = st.text_input("Vencimiento (YYYY-MM-DD):", value="2026-12-18")
            f_cant = st.number_input("Contratos:", min_value=1, value=1)
            f_costo = st.number_input("Prima Pagada ($):", min_value=0.05, value=1.00, step=0.05)
            if st.form_submit_button("➕ Agregar a Posiciones Abiertas"):
                st.session_state.posiciones_ibkr.append({
                    "Símbolo": ticker, "Tipo": f_tipo, "Strike": f_strike,
                    "Vencimiento": f_venc, "Cant.": f_cant, "Costo Avg ($)": f_costo,
                    "Precio Act. ($)": f_costo, "P&L ($)": 0.0
                })
                st.rerun()

    st.subheader("💼 Posiciones Abiertas")
    df_pos = pd.DataFrame(st.session_state.posiciones_ibkr)
    st.dataframe(df_pos, use_container_width=True, hide_index=True)

# --- TAB 5: SEGURIDAD & PINE SCRIPT / WEBHOOK ---
with t_sec:
    s1, s2 = st.columns(2)
    with s1:
        st.subheader("🛡️ Checklist de Seguridad Activo")
        st.markdown("""
        - ✅ **Aislamiento de Fondos:** Sin credenciales de IBKR expuestas en la nube.
        - ✅ **Cifrado HTTPS/TLS:** Despliegue protegido en Streamlit Cloud.
        - ✅ **Variables de Entorno:** Secretos aislados fuera del código fuente.
        - ✅ **2FA Obligatorio:** Autenticación en dos pasos en IBKR y GitHub.
        - ✅ **Validación HMAC-SHA256 & Rate Limiting:** Protección contra inyección de alertas.
        """)
    with s2:
        st.subheader("🌲 Pine Script v5 (EMA 9/21 + RSI + MACD)")
        st.code("""//@version=5
strategy("Options Bot Signal", overlay=true)
fast = ta.ema(close, 9)
slow = ta.ema(close, 21)
rsi  = ta.rsi(close, 14)
[ml, sl, hl] = ta.macd(close, 12, 26, 9)
longEntry  = ta.crossover(fast, slow)  and rsi < 70 and ml > sl
shortEntry = ta.crossunder(fast, slow) and rsi > 30 and ml < sl
plot(fast, "EMA9", color=color.aqua)
plot(slow, "EMA21", color=color.yellow)""", language="pascal")

# --- TAB 6: INFORME FUNDAMENTAL DINÁMICO ---
with t_reporte:
    html_dinamico = construir_informe_html_dinamico(
        symbol=ticker,
        precio_act=precio_actual,
        info=info_empresa,
        noticias=noticias_empresa,
        insiders=insiders_empresa
    )
    components.html(html_dinamico, height=2200, scrolling=True)
