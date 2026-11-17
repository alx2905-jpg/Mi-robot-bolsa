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
    .stTabs [data-baseweb="tab-list"] { background-color: #0b1220; border-bottom: 1px solid #162236; }
    .stTabs [data-baseweb="tab"] { color: #3a5570; font-family: 'Courier New', monospace; }
    .stTabs [aria-selected="true"] { color: #00c8ff !important; border-bottom: 2px solid #00c8ff !important; }
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
    st.caption("TradingView Live • Velas Japonesas (EMA 9/21, RSI, MACD) • Monte Carlo 500 Escenarios • Análisis Fundamental Dinámico")
with col_status:
    bot_activo = st.toggle("🟢 BOT MONITOR", value=True)

# --- BARRA LATERAL: SELECTOR LIBRE DE CUALQUIER TICKER ---
st.sidebar.header("⚙️ Selección de Activo")
ticker_input = st.sidebar.text_input("Escriba cualquier Ticker (ej. NU, DIS, TSLA, SPY):", value="NU").strip().upper()
rapidos = st.sidebar.selectbox("O elija acceso rápido:", ["Usar ticker escrito arriba", "NU", "DIS", "SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMZN", "META", "AMD", "PLTR", "MELI"])
ticker = ticker_input if rapidos == "Usar ticker escrito arriba" else rapidos
if not ticker:
    ticker = "NU"

intervalo = st.sidebar.selectbox("Temporalidad:", ["1d", "1h", "15m", "5m"], index=0)
presupuesto_max = st.sidebar.number_input("Presupuesto Máx. Contrato (USD):", min_value=50, max_value=10000, value=300, step=50)
umbral_confianza = st.sidebar.slider("Confianza Mínima IA (%):", min_value=50, max_value=99, value=97)

# ==============================================================================
# 2. MOTOR MATEMÁTICO Y GENERADOR FUNDAMENTAL DINÁMICO
# ==============================================================================
def fmt_num(val, prefix="$", suffix=""):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/D"
    abs_v = abs(val)
    if abs_v >= 1e12:
        return f"{prefix}{val/1e12:.2f}T{suffix}"
    if abs_v >= 1e9:
        return f"{prefix}{val/1e9:.2f}B{suffix}"
    if abs_v >= 1e6:
        return f"{prefix}{val/1e6:.1f}M{suffix}"
    return f"{prefix}{val:,.2f}{suffix}"

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
        return None, None, None, {}, [], None

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
    df_clean = df_diario.dropna().copy()
    corte = int(len(df_clean) * 0.8)
    modelo = CalibratedClassifierCV(RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42), method='isotonic', cv=3)
    modelo.fit(df_clean[features].iloc[:corte], df_clean['Target'].iloc[:corte])
    prob_call = modelo.predict_proba(df_diario[features].iloc[[-1]].dropna())[:, 1][0]

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

    # Extraer datos fundamentales reales y noticias del ticker
    try:
        info = activo.info or {}
    except Exception:
        info = {}
    try:
        noticias = activo.news[:3] if activo.news else []
    except Exception:
        noticias = []
    try:
        insiders = activo.insider_transactions.head(4) if activo.insider_transactions is not None else None
    except Exception:
        insiders = None

    return df_diario, df_velas, prob_call, info, noticias, insiders

def construir_informe_html_dinamico(symbol, precio_act, info, noticias, insiders):
    nombre_empresa = info.get("longName") or info.get("shortName") or symbol
    exchange = info.get("exchange", "NYSE / NASDAQ")
    sector = info.get("sector", "Mercado Financiero / Renta Variable")
    industria = info.get("industry", "Activo Cotizado")
    website = info.get("website", "")
    dominio = website.replace("https://", "").replace("http://", "").split("/")[0] if website else f"{symbol.lower()}.com"
    resumen = info.get("longBusinessSummary") or f"{nombre_empresa} ({symbol}) es una compañía líder cotizada en {exchange} dentro del sector {sector} ({industria})."

    mcap = fmt_num(info.get("marketCap"))
    rev_growth = info.get("revenueGrowth")
    rev_growth_str = f"{rev_growth*100:+.1f}%" if isinstance(rev_growth, (int, float)) else "N/D"
    rev_growth_cls = "positive" if isinstance(rev_growth, (int, float)) and rev_growth >= 0 else "negative"

    total_rev = fmt_num(info.get("totalRevenue"))
    net_income = fmt_num(info.get("netIncomeToCommon"))
    roe = info.get("returnOnEquity")
    roe_str = f"{roe*100:.1f}%" if isinstance(roe, (int, float)) else "N/D"
    fwd_pe = f"{info.get('forwardPE'):.1f}x" if isinstance(info.get("forwardPE"), (int, float)) else "N/D"
    trail_pe = f"{info.get('trailingPE'):.1f}x" if isinstance(info.get("trailingPE"), (int, float)) else "N/D"
    eps_trail = f"${info.get('trailingEps'):.2f}" if isinstance(info.get("trailingEps"), (int, float)) else "N/D"
    eps_fwd = f"${info.get('forwardEps'):.2f}" if isinstance(info.get("forwardEps"), (int, float)) else "N/D"
    target_mean = f"${info.get('targetMeanPrice'):.2f}" if isinstance(info.get("targetMeanPrice"), (int, float)) else "N/D"
    target_high = f"${info.get('targetHighPrice'):.2f}" if isinstance(info.get("targetHighPrice"), (int, float)) else "N/D"
    target_low = f"${info.get('targetLowPrice'):.2f}" if isinstance(info.get("targetLowPrice"), (int, float)) else "N/D"
    rec_key = (info.get("recommendationKey") or "HOLD").upper().replace("_", " ")

    high_52 = info.get("fiftyTwoWeekHigh") or precio_act
    low_52 = info.get("fiftyTwoWeekLow") or precio_act
    dist_ath = ((precio_act - high_52) / high_52) * 100 if high_52 else 0.0
    dist_cls = "positive" if dist_ath >= -5 else "negative"

    profit_margin = f"{info.get('profitMargins')*100:.1f}%" if isinstance(info.get("profitMargins"), (int, float)) else "N/D"
    op_margin = f"{info.get('operatingMargins')*100:.1f}%" if isinstance(info.get("operatingMargins"), (int, float)) else "N/D"
    total_cash = fmt_num(info.get("totalCash"))
    total_debt = fmt_num(info.get("totalDebt"))
    empleados = f"{info.get('fullTimeEmployees'):,}" if isinstance(info.get("fullTimeEmployees"), int) else "N/D"
    pais = info.get("country", "Global")

    # Tarjetas de noticias dinámicas
    news_cards_html = ""
    if noticias:
        for n in noticias:
            content = n.get("content", n)
            titulo_n = content.get("title", "Actualización de mercado")
            resumen_n = content.get("summary") or content.get("description") or "Reporte reciente relevante para la cotización y expectativas del activo."
            prov = content.get("provider", {}).get("displayName", "Mercado")
            link_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
            url_n = link_obj.get("url") if isinstance(link_obj, dict) else f"https://finance.yahoo.com/quote/{symbol}"
            news_cards_html += f"""
            <div class="news-card">
              <div>
                <div class="news-date">{prov} <span class="news-tag-res">• NOTICIA EN VIVO</span></div>
                <h3>{titulo_n}</h3>
                <p>{resumen_n[:220]}...</p>
              </div>
              <a href="{url_n}" target="_blank" class="news-link">Leer noticia 🡪</a>
            </div>
            """
    else:
        news_cards_html = f"""
        <div class="news-card">
          <div>
            <div class="news-date">MERCADO • {symbol}</div>
            <h3>Seguimiento corporativo y reportes trimestrales de {nombre_empresa}</h3>
            <p>Consulte los comunicados oficiales de relación con inversores y presentaciones SEC para {symbol}.</p>
          </div>
          <a href="https://finance.yahoo.com/quote/{symbol}/news" target="_blank" class="news-link">Ver noticias en Yahoo Finance 🡪</a>
        </div>
        """

    # Insider trading dinámico
    insider_html = ""
    if insiders is not None and not insiders.empty:
        for _, row in insiders.iterrows():
            nombre_ins = str(row.get("Insider", "Directivo"))
            trans = str(row.get("Text", row.get("Transaction", "Operación registrada")))
            shares = row.get("Shares", 0)
            val_ins = fmt_num(row.get("Value"))
            tag_cls = "tag-sell" if "Sale" in trans or "Sell" in trans else "tag-buy" if "Purchase" in trans or "Buy" in trans else "tag-neutral"
            tag_txt = "VENTA" if tag_cls == "tag-sell" else "COMPRA" if tag_cls == "tag-buy" else "REGISTRO"
            insider_html += f"""
            <div class="flow-item">
              <div class="flow-item-top">
                <span class="flow-name">{nombre_ins}</span>
                <span class="{tag_cls}">{tag_txt}</span>
              </div>
              <div class="flow-details">{trans} • Acciones: {shares:,} ({val_ins})</div>
            </div>
            """
    else:
        insider_html = f"""
        <div class="flow-empty">
          No se reportan transacciones recientes de directivos (Form 4) en los últimos días para <b>{symbol}</b>.
        </div>
        """

    fecha_hoy = datetime.now().strftime("%d de %B de %Y")

    return f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {{
      --navy: #11052C;
      --accent: #820AD1;
      --accent-soft: #F3E8FF;
      --accent-hover: #6804A8;
      --light: #F7F7F7;
      --card-bg: #FFFFFF;
      --text-dark: #1A1A1A;
      --text-muted: #666666;
      --green: #00875A;
      --green-bg: #E3FCEF;
      --red: #E53E3E;
      --red-bg: #FFEBE6;
      --gray-tag: #718096;
      --gray-tag-bg: #EDF2F7;
      --border-color: #E2E8F0;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Calibri, Arial, sans-serif;
      background-color: var(--light);
      color: var(--text-dark);
      line-height: 1.5;
      padding-bottom: 60px;
    }}
    .hero {{
      background-color: var(--navy);
      color: #FFFFFF;
      border-left: 14px solid var(--accent);
      padding: 32px 26px;
      margin-bottom: 22px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }}
    .hero-container {{ max-width: 1200px; margin: 0 auto; }}
    .hero-header {{ display: flex; align-items: center; gap: 20px; margin-bottom: 20px; }}
    .logo-box {{
      width: 90px; height: 90px; background-color: #FFFFFF; border-radius: 16px;
      display: flex; align-items: center; justify-content: center; flex-shrink: 0;
      box-shadow: 0 4px 12px rgba(0,0,0,0.2); padding: 10px; font-weight: 800; color: var(--navy); font-size: 22px;
    }}
    .logo-box img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
    .hero-title-area {{ flex-grow: 1; }}
    .hero-title-row {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
    .hero-title-area h1 {{ font-family: Georgia, serif; font-size: 36px; font-weight: 700; line-height: 1.1; color: #FFFFFF; }}
    .ticker-badge {{
      border: 1.5px solid var(--accent); color: var(--accent-soft); padding: 4px 14px;
      border-radius: 20px; font-size: 15px; font-weight: 700; letter-spacing: 1px; background: rgba(130, 10, 209, 0.2);
    }}
    .hero-subtitle {{ color: #A0AEC0; font-size: 15px; margin-top: 6px; }}
    .stat-strip {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-top: 18px; }}
    .stat-card {{
      background-color: #FFFFFF; color: var(--text-dark); padding: 14px; border-radius: 10px;
      border-left: 4px solid var(--accent); box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    .stat-label {{ font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; margin-bottom: 5px; }}
    .stat-value {{ font-size: 21px; font-weight: 700; color: var(--navy); }}
    .stat-sub {{ font-size: 12px; margin-top: 4px; font-weight: 600; }}
    .stat-sub.positive {{ color: var(--green); }}
    .stat-sub.negative {{ color: var(--red); }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 0 18px; }}
    .dragsec {{
      background: var(--card-bg); border-radius: 12px; padding: 22px; margin-bottom: 22px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.04); border: 1px solid var(--border-color);
    }}
    .dragsec-head {{
      display: flex; align-items: center; justify-content: space-between;
      border-bottom: 2px solid var(--light); padding-bottom: 12px; margin-bottom: 18px;
    }}
    .dragsec-head h2 {{ font-family: Georgia, serif; font-size: 22px; color: var(--navy); font-weight: 700; }}
    .drag-handle {{
      background: var(--light); border: 1px solid var(--border-color); border-radius: 6px;
      padding: 5px 10px; cursor: grab; font-size: 12px; color: var(--text-muted);
    }}
    .about-card {{ font-size: 15px; line-height: 1.7; color: #2D3748; }}
    .highlight {{ color: var(--accent); font-weight: 700; background: var(--accent-soft); padding: 2px 6px; border-radius: 4px; }}
    .grid-2x2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
    .comp-card {{ background: var(--light); border-left: 6px solid var(--accent); padding: 16px; border-radius: 8px; }}
    .comp-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
    .comp-title {{ font-weight: 700; font-size: 15px; color: var(--navy); }}
    .comp-rank {{ background: var(--accent); color: #FFFFFF; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 12px; }}
    .comp-share {{ font-size: 13px; color: var(--accent); font-weight: 600; margin-bottom: 5px; }}
    .comp-desc {{ font-size: 13.5px; color: var(--text-muted); }}
    .fund-top {{
      display: flex; align-items: center; justify-content: space-between;
      background: var(--accent-soft); border-radius: 10px; padding: 18px; margin-bottom: 18px;
    }}
    .mcap-large .val {{ font-family: Georgia, serif; font-size: 32px; font-weight: 700; color: var(--accent); display: block; }}
    .mcap-large .lbl {{ font-size: 12px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }}
    .fund-metrics-summary {{ display: flex; gap: 24px; }}
    .metric-item {{ text-align: right; }}
    .metric-item .v {{ font-size: 18px; font-weight: 700; color: var(--navy); }}
    .metric-item .l {{ font-size: 12px; color: var(--text-muted); }}
    .tables-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .fund-table {{ width: 100%; border-collapse: collapse; border-radius: 8px; overflow: hidden; }}
    .fund-table th {{ padding: 10px 14px; text-align: left; font-size: 13px; color: #FFFFFF; }}
    .fund-table.last th {{ background: var(--navy); }}
    .fund-table.next th {{ background: var(--accent); }}
    .fund-table td {{ padding: 10px 14px; border-bottom: 1px solid var(--border-color); font-size: 13.5px; }}
    .tech-grid {{ display: grid; grid-template-columns: 260px 1fr; gap: 18px; }}
    .ath-box {{
      background: var(--navy); color: #FFFFFF; border-radius: 10px; padding: 20px;
      display: flex; flex-direction: column; justify-content: center; text-align: center;
    }}
    .ath-number {{ font-family: Georgia, serif; font-size: 52px; font-weight: 700; color: #FFFFFF; line-height: 1; margin-bottom: 10px; }}
    .ath-details {{ font-size: 13px; color: #A0AEC0; line-height: 1.6; }}
    .tech-bullets-card {{
      background: #FFFFFF; border-left: 6px solid var(--accent); border-radius: 8px; padding: 18px;
      border: 1px solid var(--border-color);
    }}
    .tech-bullets-card h3 {{ font-family: Georgia, serif; font-size: 17px; color: var(--navy); margin-bottom: 10px; }}
    .tech-bullets-card ul {{ list-style: none; }}
    .tech-bullets-card li {{ position: relative; padding-left: 18px; margin-bottom: 8px; font-size: 13.5px; color: #4A5568; }}
    .tech-bullets-card li::before {{ content: "•"; color: var(--accent); font-size: 18px; position: absolute; left: 0; top: -2px; }}
    .news-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
    .news-card {{
      background: #FFFFFF; border-top: 4px solid var(--accent); border-radius: 8px; padding: 16px;
      border: 1px solid var(--border-color); display: flex; flex-direction: column; justify-content: space-between;
    }}
    .news-date {{ font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 700; margin-bottom: 6px; }}
    .news-tag-res {{ color: var(--accent); font-weight: 700; }}
    .news-card h3 {{ font-family: Georgia, serif; font-size: 15px; color: var(--navy); margin-bottom: 8px; line-height: 1.3; }}
    .news-card p {{ font-size: 13px; color: #4A5568; margin-bottom: 12px; }}
    .news-link {{ font-size: 13px; font-weight: 700; color: var(--accent); text-decoration: none; }}
    .flow-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .flow-card {{ background: #FFFFFF; border: 1px solid var(--border-color); border-radius: 8px; padding: 18px; }}
    .flow-header h3 {{ font-family: Georgia, serif; font-size: 17px; color: var(--navy); }}
    .flow-header .sub {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 12px; }}
    .flow-item {{ padding: 9px; background: var(--light); border-radius: 6px; font-size: 12.5px; margin-bottom: 8px; }}
    .flow-item-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px; }}
    .flow-name {{ font-weight: 700; color: var(--navy); }}
    .tag-sell {{ background: var(--red-bg); color: var(--red); font-weight: 700; padding: 2px 6px; border-radius: 4px; font-size: 10px; }}
    .tag-buy {{ background: var(--green-bg); color: var(--green); font-weight: 700; padding: 2px 6px; border-radius: 4px; font-size: 10px; }}
    .tag-neutral {{ background: var(--gray-tag-bg); color: var(--gray-tag); font-weight: 700; padding: 2px 6px; border-radius: 4px; font-size: 10px; }}
    .flow-empty {{ background: var(--light); border: 1px dashed var(--border-color); border-radius: 6px; padding: 24px; text-align: center; color: var(--text-muted); font-size: 13px; }}
    .footer {{ max-width: 1200px; margin: 20px auto 0 auto; text-align: center; font-size: 12px; color: var(--text-muted); }}
    @media (max-width: 768px) {{
      .stat-strip {{ grid-template-columns: 1fr 1fr; }}
      .grid-2x2, .tables-grid, .tech-grid, .news-grid, .flow-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-container">
      <div class="hero-header">
        <div class="logo-box">
          <img src="https://logo.clearbit.com/{dominio}" alt="{symbol}" onerror="this.style.display='none'; this.parentNode.innerText='{symbol}';">
        </div>
        <div class="hero-title-area">
          <div class="hero-title-row">
            <h1>{nombre_empresa}</h1>
            <span class="ticker-badge">{exchange}: {symbol}</span>
          </div>
          <p class="hero-subtitle">Análisis fundamental y técnico en vivo • {sector} • {fecha_hoy}</p>
        </div>
      </div>

      <div class="stat-strip">
        <div class="stat-card">
          <div class="stat-label">Precio Actual</div>
          <div class="stat-value">${precio_act:.2f}</div>
          <div class="stat-sub positive">Consenso: {rec_key}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Market Cap</div>
          <div class="stat-value">{mcap}</div>
          <div class="stat-sub">P/E Forward: {fwd_pe}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Crecimiento Ventas YoY</div>
          <div class="stat-value">{rev_growth_str}</div>
          <div class="stat-sub {rev_growth_cls}">Ingresos: {total_rev}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Beneficio Neto / ROE</div>
          <div class="stat-value">{net_income}</div>
          <div class="stat-sub positive">ROE: {roe_str}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Distancia a Máx 52s</div>
          <div class="stat-value">{dist_ath:.1f}%</div>
          <div class="stat-sub {dist_cls}">Máx 52s: ${high_52:.2f}</div>
        </div>
      </div>
    </div>
  </header>

  <main class="container">
    <div id="dragzone">

      <section class="dragsec">
        <div class="dragsec-head">
          <h2>¿A qué se dedica {nombre_empresa}?</h2>
          <span class="drag-handle">☰ Perfil Corporativo</span>
        </div>
        <div class="about-card">
          <span class="highlight">{sector} • {industria}</span> — {resumen}
        </div>
      </section>

      <section class="dragsec">
        <div class="dragsec-head">
          <h2>Posición financiera y estructura</h2>
          <span class="drag-handle">☰ Balance</span>
        </div>
        <div class="grid-2x2">
          <div class="comp-card">
            <div class="comp-header">
              <span class="comp-title">Márgenes de Rentabilidad</span>
              <span class="comp-rank">Eficiencia</span>
            </div>
            <div class="comp-share">Margen Neto: {profit_margin} • Operativo: {op_margin}</div>
            <p class="comp-desc">Capacidad de conversión de ingresos ({total_rev}) en utilidad neta ({net_income}) con un retorno sobre el patrimonio (ROE) de {roe_str}.</p>
          </div>
          <div class="comp-card">
            <div class="comp-header">
              <span class="comp-title">Solidez de Caja vs Deuda</span>
              <span class="comp-rank">Balance</span>
            </div>
            <div class="comp-share">Caja Total: {total_cash} • Deuda Total: {total_debt}</div>
            <p class="comp-desc">Posición de liquidez disponible para financiar crecimiento orgánico y operaciones globales ({pais} • {empleados} empleados).</p>
          </div>
        </div>
      </section>

      <section class="dragsec">
        <div class="dragsec-head">
          <h2>Datos fundamentales y valoración</h2>
          <span class="drag-handle">☰ Fundamentales</span>
        </div>
        <div class="fund-top">
          <div class="mcap-large">
            <span class="lbl">Capitalización de Mercado ({symbol})</span>
            <span class="val">{mcap} USD</span>
          </div>
          <div class="fund-metrics-summary">
            <div class="metric-item"><div class="v">{trail_pe}</div><div class="l">Trailing P/E</div></div>
            <div class="metric-item"><div class="v">{fwd_pe}</div><div class="l">Forward P/E</div></div>
            <div class="metric-item"><div class="v">{target_mean}</div><div class="l">Precio Objetivo Analistas</div></div>
          </div>
        </div>

        <div class="tables-grid">
          <table class="fund-table last">
            <thead><tr><th colspan="2">Métricas Actuales (TTM)</th></tr></thead>
            <tbody>
              <tr><td>Ingresos Totales (Revenue)</td><td><b>{total_rev}</b></td></tr>
              <tr><td>Utilidad Neta (Net Income)</td><td><b>{net_income}</b></td></tr>
              <tr><td>EPS (Últimos 12 meses)</td><td><b>{eps_trail}</b></td></tr>
              <tr><td>Retorno sobre Patrimonio (ROE)</td><td><b>{roe_str}</b></td></tr>
            </tbody>
          </table>

          <table class="fund-table next">
            <thead><tr><th colspan="2">Consenso y Proyección de Analistas</th></tr></thead>
            <tbody>
              <tr><td>Recomendación Consenso</td><td><b>{rec_key}</b></td></tr>
              <tr><td>EPS Estimado (Forward)</td><td><b>{eps_fwd}</b></td></tr>
              <tr><td>Precio Objetivo Medio</td><td><b>{target_mean}</b></td></tr>
              <tr><td>Rango Objetivos (Mín – Máx)</td><td><b>{target_low} – {target_high}</b></td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="dragsec">
        <div class="dragsec-head">
          <h2>Análisis técnico — Distancia a máximos 52 semanas</h2>
          <span class="drag-handle">☰ Niveles</span>
        </div>
        <div class="tech-grid">
          <div class="ath-box">
            <div class="ath-number">{dist_ath:.1f}%</div>
            <div class="ath-details">
              Máximo 52 Semanas: <b>${high_52:.2f} USD</b><br>
              Cotización Actual: <b>${precio_act:.2f} USD</b><br>
              Mínimo 52 Semanas: <b>${low_52:.2f} USD</b>
            </div>
          </div>
          <div class="tech-bullets-card">
            <h3>Estructura de Precios ({symbol})</h3>
            <ul>
              <li><b>Rango anual:</b> El activo cotiza entre un soporte anual de <b>${low_52:.2f}</b> y una resistencia máxima de <b>${high_52:.2f}</b>.</li>
              <li><b>Objetivo institucional:</b> El consenso de analistas proyecta un valor razonable medio de <b>{target_mean}</b> (máximo estimado en <b>{target_high}</b>).</li>
              <li><b>Sincronización multi-pestaña:</b> Puede contrastar estos niveles fundamentales con el gráfico interactivo de TradingView (Pestaña 1) y la proyección Monte Carlo de 500 escenarios (Pestaña 2).</li>
            </ul>
          </div>
        </div>
      </section>

      <section class="dragsec">
        <div class="dragsec-head">
          <h2>Últimas noticias ({symbol})</h2>
          <span class="drag-handle">☰ Noticias</span>
        </div>
        <div class="news-grid">
          {news_cards_html}
        </div>
      </section>

      <section class="dragsec">
        <div class="dragsec-head">
          <h2>Insider trading — Movimientos de directivos</h2>
          <span class="drag-handle">☰ SEC Form 4</span>
        </div>
        <div class="flow-grid">
          <div class="flow-card">
            <div class="flow-header">
              <h3>Transacciones recientes de ejecutivos</h3>
              <div class="sub">Registros SEC Form 4 • {symbol}</div>
            </div>
            {insider_html}
          </div>
          <div class="flow-card">
            <div class="flow-header">
              <h3>Resumen de consenso institucional</h3>
              <div class="sub">Valoración y cobertura</div>
            </div>
            <div class="flow-empty">
              Calificación actual del consenso: <b>{rec_key}</b><br><br>
              Precio Objetivo Medio: <b>{target_mean}</b> vs Precio Actual: <b>${precio_act:.2f}</b>
            </div>
          </div>
        </div>
      </section>

    </div>
  </main>

  <footer class="footer">
    Informe Fundamental Dinámico generado en tiempo real para {nombre_empresa} ({symbol}) • Fuentes: Yahoo Finance, SEC Filings & TradingView
  </footer>

  <script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.2/Sortable.min.js"></script>
  <script>
    document.addEventListener('DOMContentLoaded', function() {{
      var el = document.getElementById('dragzone');
      if (el) {{
        Sortable.create(el, {{ handle: '.drag-handle', animation: 200 }});
      }}
    }});
  </script>
</body>
</html>
"""

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
# 4. LAS 6 PESTAÑAS DINÁMICAS
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
    <div class="tradingview-widget-container" style="height:500px;width:100%;">
      <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {{
        "autosize": true,
        "symbol": "{ticker}",
        "interval": "{tv_interval}",
        "timezone": "America/Bogota",
        "theme": "dark",
        "style": "1",
        "locale": "es",
        "allow_symbol_change": true,
        "calendar": false,
        "studies": ["STD;EMA", "STD;RSI"],
        "support_host": "https://www.tradingview.com"
      }}
      </script>
    </div>
    """
    components.html(tv_html, height=510)

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
    fig
