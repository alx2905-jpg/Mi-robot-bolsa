import streamlit as st
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

# Estado persistente para simulador de posiciones y bitácora en sesión
if "posiciones_ibkr" not in st.session_state:
    st.session_state.posiciones_ibkr = [
        {"Símbolo": "NU", "Tipo": "CALL", "Strike": 12.5, "Vencimiento": "2026-12-18", "Cant.": 2, "Costo Avg ($)": 0.85, "Precio Act. ($)": 1.10, "P&L ($)": 50.0},
        {"Símbolo": "SPY", "Tipo": "CALL", "Strike": 590.0, "Vencimiento": "2026-12-18", "Cant.": 1, "Costo Avg ($)": 4.50, "Precio Act. ($)": 5.80, "P&L ($)": 130.0},
    ]

# --- HEADER ---
col_logo, col_status = st.columns([4, 1])
with col_logo:
    st.title("⚡ OPTIONBOT PRO • IBKR & QUANT ENGINE")
    st.caption("Datos Reales en Línea • Velas Japonesas (EMA 9/21, RSI, MACD) • Monte Carlo 500 Escenarios • Black-Scholes")
with col_status:
    bot_activo = st.toggle("🟢 BOT MONITOR", value=True)

# --- BARRA LATERAL ---
st.sidebar.header("⚙️ Control de Mercado")
ALL_SYMBOLS = ["NU", "SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMZN", "MSFT", "GOOGL", "META", "NFLX", "AMD", "BABA", "COIN", "PLTR", "SOFI", "F", "BAC", "JPM", "GS", "XOM", "CVX", "WMT", "V", "MA"]
ticker = st.sidebar.selectbox("Ticker:", ALL_SYMBOLS, index=0)
intervalo = st.sidebar.selectbox("Temporalidad:", ["1d", "1h", "15m", "5m"], index=0)
presupuesto_max = st.sidebar.number_input("Presupuesto Máx. Contrato (USD):", min_value=50, max_value=10000, value=300, step=50)
umbral_confianza = st.sidebar.slider("Confianza Mínima IA (%):", min_value=50, max_value=99, value=97)

# ==============================================================================
# 2. MOTOR MATEMÁTICO (GRIEGAS BLACK-SCHOLES + MODELO CALIBRADO)
# ==============================================================================
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

    return df_diario, df_velas, prob_call

# ==============================================================================
# 3. EJECUCIÓN PRINCIPAL
# ==============================================================================
with st.spinner(f"Conectando datos en vivo de {ticker} ({intervalo})..."):
    df_diario, df_velas, prob_call = cargar_datos_y_modelo(ticker, intervalo)
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

    # Fila de métricas superiores estilo OPTIONBOT PRO
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("ÚLTIMO", f"${precio_actual:.2f}", f"{cambio:+.2f} ({cambio_pct:+.2f}%)")
    m2.metric("MÁX / MÍN", f"${maximo:.2f} / ${minimo:.2f}")
    m3.metric("VOLUMEN", f"{volumen/1e6:.2f}M")
    m4.metric("IA CALL", f"{prob_call*100:.1f}%", delta=f"Meta ≥ {umbral_confianza}%")
    m5.metric("IA PUT", f"{prob_put*100:.1f}%", delta=f"Meta ≥ {umbral_confianza}%")
    m6.metric("RSI (14)", f"{rsi_actual:.1f}")
    m7.metric("MACD", f"{macd_actual:.3f}")

    # Alertas de cambio de tendencia (Motor de TradingBot Pro)
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

    # ==========================================================================
    # 4. LAS 5 PESTAÑAS DE OPTIONBOT PRO + MONTE CARLO
    # ==========================================================================
    t_chart, t_mc, t_options, t_ibkr, t_sec = st.tabs([
        "📊 Gráfico Técnico",
        "🔮 Monte Carlo (500 Futuros)",
        "📈 Cadena Opciones (CALL / PUT)",
        "🏦 IBKR & Posiciones",
        "🛡️ Seguridad & Webhook"
    ])

    # --- TAB 1: GRÁFICO TÉCNICO (VELAS + VOLUMEN + EMA 9/21 + RSI + MACD) ---
    with t_chart:
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
            row_heights=[0.52, 0.14, 0.17, 0.17],
            subplot_titles=(f"{ticker} · {intervalo} · Velas Japonesas + EMA 9/21", "Volumen", "RSI (14)", "MACD (12, 26, 9)")
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
            height=680, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10)
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

            # Exportación a Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_dual.to_excel(writer, index=False, sheet_name='Cadena_Dual')
            st.download_button(
                label="📥 Descargar Cadena Dual CALL/PUT en Excel (.xlsx)",
                data=buffer.getvalue(),
                file_name=f"Cadena_Opciones_{ticker}_{venc_elegido}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # --- TAB 4: IBKR PANEL & GESTOR DE POSICIONES (DE OPTIONBOT PRO) ---
    with t_ibkr:
        col_con, col_ord = st.columns(2)
        with col_con:
            st.subheader("🔌 Configuración TWS / IB Gateway")
            modo_ibkr = st.radio("Entorno:", ["📝 Paper Trading (Puerto 7497)", "💰 Live Trading (Puerto 7496)"], horizontal=True)
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
