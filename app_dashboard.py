import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime
import io

# Configuración de página adaptable a Celular, iPad y PC
st.set_page_config(page_title="Centro de Control Cuantitativo", page_icon="📈", layout="wide")

st.title("📈 Centro de Análisis de Opciones y Probabilidad")
st.caption("Sistema Cuantitativo • Simulación de 500 Escenarios y Griegas Black-Scholes")

# --- BARRA LATERAL (CONTROLES TÁCTILES) ---
st.sidebar.header("⚙️ Parámetros de Búsqueda")
ticker = st.sidebar.selectbox("Seleccione Acción:", ["NU", "AAPL", "NVDA", "AMD", "PLTR", "SOFI", "SPY"])
presupuesto_max = st.sidebar.number_input("Presupuesto Máximo por Contrato (USD):", min_value=50, max_value=5000, value=300, step=50)
tipo_operacion = st.sidebar.radio("Dirección a Evaluar:", ["CALL (Alcista)", "PUT (Bajista)"])

# --- FUNCIONES MATEMÁTICAS ---
def calcular_griegas(S, K, T, sigma, tipo="call", r=0.045):
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "theta": 0, "vega": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == "call":
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return {"delta": round(delta, 3), "theta": round(theta, 3), "vega": round(vega, 3)}

@st.cache_data(ttl=900)  # Guarda en memoria 15 min para cargar instantáneo en iPad/Celular
def analizar_accion(symbol):
    activo = yf.Ticker(symbol)
    df = activo.history(period="2y", interval="1d")
    df['Ret_1d'] = df['Close'].pct_change()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['Dist_EMA21'] = (df['Close'] - df['EMA_21']) / df['EMA_21']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    df['Vol_20d'] = df['Ret_1d'].rolling(20).std()
    df['Vol_Rel'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['Target'] = (df['Close'].shift(-5) / df['Close'] - 1 >= 0.02).astype(int)
    
    features = ['Ret_1d', 'Dist_EMA21', 'RSI', 'Vol_20d', 'Vol_Rel']
    df_clean = df.dropna().copy()
    corte = int(len(df_clean) * 0.8)
    
    modelo = CalibratedClassifierCV(RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42), method='isotonic', cv=3)
    modelo.fit(df_clean[features].iloc[:corte], df_clean['Target'].iloc[:corte])
    
    prob_call = modelo.predict_proba(df[features].iloc[[-1]].dropna())[:, 1][0]
    return df, prob_call

# --- EJECUCIÓN VISUAL ---
with st.spinner(f"Analizando {ticker} y simulando 500 escenarios futuros..."):
    df, prob_call = analizar_accion(ticker)
    precio_actual = float(df['Close'].iloc[-1])
    rsi_actual = float(df['RSI'].iloc[-1])
    prob_mostrar = prob_call if "CALL" in tipo_operacion else (1 - prob_call)

    # Tarjetas superiores de métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Precio Actual", f"${precio_actual:.2f}")
    col2.metric("Probabilidad del Modelo", f"{prob_mostrar*100:.2f}%", delta="Meta: >= 97%")
    col3.metric("RSI (14 días)", f"{rsi_actual:.1f}")

    # Buscar mejor contrato
    activo = yf.Ticker(ticker)
    hoy = datetime.now()
    venc_elegido, dias_venc = None, 30
    for v in activo.options:
        d = (datetime.strptime(v, "%Y-%m-%d") - hoy).days
        if 21 <= d <= 45:
            venc_elegido, dias_venc = v, d
            break

    mejor_op = None
    if venc_elegido:
        tipo_str = "call" if "CALL" in tipo_operacion else "put"
        cadena = activo.option_chain(venc_elegido).calls if tipo_str == "call" else activo.option_chain(venc_elegido).puts
        T = dias_venc / 365.0
        mejor_score = -999

        for _, row in cadena.iterrows():
            if row['openInterest'] < 100 or row['ask'] <= 0 or row['bid'] <= 0:
                continue
            mid = round((row['bid'] + row['ask']) / 2, 2)
            if mid * 100 > presupuesto_max:
                continue
            g = calcular_griegas(precio_actual, row['strike'], T, row['impliedVolatility'], tipo=tipo_str)
            if 0.40 <= abs(g['delta']) <= 0.65:
                score = (abs(g['delta']) * 100) - (abs(g['theta']) * 50)
                if score > mejor_score:
                    mejor_score = score
                    mejor_op = {
                        "Contrato": row['contractSymbol'], "Strike": row['strike'],
                        "Vencimiento": venc_elegido, "Días": dias_venc,
                        "Entrada Mid": mid, "Costo Total": round(mid * 100, 0),
                        "Delta": g['delta'], "Theta": g['theta'], "Vega": g['vega']
                    }

    # --- GRÁFICO DE 500 ESCENARIOS (MONTE CARLO) ---
    st.subheader(f"🔮 Proyección de 500 Futuros Posibles ({dias_venc} días)")
    retornos = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    media, vol = retornos.mean(), retornos.std()
    np.random.seed(42)
    caminos = np.zeros((dias_venc, 500))
    caminos[0] = precio_actual
    for t in range(1, dias_venc):
        caminos[t] = caminos[t - 1] * np.exp((media - 0.5 * vol ** 2) + vol * np.random.normal(0, 1, 500))

    ruta_acertada = np.median(caminos, axis=1)
    p95 = np.percentile(caminos, 95, axis=1)
    p5 = np.percentile(caminos, 5, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    eje_x = np.arange(0, dias_venc)
    ax.plot(eje_x, caminos, color='#B0BEC5', alpha=0.12, linewidth=0.8)
    ax.fill_between(eje_x, p5, p95, color='#1F4E78', alpha=0.15, label='Zona Probable (90%)')
    ax.plot(eje_x, ruta_acertada, color='#00C853', linewidth=2.5, label=f'Ruta Más Acertada (${ruta_acertada[-1]:.2f})')
    if mejor_op:
        ax.axhline(y=mejor_op["Strike"], color='#FF5252', linestyle='--', linewidth=1.8, label=f'Strike Óptimo (${mejor_op["Strike"]})')
    ax.set_ylabel("Precio (USD)")
    ax.set_xlabel("Días hacia el futuro")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.5)
    st.pyplot(fig)

    # --- DETALLE DEL CONTRATO Y DESCARGA DE EXCEL ---
    st.subheader("📑 Contrato de Opciones Recomendado y Griegas")
    if mejor_op:
        df_op = pd.DataFrame([mejor_op])
        st.dataframe(df_op, use_container_width=True, hide_index=True)

        # Botón de descarga directa de Excel en iPad / Celular / PC
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_op.to_excel(writer, index=False, sheet_name='Operacion_Sugerida')
        st.download_button(
            label="📥 Descargar Reporte en Excel (.xlsx)",
            data=buffer.getvalue(),
            file_name=f"Analisis_{ticker}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning(f"No se encontraron contratos líquidos dentro de su presupuesto de ${presupuesto_max} USD.")
