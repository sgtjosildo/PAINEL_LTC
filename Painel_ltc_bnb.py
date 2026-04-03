import streamlit as st
import pandas as pd
import statsmodels.api as sm
from binance.client import Client
import plotly.express as px
import time

# 🔑 Substitua pelas suas chaves da Binance
client = Client(api_key='SUA_API_KEY', api_secret='SUA_API_SECRET')

st.title("📊 Painel de Cointegração LTC/BNB em Tempo Real")

ltc_prices, bnb_prices = [], []
placeholder = st.empty()

while True:
    ltc = float(client.get_symbol_ticker(symbol='LTCUSDT')['price'])
    bnb = float(client.get_symbol_ticker(symbol='BNBUSDT')['price'])
    ltc_prices.append(ltc)
    bnb_prices.append(bnb)

    # Teste de cointegração
    if len(ltc_prices) > 20:
        p_value = sm.tsa.stattools.coint(ltc_prices, bnb_prices)[1]
    else:
        p_value = None

    # Gráfico
    df = pd.DataFrame({'LTC': ltc_prices, 'BNB': bnb_prices})
    fig = px.line(df, title="Cotação LTC vs BNB")

    with placeholder.container():
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Último preço LTC: {ltc:.2f} USDT")
        st.write(f"Último preço BNB: {bnb:.2f} USDT")

        if p_value:
            st.write(f"📈 p-valor da cointegração: {p_value:.4f}")

            # Alertas visuais
            if p_value < 0.05:
                st.success("✅ Cointegração forte detectada — possível oportunidade de arbitragem!")
            elif p_value < 0.1:
                st.warning("⚠️ Cointegração moderada — acompanhe de perto.")
            else:
                st.error("❌ Sem evidência de cointegração significativa no momento.")

    time.sleep(30)  # atualiza a cada 30 segundos