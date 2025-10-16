# file: app_streamlit.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import statsmodels.api as sm

#st.set_page_config(layout="wide", page_title="Mini Toolbox")

# Sidebar controls
with st.sidebar:
    st.title("Toolbox")
    start = st.date_input("From", value=pd.to_datetime("2024-01-01"))
    end   = st.date_input("To", value=pd.to_datetime("today"))
    market = st.selectbox("Market", ["^NSEI"])
    tickers = st.multiselect("Predictors", ["RELIANCE.NS","HDFCBANK.NS","INFY.NS","SBIN.NS","CL=F","INR=X"], default=["RELIANCE.NS","HDFCBANK.NS","CL=F"])
    model_type = st.selectbox("Model", ["MLR (returns)", "Logistic (direction)"])
    run = st.button("RUN")

if run:
    # fetch
    tickers_all = [market] + tickers
    raw = yf.download(tickers_all, start=start, end=end, progress=False)["Close"].dropna()
    returns = raw.pct_change().dropna()

    # fit simple MLR on returns
    y = returns[market]
    X = returns[tickers]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    # predictions
    preds = model.predict(X)
    pred_price = raw[market].shift(1) * (1 + preds)  # approximate
    
    # Candlestick + overlay predicted line
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=raw.index, open=raw[market], high=raw[market], low=raw[market], close=raw[market],
        name="NIFTY"
    ))
    fig.add_trace(go.Scatter(x=pred_price.index, y=pred_price, mode="lines", name="Predicted"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal card
    latest_ret = preds.iloc[-1]
    st.metric("Predicted Return (%)", f"{latest_ret*100:.3f}")
    if latest_ret > 0.001:
        st.success("BUY")
    elif latest_ret < -0.001:
        st.error("SELL")
    else:
        st.info("HOLD")
    
    # Diagnostics and coefficients
    st.subheader("Model Summary")
    st.write(model.summary())
    st.subheader("Rolling Betas (60d)")
    rolling = returns.rolling(60).apply(lambda df: sm.OLS(df[market], sm.add_constant(df[tickers])).fit().params[1], raw=False)
    st.line_chart(rolling)
