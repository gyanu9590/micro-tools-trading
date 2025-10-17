# file: app_streamlit.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import statsmodels.api as sm          # optional: for OLS summary
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime
def run():

    #st.set_page_config(layout="wide", page_title="Mini Toolbox")

    # -------------------------
    # Sidebar controls
    # -------------------------
    with st.sidebar:
        st.title("Toolbox")
        start = st.date_input("From", value=pd.to_datetime("2024-01-01"))
        end = st.date_input("To", value=pd.to_datetime("today"))
        market = st.selectbox("Market (y)", ["^NSEI"])
        tickers = st.multiselect(
            "Predictors (X)",
            ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS", "CL=F", "INR=X"],
            default=["RELIANCE.NS", "HDFCBANK.NS", "CL=F"],
        )
        model_type = st.selectbox("Model", ["MLR (returns)", "Logistic (direction)"])
        run = st.button("RUN")

    # -------------------------
    # Helper: cached download
    # -------------------------
    @st.cache_data(ttl=60 * 30)
    def download_data(market_ticker, predictor_list, start_date, end_date):
        # Download full OHLC for market AND Close for predictors
        tickers_all = list(set([market_ticker] + predictor_list))
        data = yf.download(tickers_all, start=start_date, end=end_date, progress=False)
        # If data is empty, return None
        if data is None or data.empty:
            return None
        else:
            scaler = StandardScaler()
            data.columns = pd.MultiIndex.from_tuples(data.columns)

        
        # Normalize MultiIndex if single ticker returned differently
            return data

    # -------------------------
    # Utility functions
    # -------------------------
    def prepare_returns(data, market_ticker, predictors):
        # `data` expected to be multiindex columns like ('Close', ticker)
        # Build close-only frame for tickers of interest
        if ("Close", market_ticker) in data.columns:
            close = data["Close"]
        else:
            # Sometimes yfinance returns single-level df for close column selection
            # Try fallback: if "Close" is not present, but columns are tickers directly
            if isinstance(data.columns, pd.Index) and market_ticker in data.columns:
                close = data
            else:
                raise ValueError("Can't find Close prices in downloaded data.")
        # Build returns (pct_change) on close prices
        closes = close[[market_ticker] + [t for t in predictors if t in close.columns]].dropna()
        returns = closes.pct_change().dropna()
        return closes, returns

    # -------------------------
    # Main run
    # -------------------------
    if run:
        if len(tickers) == 0:
            st.error("Select at least one predictor ticker.")
            st.stop()

        with st.spinner("Downloading data and preparing model..."):
            raw_data = download_data(market, tickers, start, end)

        if raw_data is None or raw_data.empty:
            st.error("No data downloaded. Check tickers or date range.")
            st.stop()

        # Prepare closes and returns
        try:
            closes, returns = prepare_returns(raw_data, market, tickers)
        except Exception as e:
            st.error(f"Failed to prepare returns: {e}")
            st.stop()

        if returns.empty or len(returns) < 20:
            st.warning("Not enough data after dropna. Choose a wider date range.")
            st.stop()

        st.subheader("Data preview (last rows)")
        st.dataframe(closes.tail(5))

        # Choose model path
        if model_type == "Logistic (direction)":
            st.info("Logistic direction model not implemented yet. Showing MLR as default.")
            # Continue with MLR for now (you can implement logistic later)

        # Build X, y
        y = returns[market]
        X = returns[[t for t in tickers if t in returns.columns]]

        # Standardize X
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

        # Fit linear regression
        lr = LinearRegression()
        lr.fit(X_scaled, y)

        # Predictions (same index as y)
        preds = pd.Series(lr.predict(X_scaled), index=y.index, name="predicted_return")

        # Predicted price: use previous close (align)
        # For predicted return at day t, predicted price ~ close_{t-1} * (1 + pred_t)
        prev_close = closes[market].shift(1).reindex(preds.index)
        pred_price = prev_close * (1 + preds)

        # Get market OHLC for plotting candlestick if available
        # raw_data should have multiindex columns like ('Open', ticker)
        try:
            if ("Open", market) in raw_data.columns:
                market_ohlc = raw_data.loc[:, ["Open", "High", "Low", "Close"]][market]
                market_ohlc = market_ohlc.dropna()
            else:
                # fallback: use close as flat OHLC (less ideal)
                market_ohlc = pd.DataFrame(
                    {
                        "Open": closes[market],
                        "High": closes[market],
                        "Low": closes[market],
                        "Close": closes[market],
                    }
                )
        except Exception:
            market_ohlc = pd.DataFrame(
                {
                    "Open": closes[market],
                    "High": closes[market],
                    "Low": closes[market],
                    "Close": closes[market],
                }
            )

        # -------------------------
        # Plot: Candlestick with Predicted Price
        # -------------------------
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=market_ohlc.index,
                open=market_ohlc["Open"],
                high=market_ohlc["High"],
                low=market_ohlc["Low"],
                close=market_ohlc["Close"],
                name="NIFTY (OHLC)",
            )
        )

        # Plot predicted price aligned (drop NA)
        fig.add_trace(
            go.Scatter(
                x=pred_price.dropna().index,
                y=pred_price.dropna().values,
                mode="lines",
                name="Predicted Price (approx)",
            )
        )
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # Latest prediction & signal
        # -------------------------
        latest_pred = preds.iloc[-1]
        latest_pred_pct = latest_pred * 100.0
        st.metric("Predicted return (latest)", f"{latest_pred_pct:.4f} %")

        # Signal thresholds (example: 0.1% threshold)
        thr_pct = 0.1
        if latest_pred_pct > thr_pct:
            st.success("STRONG BUY")
        elif latest_pred_pct > 0:
            st.info("BUY")
        elif latest_pred_pct < -thr_pct:
            st.error("STRONG SELL")
        elif latest_pred_pct < 0:
            st.info("SELL")
        else:
            st.info("HOLD")

        # -------------------------
        # Model diagnostics
        # -------------------------
        st.subheader("Model diagnostics")

        # Coefficients correspond to standardized features (because of StandardScaler)
        coefs = pd.Series(lr.coef_, index=X_scaled.columns)
        st.write("Note: coefficients shown are for scaled features (StandardScaler).")
        st.dataframe(pd.concat([coefs.rename("coef")], axis=1))

        st.write("Intercept:", float(lr.intercept_))
        st.write("R² on training data:", float(lr.score(X_scaled, y)))

        # Plot Predicted vs Actual
        st.subheader("Predicted vs Actual Returns")
        df_compare = pd.DataFrame({"Actual": y, "Predicted": preds})
        st.line_chart(df_compare)

        # Optional: show statsmodels OLS summary (unscaled X for interpretability)
        try:
            show_ols = st.checkbox("Show OLS summary (unscaled X, for stats) - slower")
            if show_ols:
                X_ols = sm.add_constant(X)  # use raw returns (not scaled) for OLS
                ols_model = sm.OLS(y, X_ols).fit()
                st.text(ols_model.summary().as_text())
        except Exception as e:
            st.warning("Could not compute OLS summary: " + str(e))

        st.success("Run complete.")
