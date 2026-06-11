import yfinance as yf
import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Stock Sector Performance Dashboard")
st.write("Analyzing live market data across major S&P 500 sectors")

# Define sector ETFs
sectors = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Communication Services": "XLC"
}

period = "6mo"

@st.cache_data
def fetch_sector_data(sectors, period):
    data = {}
    for sector, ticker in sectors.items():
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        data[sector] = df
    return data
def calculate_scores(sector_data):
    scores = []
    for sector, df in sector_data.items():
        close = df["Close"].squeeze().dropna()
        momentum = float((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100)
        volatility = float(close.pct_change().std() * 100)
        vol_adjusted_return = momentum / volatility if volatility != 0 else 0
        scores.append({
            "Sector": sector,
            "Momentum (%)": round(momentum, 2),
            "Volatility (%)": round(volatility, 2),
            "Vol-Adjusted Return": round(vol_adjusted_return, 2)
        })
    return pd.DataFrame(scores)
st.sidebar.header("Filters")

selected_sectors = st.sidebar.multiselect(
    "Select Sectors",
    options=list(sectors.keys()),
    default=list(sectors.keys())
)

selected_period = st.sidebar.selectbox(
    "Select Time Period",
    options=["1mo", "3mo", "6mo", "1y", "2y"],
    index=2
)

sector_data = fetch_sector_data(sectors, selected_period)
scores_df = calculate_scores(sector_data)
st.subheader("Sector Momentum (%)")
fig = px.bar(
    scores_df,
    x="Sector",
    y="Momentum (%)",
    color="Momentum (%)",
    color_continuous_scale="RdYlGn",
    title="20-Day Momentum by Sector"
)
st.plotly_chart(fig, use_container_width=True)
st.subheader("Sector Volatility (%)")
fig2 = px.bar(
    scores_df,
    x="Sector",
    y="Volatility (%)",
    color="Volatility (%)",
    color_continuous_scale="RdYlGn_r",
    title="Daily Return Volatility by Sector"
)
st.plotly_chart(fig2, use_container_width=True)
st.subheader("Vol-Adjusted Return by Sector")
fig3 = px.bar(
    scores_df,
    x="Sector",
    y="Vol-Adjusted Return",
    color="Vol-Adjusted Return",
    color_continuous_scale="RdYlGn",
    title="Momentum Adjusted for Volatility"
)
st.plotly_chart(fig3, use_container_width=True)