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
st.subheader("Sector Overview")
for sector in selected_sectors:
    df = sector_data[sector]
    st.write(f"**{sector}**")
    st.write(df.tail())
    st.subheader("Sector Scores")
    st.dataframe(scores_df.sort_values("Momentum (%)", ascending=False))