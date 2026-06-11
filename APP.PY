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
@st.cache_data
def fetch_sector_data(sectors, period):
    data = {}
    for sector, ticker in sectors.items():
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        data[sector] = df
    return data

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

st.subheader("Sector Overview")
for sector in selected_sectors:
    df = sector_data.get(sector)
    st.write(f"**{sector}**")
    st.write(df.tail())
