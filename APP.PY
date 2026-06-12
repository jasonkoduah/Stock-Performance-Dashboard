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
sector_stocks = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL"],
    "Healthcare": ["UNH", "JNJ", "LLY", "ABBV", "MRK"],
    "Financials": ["JPM", "BAC", "GS", "WFC", "MS"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
    "Industrials": ["CAT", "BA", "UPS", "HON", "GE"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "Real Estate": ["PLD", "AMT", "EQIX", "SPG", "O"],
    "Materials": ["LIN", "SHW", "FCX", "APD", "NEM"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "TMUS"]
}
period = "6mo"

@st.cache_data
def fetch_sector_data(sectors, period):
    data = {}
    for sector, ticker in sectors.items():
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        data[sector] = df
    return data
@st.cache_data
def fetch_stock_data(tickers, period):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        data[ticker] = df
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
    df_scores = pd.DataFrame(scores)
    
    df_scores["Momentum Score"] = (df_scores["Momentum (%)"] - df_scores["Momentum (%)"].min()) / (df_scores["Momentum (%)"].max() - df_scores["Momentum (%)"].min()) * 100
    df_scores["Volatility Score"] = 100 - (df_scores["Volatility (%)"] - df_scores["Volatility (%)"].min()) / (df_scores["Volatility (%)"].max() - df_scores["Volatility (%)"].min()) * 100
    df_scores["Vol-Adj Score"] = (df_scores["Vol-Adjusted Return"] - df_scores["Vol-Adjusted Return"].min()) / (df_scores["Vol-Adjusted Return"].max() - df_scores["Vol-Adjusted Return"].min()) * 100
    
    df_scores["Composite Score"] = (
        df_scores["Momentum Score"] * 0.4 +
        df_scores["Volatility Score"] * 0.2 +
        df_scores["Vol-Adj Score"] * 0.4
    ).round(2)
    def signal(score):
       if score >= 60:
        return "Strong"
       elif score >= 40:
        return "Neutral"
       else:
        return "Weak"

    df_scores["Signal"] = df_scores["Composite Score"].apply(signal)
    return df_scores
st.sidebar.header("Filters")

selected_sectors = st.sidebar.multiselect(
    "Select Sectors",
    options=list(sectors.keys()),
    default=list(sectors.keys())
)
st.sidebar.header("Stock Drill-Down")
drilldown_sector = st.sidebar.selectbox(
    "Select a sector to view individual stocks",
    options=["None"] + selected_sectors
)


selected_period = st.sidebar.selectbox(
    "Select Time Period",
    options=["1mo", "3mo", "6mo", "1y", "2y"],
    index=2
)
if drilldown_sector != "None":
    st.subheader(f"{drilldown_sector}: Top Stocks")
    stock_data = fetch_stock_data(sector_stocks[drilldown_sector], selected_period)

    for ticker, df in stock_data.items():
        close = df["Close"].squeeze().dropna()
        fig_stock = px.line(
            close,
            title=f"{ticker} - Closing Price"
        )
        st.plotly_chart(fig_stock, use_container_width=True)
    close = df["Close"].squeeze().dropna()
    fig_stock = px.line(
        close,
        title=f"{ticker} - Closing Price"
    )
    st.plotly_chart(fig_stock, use_container_width=True, key=f"stock_chart_{ticker}")
sector_data = fetch_sector_data(sectors, selected_period)
scores_df = calculate_scores(sector_data)
scores_df = scores_df[scores_df["Sector"].isin(selected_sectors)]
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
st.subheader("Composite Score by Sector")
fig4 = px.bar(
    scores_df.sort_values("Composite Score", ascending=False),
    x="Sector",
    y="Composite Score",
    color="Composite Score",
    color_continuous_scale="RdYlGn",
    title="Overall Sector Ranking (Composite Score)"
)
st.plotly_chart(fig4, use_container_width=True)
st.subheader("Sector Signals")
for _, row in scores_df.sort_values("Composite Score", ascending=False).iterrows():
    if row["Signal"] == "Strong":
        st.success(f"**{row['Sector']}**: {row['Signal']} (Score: {row['Composite Score']})")
    elif row["Signal"] == "Neutral":
        st.warning(f"**{row['Sector']}**: {row['Signal']} (Score: {row['Composite Score']})")
    else:
        st.error(f"**{row['Sector']}**: {row['Signal']} (Score: {row['Composite Score']})")

with st.expander("Methodology and Limitations"):
    st.write("""
    **Composite Score Calculation**
    
    Each sector receives a composite score (0-100) based on three factors:
    - Momentum (40%): 20-day percentage price change
    - Volatility-Adjusted Return (40%): Momentum divided by volatility, rewarding consistent gains over erratic ones
    - Volatility (20%): Daily return standard deviation, with lower volatility scoring higher
    
    All factors are normalized to a 0-100 scale before weighting, so no single factor dominates due to differences in units or magnitude.
    
    **Signal Labels**
    
    Strong (60+), Neutral (40-59), and Weak (below 40) reflect each sector's composite score relative to the other sectors currently selected. These are not predictions or investment recommendations.
    
    **Limitations**
    
    - The 20-day window captures short-term momentum, which may reflect temporary conditions rather than durable trends
    - Sector ETFs move with the broader market, so rankings partly reflect overall market direction in addition to sector-specific performance
    - This model has not been backtested for predictive accuracy
    """)  