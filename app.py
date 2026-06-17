import yfinance as yf
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Sector Performance Dashboard", layout="wide")

# --- Visual constants ---
ACCENT_COLOR = "#4C72B0"
SIGNAL_COLORS = {"Strong": "#2E8B57", "Neutral": "#E8A33D", "Weak": "#D9534F"}
SIGNAL_ORDER = ["Strong", "Neutral", "Weak"]
PLOTLY_TEMPLATE = "plotly_white"

st.title("Stock Sector Performance Dashboard")
st.write("Composite score blends 20-day momentum and volatility-adjusted return. See Methodology for details.")

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


# --- Sidebar: filters grouped logically ---
st.sidebar.header("Filters")

selected_period = st.sidebar.selectbox(
    "Select Time Period",
    options=["1mo", "3mo", "6mo", "1y", "2y"],
    index=2
)

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


# --- Data loading with status indicator ---
with st.status("Loading sector data...", expanded=False) as status:
    sector_data = fetch_sector_data(sectors, selected_period)
    status.update(label="Calculating sector scores...")
    scores_df_all = calculate_scores(sector_data)
    status.update(label="Data loaded", state="complete")

scores_df = scores_df_all[scores_df_all["Sector"].isin(selected_sectors)]

if scores_df.empty:
    st.warning("Select at least one sector from the sidebar to view the dashboard.")
    st.stop()

scores_df = scores_df.sort_values("Composite Score", ascending=False).reset_index(drop=True)
sector_order = scores_df["Sector"].tolist()

last_date = sector_data[list(sector_data.keys())[0]].index[-1].strftime("%B %d, %Y")
st.caption(f"Data as of {last_date}")


tab_overview, tab_drilldown = st.tabs(["Sector Overview", "Stock Drill-Down"])

with tab_overview:
    top = scores_df.iloc[0]
    bottom = scores_df.iloc[-1]
    avg_score = scores_df["Composite Score"].mean()
    strong_count = int((scores_df["Signal"] == "Strong").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Top sector", f"{top['Sector']} ({top['Composite Score']:.0f})")
    col2.metric("Bottom sector", f"{bottom['Sector']} ({bottom['Composite Score']:.0f})")
    col3.metric("Average composite score", f"{avg_score:.0f}")
    col4.metric("Strong sectors", f"{strong_count} / {len(scores_df)}")

    st.subheader("Composite Score by Sector")
    fig4 = px.bar(
        scores_df,
        x="Sector",
        y="Composite Score",
        color="Signal",
        color_discrete_map=SIGNAL_COLORS,
        category_orders={"Sector": sector_order, "Signal": SIGNAL_ORDER},
        template=PLOTLY_TEMPLATE,
        title="Overall Sector Ranking (Composite Score)"
    )
    fig4.update_xaxes(tickangle=-30)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Supporting Metrics")
    metric_specs = [
        ("Momentum (%)", "20-Day Momentum"),
        ("Volatility (%)", "Daily Return Volatility"),
        ("Vol-Adjusted Return", "Momentum Adjusted for Volatility"),
    ]
    metric_cols = st.columns(3)
    for col, (metric_col, metric_title) in zip(metric_cols, metric_specs):
        fig = px.bar(
            scores_df,
            x="Sector",
            y=metric_col,
            category_orders={"Sector": sector_order},
            template=PLOTLY_TEMPLATE,
            title=metric_title
        )
        fig.update_traces(marker_color=ACCENT_COLOR)
        fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
        fig.update_layout(height=320)
        col.plotly_chart(fig, use_container_width=True)

    with st.expander("Methodology and Limitations"):
        st.write("""
        **Composite Score Calculation**

        Each sector receives a composite score (0-100) based on three factors:
        - Momentum (40%): 20-day percentage price change
        - Volatility-Adjusted Return (40%): Momentum divided by volatility, rewarding consistent gains over erratic ones
        - Volatility (20%): Daily return standard deviation, with lower volatility scoring higher

        All factors are normalized to a 0-100 scale before weighting, so no single factor dominates due to differences in units or magnitude.

        **Signal Labels**

        Strong (60+), Neutral (40-59), and Weak (below 40) reflect each sector's composite score relative to all ten sectors, regardless of which sectors are currently shown by the sector filter. These are not predictions or investment recommendations.

        **Limitations**

        - The 20-day window captures short-term momentum, which may reflect temporary conditions rather than durable trends
        - Sector ETFs move with the broader market, so rankings partly reflect overall market direction in addition to sector-specific performance
        - This model has not been backtested for predictive accuracy
        """)

with tab_drilldown:
    if drilldown_sector != "None":
        st.subheader(f"{drilldown_sector}: Top Stocks")
        with st.spinner(f"Loading {drilldown_sector} stock data..."):
            stock_data = fetch_stock_data(sector_stocks[drilldown_sector], selected_period)

        for ticker, df in stock_data.items():
            close = df["Close"].squeeze().dropna()
            fig_stock = px.line(
                close,
                title=f"{ticker} - Closing Price",
                template=PLOTLY_TEMPLATE
            )
            fig_stock.update_traces(line_color=ACCENT_COLOR)
            st.plotly_chart(fig_stock, use_container_width=True, key=f"stock_chart_{ticker}")
    else:
        st.info("Select a sector from the sidebar under Stock Drill-Down to view individual stock charts.")
