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
st.write("Composite score blends structural trend, volatility, and risk-adjusted momentum. See Methodology for details.")

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


# ==========================================
#  CALCULATION LOGIC (FIXED OVERLAPS)
# ==========================================
def calculate_scores(sector_data):
    scores = []
    for sector, df in sector_data.items():
        close = df["Close"].squeeze().dropna()
        
        # FEATURE 1: Structural Trend Strength (Distance from 200-day SMA)
        sma_span = min(200, len(close))
        sma_200 = close.rolling(window=sma_span).mean().iloc[-1]
        trend_strength = float(((close.iloc[-1] - sma_200) / sma_200) * 100)
        
        # FEATURE 2: Smoothed 20-day Momentum (Reduces single-day noise)
        sma_20 = close.rolling(window=20).mean()
        momentum_20d = float(((sma_20.iloc[-1] - sma_20.iloc[-20]) / sma_20.iloc[-20]) * 100)
        
        # FEATURE 3: Asset Volatility
        volatility = float(close.pct_change().std() * 100)
        
        # FEATURE 4: Risk-Adjusted Return Component
        vol_adjusted_return = momentum_20d / volatility if volatility != 0 else 0
        
        scores.append({
            "Sector": sector,
            "Trend Strength (%)": round(trend_strength, 2),
            "Volatility (%)": round(volatility, 2),
            "Vol-Adjusted Return": round(vol_adjusted_return, 2)
        })
    df_scores = pd.DataFrame(scores)

    # Isolated Normalizations (Min-Max Scaling)
    df_scores["Trend Score"] = (df_scores["Trend Strength (%)"] - df_scores["Trend Strength (%)"].min()) / (df_scores["Trend Strength (%)"].max() - df_scores["Trend Strength (%)"].min()) * 100
    df_scores["Volatility Score"] = 100 - (df_scores["Volatility (%)"] - df_scores["Volatility (%)"].min()) / (df_scores["Volatility (%)"].max() - df_scores["Volatility (%)"].min()) * 100
    df_scores["Vol-Adj Score"] = (df_scores["Vol-Adjusted Return"] - df_scores["Vol-Adjusted Return"].min()) / (df_scores["Vol-Adjusted Return"].max() - df_scores["Vol-Adjusted Return"].min()) * 100

    #  Weights Structure 
    df_scores["Composite Score"] = (
        df_scores["Trend Score"] * 0.4 +
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

    # ==========================================
    # SUPPORTING METRICS PLOTS & LABELS
    # ==========================================
    st.subheader("Supporting Metrics")
    metric_specs = [
        ("Trend Strength (%)", "Long-Term Trend Strength (vs 200D SMA)"),
        ("Volatility (%)", "Daily Return Volatility"),
        ("Vol-Adjusted Return", "20-Day Momentum Smoothed & Vol-Adjusted"),
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

    # ==========================================
    # METHODOLOGY & EXPLANATIONS
    # ==========================================
    with st.expander("Methodology and Limitations"):
        st.write("""
        **Composite Score Calculation**

        Each sector receives a composite score (0-100) combining three distinct factors:
        - **Trend Strength (40%):** Evaluates the current price relative to its 200-day Simple Moving Average to measure macroeconomic trend health.
        - **Volatility-Adjusted Return (40%):** A risk-efficiency metric. It divides 20-day moving average momentum by daily volatility, rewarding steady growth trends while penalizing erratic spikes.
        - **Volatility Management (20%):** Standard deviation of daily returns. Lower baseline volatility yields a higher score to reward structural stability.

        Factors are mathematically normalized via Min-Max scaling to a strict 0-100 range prior to weighting, ensuring that varying units of measurement do not artificially skew final performance outputs.

        **Signal Labels**

        Strong (60+), Neutral (40-59), and Weak (below 40) represent a cross-sectional market rank compared to all tracked structural sectors simultaneously, rather than a selective peer group filter.

        **Limitations**
        - Historical trend features can lag during fast, fundamental macroeconomic shifts.
        - Volatility estimates assume standard normal return distributions, which might overlook sharp tail-risk events.
        - The composite score is a relative measure and does not guarantee future performance.
        """)

# ==========================================
# MODERNIZED UX 
# ==========================================
# ==========================================
# DARK-MODE UX 
# ==========================================
with tab_drilldown:
    # --- Custom CSS Injections for Tag Contrast ---
    st.markdown("""
        <style>
        /* Fix the ugly multi-select pill tag backgrounds to dark gray with white text */
        span[data-baseweb="tag"] {
            background-color: #21262D !important;
            color: #E6EDF2 !important;
            border: 1px solid #30363D !important;
            border-radius: 4px !important;
        }
        /* Style the little 'x' button inside the tags */
        span[data-baseweb="tag"] role[button] {
            color: #8B949E !important;
        }
        /* Sleek Skeleton Loading Animation CSS */
        .skeleton-card {
            background: linear-gradient(90deg, #161b22 25%, #21262d 50%, #161b22 75%);
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
            border-radius: 8px;
            height: 280px;
            margin-bottom: 1rem;
            border: 1px solid #30363D;
        }
        @keyframes loading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        </style>
    """, unsafe_allow_html=True)

    if drilldown_sector != "None":
        st.subheader(f"{drilldown_sector}: Top Stocks Summary")
        
        # Premium Shimmer Skeleton Loading Interface
        # This replaces the native 'Running fetch_stock_data' block entirely
        with st.empty():
            # Build an empty grid placeholder first
            grid_cols = st.columns(2)
            with grid_cols[0]:
                st.markdown('<div class="skeleton-card"></div>', unsafe_allow_html=True)
            with grid_cols[1]:
                st.markdown('<div class="skeleton-card"></div>', unsafe_allow_html=True)
            
            # Silently download the assets in the background
            stock_data = fetch_stock_data(sector_stocks[drilldown_sector], selected_period)
            # Instantly clears the placeholder animations once complete
            st.write("") 

        # 1. Initialize our clean 2-column layout grid matrix
        cols = st.columns(2)
        
        # 2. Distribute stock assets across layout slots
        for idx, (ticker, df) in enumerate(stock_data.items()):
            close = df["Close"].squeeze().dropna()
            
            fig_stock = px.line(
                close,
                labels={"value": "Price (USD)", "Date": ""},
                template="plotly_dark"
            )
            
            # Applied a clean electric-blue line accent that stands out against deep gray background canvas
            fig_stock.update_traces(
                line_color="#58A6FF", 
                line_width=2.5,
                hovertemplate="<b>Price:</b> $%{y:.2f}<extra></extra>"
            )
            
            fig_stock.update_layout(
                title={
                    'text': f"<b>{ticker}</b> — Historical Performance",
                    'y': 0.9,
                    'x': 0.05,
                    'xanchor': 'left',
                    'yanchor': 'top',
                    'font': dict(size=14, color="#E6EDF2")
                },
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=10, r=10, t=50, b=10), 
                hovermode="x",
                showlegend=False,
                xaxis=dict(
                    showgrid=True, 
                    gridcolor="#21262D", 
                    linecolor="#30363D"
                ),
                yaxis=dict(
                    showgrid=True, 
                    gridcolor="#21262D", 
                    position=1, 
                    side="right"
                )
            )
            
            target_col = cols[idx % 2]
            with target_col.container(border=True):
                st.plotly_chart(fig_stock, use_container_width=True, key=f"modern_chart_{ticker}")
    else:
        st.info("Select a sector from the sidebar under Stock Drill-Down to view individual stock charts.")
