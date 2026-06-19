import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

st.set_page_config(page_title="Sector Performance Dashboard", layout="wide")

# ==========================================
# TITLE & DESCRIPTION
# ==========================================
st.title("Stock Sector Performance Dashboard")
st.write(
    "**How it works:** This dashboard ranks stock market sectors using a combination of trend strength and risk metrics. "
    "Under the **Stock Performance** tab, the analysis engine combines two separate components to look at individual stocks: one maps out where the price "
    "is likely heading over the next 14 days, while the other flags whether the stock is an active 'Buy' or a safer 'Hold' at the moment."
)

# Define sector ETFs & Stocks
sectors = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF", "Energy": "XLE",
    "Consumer Discretionary": "XLY", "Industrials": "XLI", "Utilities": "XLU",
    "Real Estate": "XLRE", "Materials": "XLB", "Communication Services": "XLC"
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

COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AVGO": "Broadcom", "ORCL": "Oracle",
    "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "LLY": "Eli Lilly", "ABBV": "AbbVie", "MRK": "Merck",
    "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs", "WFC": "Wells Fargo", "MS": "Morgan Stanley",
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips", "SLB": "Schlumberger", "EOG": "EOG Resources",
    "AMZN": "Amazon", "TSLA": "Tesla", "HD": "Home Depot", "MCD": "McDonald's", "NKE": "Nike",
    "CAT": "Caterpillar", "BA": "Boeing", "UPS": "UPS", "HON": "Honeywell", "GE": "GE Aerospace",
    "NEE": "NextEra Energy", "DUK": "Duke Energy", "SO": "Southern Company", "D": "Dominion Energy", "AEP": "AEP",
    "PLD": "Prologis", "AMT": "American Tower", "EQIX": "Equinix", "SPG": "Simon Property", "O": "Realty Income",
    "LIN": "Linde", "SHW": "Sherwin-Williams", "FCX": "Freeport-McMoRan", "APD": "Air Products", "NEM": "Newmont",
    "GOOGL": "Alphabet", "META": "Meta", "NFLX": "Netflix", "DIS": "Disney", "TMUS": "T-Mobile"
}

ACCENT_COLOR = "#4C72B0"
SIGNAL_COLORS = {"Strong": "#2E8B57", "Neutral": "#E8A33D", "Weak": "#D9534F"}
SIGNAL_ORDER = ["Strong", "Neutral", "Weak"]
PLOTLY_TEMPLATE = "plotly_white"

# --- Sidebar Layout Setup ---
st.sidebar.header("Filters")
selected_period = st.sidebar.selectbox("Select Time Period", options=["1mo", "3mo", "6mo", "1y", "2y"], index=2)
selected_sectors = st.sidebar.multiselect("Select Sectors", options=list(sectors.keys()), default=list(sectors.keys()))

st.sidebar.header("Stock Performance")
drilldown_sector = st.sidebar.selectbox("Select a sector to view individual stocks", options=["None"] + selected_sectors)


@st.cache_data
def fetch_sector_data(sectors, period):
    data = {}
    for sector, ticker in sectors.items():
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        data[sector] = df
    return data


# Startup Data Load Environment
with st.status("Initializing System: Downloading global market layers...", expanded=False) as startup_status:
    sector_data = fetch_sector_data(sectors, selected_period)
    startup_status.update(label="System Ready. Generating performance matrices...", state="complete")

st.write("")


def calculate_scores(sector_data):
    scores = []
    for sector, df in sector_data.items():
        close = df["Close"].squeeze().dropna()
        sma_span = min(200, len(close))
        sma_200 = close.rolling(window=sma_span).mean().iloc[-1]
        trend_strength = float(((close.iloc[-1] - sma_200) / sma_200) * 100)

        sma_20 = close.rolling(window=20).mean()
        momentum_20d = float(((sma_20.iloc[-1] - sma_20.iloc[-20]) / sma_20.iloc[-20]) * 100)

        volatility = float(close.pct_change().std() * 100)
        vol_adjusted_return = momentum_20d / volatility if volatility != 0 else 0

        scores.append({
            "Sector": sector, "Trend Strength (%)": round(trend_strength, 2),
            "Volatility (%)": round(volatility, 2), "Vol-Adjusted Return": round(vol_adjusted_return, 2)
        })
    df_scores = pd.DataFrame(scores)

    df_scores["Trend Score"] = (df_scores["Trend Strength (%)"] - df_scores["Trend Strength (%)"].min()) / (df_scores["Trend Strength (%)"].max() - df_scores["Trend Strength (%)"].min()) * 100
    df_scores["Volatility Score"] = 100 - (df_scores["Volatility (%)"] - df_scores["Volatility (%)"].min()) / (df_scores["Volatility (%)"].max() - df_scores["Volatility (%)"].min()) * 100
    df_scores["Vol-Adj Score"] = (df_scores["Vol-Adjusted Return"] - df_scores["Vol-Adjusted Return"].min()) / (df_scores["Vol-Adjusted Return"].max() - df_scores["Vol-Adjusted Return"].min()) * 100

    df_scores["Composite Score"] = (df_scores["Trend Score"] * 0.4 + df_scores["Volatility Score"] * 0.2 + df_scores["Vol-Adj Score"] * 0.4).round(2)

    def signal(score):
        if score >= 60: return "Strong"
        elif score >= 40: return "Neutral"
        else: return "Weak"

    df_scores["Signal"] = df_scores["Composite Score"].apply(signal)
    return df_scores


def calculate_hybrid_forecast(df_stock, sector_features, horizon=14):
    df_prep = df_stock["Close"].reset_index()
    df_prep.columns = ["ds", "y"]
    df_prep["ds"] = df_prep["ds"].dt.tz_localize(None)

    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        interval_width=0.95,
        uncertainty_samples=500
    )
    m.fit(df_prep)

    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)

    trend_score = sector_features["Trend Score"]
    vol_score = sector_features["Volatility Score"]
    vol_adj_score = sector_features["Vol-Adj Score"]

    buy_conditions = sum([
        trend_score >= 60,
        vol_score >= 50,
        vol_adj_score >= 60
    ])

    if buy_conditions >= 2:
        model_signal = "Model Signal: OUTPERFORM (Buy)"
    else:
        model_signal = "Model Signal: NEUTRAL (Hold)"

    return forecast, model_signal


# Process score arrays
scores_df_all = calculate_scores(sector_data)
scores_df = scores_df_all[scores_df_all["Sector"].isin(selected_sectors)]

if scores_df.empty:
    st.warning("Select at least one sector from the sidebar to view the dashboard.")
    st.stop()

scores_df = scores_df.sort_values("Composite Score", ascending=False).reset_index(drop=True)
sector_order = scores_df["Sector"].tolist()

last_date = sector_data[list(sector_data.keys())[0]].index[-1].strftime("%B %d, %Y")
st.caption(f"Data as of {last_date}")

# Setup Tabs
tab_overview, tab_drilldown = st.tabs(["Sector Overview", "Stock Performance"])

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
        scores_df, x="Sector", y="Composite Score", color="Signal",
        color_discrete_map=SIGNAL_COLORS, category_orders={"Sector": sector_order, "Signal": SIGNAL_ORDER},
        template=PLOTLY_TEMPLATE, title="Overall Sector Ranking (Composite Score)"
    )
    fig4.update_xaxes(tickangle=-30)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Supporting Metrics")
    metric_specs = [
        ("Trend Strength (%)", "Long-Term Trend Strength (vs 200D SMA)"),
        ("Volatility (%)", "Daily Return Volatility"),
        ("Vol-Adjusted Return", "20-Day Momentum Smoothed & Vol-Adjusted"),
    ]
    metric_cols = st.columns(3)
    for col, (metric_col, metric_title) in zip(metric_cols, metric_specs):
        fig = px.bar(scores_df, x="Sector", y=metric_col, category_orders={"Sector": sector_order}, template=PLOTLY_TEMPLATE, title=metric_title)
        fig.update_traces(marker_color=ACCENT_COLOR)
        fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
        fig.update_layout(height=320)
        col.plotly_chart(fig, use_container_width=True)


# ==========================================
# DRILLDOWN CONTENT LAYERS
# ==========================================
with tab_drilldown:
    st.markdown("""
        <style>
        span[data-baseweb="tag"] { background-color: #21262D !important; color: #E6EDF2 !important; border: 1px solid #30363D !important; border-radius: 4px !important; }
        .skeleton-card { background: linear-gradient(90deg, #161b22 25%, #21262d 50%, #161b22 75%); background-size: 200% 100%; animation: loading 1.5s infinite; border-radius: 8px; height: 280px; margin-bottom: 1rem; border: 1px solid #30363D; }
        @keyframes loading { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
        .insight-box { background-color: #161B22; padding: 14px; border-radius: 6px; border: 1px solid #30363D; font-size: 13px; color: #C9D1D9; margin-top: -5px; margin-bottom: 15px; line-height: 1.5; }
        </style>
    """, unsafe_allow_html=True)

    if drilldown_sector != "None":
        st.subheader(f"{drilldown_sector}: Top Stocks Predictive Matrix")
        sector_meta = scores_df[scores_df["Sector"] == drilldown_sector].iloc[0]
        tickers_to_load = sector_stocks[drilldown_sector]

        stock_data = {}
        with st.container():
            progress_text = st.empty()
            progress_bar = st.progress(0)
            skeleton_grid = st.empty()
            with skeleton_grid.container():
                g_cols = st.columns(2)
                g_cols[0].markdown('<div class="skeleton-card"></div>', unsafe_allow_html=True)
                g_cols[1].markdown('<div class="skeleton-card"></div>', unsafe_allow_html=True)

            for idx, ticker in enumerate(tickers_to_load):
                progress_text.markdown(f"**Predictive Engine:** Quantifying metrics for **{ticker}**...")
                percent_complete = int(((idx + 1) / len(tickers_to_load)) * 100)
                progress_bar.progress(percent_complete)
                single_df = yf.download(ticker, period=selected_period, auto_adjust=True, progress=False)
                stock_data[ticker] = single_df

            progress_text.empty()
            progress_bar.empty()
            skeleton_grid.empty()

        cols = st.columns(2)
        for idx, (ticker, df) in enumerate(stock_data.items()):
            forecast, model_badge = calculate_hybrid_forecast(df, sector_meta, horizon=14)

            hist_df = df["Close"].reset_index()
            hist_df.columns = ["Date", "Price"]
            hist_df["Date"] = hist_df["Date"].dt.tz_localize(None)

            last_hist_date = hist_df["Date"].max()
            last_hist_price = float(hist_df["Price"].squeeze().dropna().iloc[-1])

            future_df = forecast[forecast["ds"] > last_hist_date].copy()
            prophet_boundary_price = float(forecast[forecast["ds"] <= last_hist_date]["yhat"].iloc[-1])
            vertical_bias = last_hist_price - prophet_boundary_price

            future_df["yhat"] += vertical_bias
            future_df["yhat_upper"] += vertical_bias
            future_df["yhat_lower"] += vertical_bias

            upper_vals = future_df[["yhat_upper", "yhat_lower"]].max(axis=1)
            lower_vals = future_df[["yhat_upper", "yhat_lower"]].min(axis=1)
            future_df["yhat_upper"] = upper_vals
            future_df["yhat_lower"] = lower_vals

            connect_row = pd.DataFrame({
                'ds': [last_hist_date],
                'yhat': [float(last_hist_price)],
                'yhat_upper': [float(last_hist_price)],
                'yhat_lower': [float(last_hist_price)]
            })

            future_df = pd.concat([connect_row, future_df]).sort_values("ds").reset_index(drop=True)
            future_df["ds"] = pd.to_datetime(future_df["ds"])

            is_bullish_line = future_df["yhat"].iloc[-1] > last_hist_price

            if "OUTPERFORM" in model_badge:
                badge_color = "#2E8B57"
                line_color = "#2E8B57"
                fill_color = "rgba(46,139,87,0.25)"
                if not is_bullish_line:
                    insight_text = f"🟢 <b>Model Value Alert:</b> Even though the short-term chart line points slightly downward, the model has flagged this stock as a <b>BUY</b>. This means the stock has dropped recently, making it look discounted. Because its underlying sector fundamentals are rock solid, the framework considers this a great long-term bargain entry point."
                else:
                    insight_text = f"🟢 <b>Model Momentum Alert:</b> All metrics are fully aligned. Both the historical price momentum and the sector's risk data suggest room for growth. The stock is backed by strong market demand and stable trading conditions."
            else:
                badge_color = "#8B949E"
                line_color = "#E8A33D"
                fill_color = "rgba(232,163,61,0.25)"
                if is_bullish_line:
                    insight_text = f"⚠️ <b>Risk Warning:</b> The chart shows a slight upward trend based on past seasonal behavior, but the classification model still warns you to <b>HOLD</b>. Sharp, sudden jumps in market volatility mean this mini-rally is unpredictable and carries a high risk of reversing."
                else:
                    insight_text = f"🛑 <b>Sideways Trend:</b> Price momentum is slowing down and baseline sector risks are expanding. There is no clear advantage to entering a new position right now; sitting on cash is the safer move."

            company_name = COMPANY_NAMES.get(ticker, ticker)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=hist_df["Date"], y=hist_df["Price"],
                name="Historical", mode="lines", line=dict(color="#58A6FF", width=2.5)
            ))
            fig.add_trace(go.Scatter(
                x=future_df["ds"], y=future_df["yhat"],
                name="Forecast", mode="lines", line=dict(color=line_color, width=2)
            ))
            fig.add_trace(go.Scatter(
                x=future_df["ds"], y=future_df["yhat_upper"],
                mode="lines", line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=future_df["ds"], y=future_df["yhat_lower"],
                mode="lines", line=dict(color="rgba(0,0,0,0)"),
                fill='tonexty', fillcolor=fill_color,
                name="Uncertainty Bound", hoverinfo="skip"
            ))

            fig.update_layout(
                template="plotly_dark",
                title={
                    'text': f"<b>{company_name} ({ticker})</b> — {last_hist_price:.2f} USD <span style='color:{badge_color}; font-size:12px; margin-left:10px;'>● {model_badge}</span>",
                    'y': 0.9, 'x': 0.05, 'xanchor': 'left', 'yanchor': 'top',
                    'font': dict(size=14, color="#E6EDF2")
                },
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=260, margin=dict(l=10, r=10, t=60, b=10),
                hovermode="x unified", showlegend=False,
                xaxis=dict(showgrid=True, gridcolor="#21262D", linecolor="#30363D"),
                yaxis=dict(showgrid=True, gridcolor="#21262D", position=1, side="right")
            )

            target_col = cols[idx % 2]
            with target_col.container(border=True):
                st.plotly_chart(fig, use_container_width=True, key=f"ml_chart_{ticker}")
                st.markdown(f'<div class="insight-box">{insight_text}</div>', unsafe_allow_html=True)
    else:
        st.info("Select an active sector from the sidebar under Stock Performance to initialize individual asset streams.")

# ==========================================
#  METHODOLOGY CONTAINER
# ==========================================
st.markdown("---")
with st.expander("Methodology and System Limitations"):
    st.write("""
    **Composite Score Calculation**

    Each sector receives a composite score (0-100) combining three distinct factors:
    - **Trend Strength (40%):** Evaluates the current price relative to its 200-day Simple Moving Average to measure macroeconomic trend health.
    - **Volatility-Adjusted Return (40%):** A risk-efficiency metric. It divides 20-day moving average momentum by daily volatility, rewarding steady growth trends while penalizing erratic spikes.
    - **Volatility Management (20%):** Standard deviation of daily returns. Lower baseline volatility yields a higher score to reward structural stability.

    Factors are mathematically normalized via Min-Max scaling to a 0-100 range prior to weighting, ensuring that varying units of measurement do not artificially skew final performance outputs.

    **Signal Labels**

    Strong (60+), Neutral (40-59), and Weak (below 40) represent a cross-sectional market rank compared to all tracked structural sectors simultaneously, rather than a selective peer group filter.

    **Stock-Level Signal Engine**
    - **Prophet Time-Series Forecast:** Models price history to generate a localized 14-day expected price trend with uncertainty bounds.
    - **Rule-Based Signal:** Evaluates the sector's normalized composite features including the trend score, volatility score, and volatility-adjusted return score against defined thresholds. A Buy signal requires at least 2 of 3 conditions to pass, grounding the output in the same scoring logic used across the sector overview.

    **Limitations**
    - Historical trend features can lag during fast, fundamental macroeconomic shifts.
    - Volatility estimates assume standard normal return distributions, which may underweight sharp tail-risk events.
    - The Prophet forecast is trained on price history only and does not incorporate earnings, macro events, or sentiment data.
    - The composite score and Buy/Hold signal are relative measures and do not constitute investment advice.
    - The stock drill-down uses a fixed list of 5 representative companies per sector rather than the full ETF composition.
    """)