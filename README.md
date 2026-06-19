Stock Sector Performance Dashboard

An interactive dashboard built with Streamlit that analyzes live S&P 500 sector performance using a custom composite scoring model and a 14-day price forecast engine.

Live App: https://stock-performance-dashboard-mivfmmdeutuh7cvgitx9nn.streamlit.app/


Features


Live data pipeline pulling real-time price data for 10 S&P 500 sector ETFs via yfinance, with caching to minimize redundant API calls
Custom composite scoring model combining trend strength, volatility management, and volatility-adjusted return — normalized to a 0-100 scale via Min-Max scaling
Color-coded Strong / Neutral / Weak signals ranking sectors by composite score, with interactive bar charts for each underlying factor
KPI cards surfacing top sector, bottom sector, average composite score, and strong sector count
Sector and time period filters that dynamically update all visualizations
Stock-level drill-down with 14-day Prophet time-series price forecasts and uncertainty bounds for 5 representative stocks per sector
Rule-based Buy / Hold signal engine grounded in the same composite scoring thresholds used across the sector overview
Methodology and limitations section documenting all scoring assumptions


Tech Stack


Python
Streamlit
yfinance
pandas
Plotly
Prophet


Running Locally


Clone the repository
Install dependencies:


   pip install yfinance pandas streamlit plotly prophet scikit-learn


Run:


   streamlit run app.py

Methodology

See the in-app Methodology and System Limitations section for full details on the composite scoring model, signal thresholds, and forecast architecture.
