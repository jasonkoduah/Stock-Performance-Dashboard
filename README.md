# Stock Sector Performance Dashboard

An interactive dashboard built with Streamlit that analyzes live S&P 500 sector performance using a custom composite scoring model.

## Features
- Live data pulled via yfinance for 10 S&P 500 sector ETFs
- Composite scoring model combining momentum, volatility, and volatility-adjusted return
- Color-coded Strong/Neutral/Weak signals for each sector
- Interactive filters for sector selection and time period
- Stock-level drill-down with individual price trend charts
- Methodology section documenting scoring assumptions and limitations

## Tech Stack
- Python
- Streamlit
- yfinance
- pandas
- Plotly

## Running Locally
1. Clone the repository
2. Install dependencies: `pip install yfinance pandas streamlit plotly`
3. Run: streamlit run app.py`

## Methodology
See the in-app Methodology and Limitations section for details on the scoring model.
