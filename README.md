# ARRY Trading Desk — dashboard (Streamlit + static HTML)

| File | What it is |
|---|---|
| `app.py` | Streamlit app (Python). Analytics at the top, UI in `main()`. Any Yahoo ticker can be typed in the sidebar. |
| `requirements.txt` | Dependencies (`streamlit`, `streamlit-echarts`, `yfinance`, `statsmodels`, `arch`, `scipy`, …). |
| `dashboard.html` | Self-contained interactive dashboard (ECharts inlined, works offline). Data snapshot built from the same analytics. |
| `build_html.py` + `dashboard_template.html` | Regenerates `dashboard.html` (`python build_html.py`, needs internet). |
| `.streamlit/config.toml` | Dark trader theme for Streamlit. |
| `source files/` | Q3-2026 press release, 10-Q, presentation, course prompts, ARRY link list. |

## Run
```bash
pip install -r requirements.txt
streamlit run app.py          # live app (deploy on Streamlit Community Cloud with the same two files + .streamlit/)
python build_html.py          # refresh dashboard.html
```

## Data decision
The URL list in `info links ARRY.txt` was reachable (Finviz, Yahoo Finance, Google News respond; arraytechinc.com
pages were not scraped — the Q3 filings already in `source files/` cover the company facts). Primary source is
**yfinance** (3y OHLCV, statements, option chains); **Finviz** is scraped best-effort for the snapshot ratios, profile,
peers, analyst actions and headlines (it falls back gracefully if blocked); **Google News RSS** adds headlines.

## Method notes
* **Candlestick signals** (hammer, shooting star, bullish/bearish engulfing, morning/evening star, piercing line, dark-cloud cover)
  are *raw* signals. A signal is kept (“confirmed”) only if it passes ≥ N of 3 checks — prior trend vs EMA28, volume ≥ k × 20-day average,
  RSI/Bollinger location — **and** the next bar follows through (no look-ahead: entry at the confirmation close). The rest are shown as “filtered”.
  The Technical tab back-tests raw vs confirmed signals.
* **Forecast**: ARIMA(p,d,q) on log-price (AIC grid, ADF test → d=1) + AR(1)-GARCH(1,1)-t Monte-Carlo (fat tails, volatility clustering).
  Drift is shrunk toward zero. ARIMA(0,1,0) is what AIC picks for ARRY — i.e. a random walk — so the value is in the probability bands.
  Walk-forward back-test is reported honestly (point forecast ≈ naive; bands are calibrated).
* **Options**: Black-Scholes deltas from listed IV; covered call, cash-secured put, put credit spread, bull call spread, collar; fit labels depend on
  IV/HV90 and the horizon bias. Chains for ARRY are thin — check liquidity flags.
* **Composite score** per horizon (1w / 3m / 6m / 12m) blends technical, forecast, fundamental/valuation and news sentiment, with weights shifting
  from technical (short) to fundamental (long). It is a heuristic, not advice.
* Course prompts: *Prompt para analizar reporte anual o trimestral* → “Report analysis Q3-26” tab; *PROMPT para analizar KPIs del negocio* → “Revenue tree” tab.

Educational use only — not investment advice.
