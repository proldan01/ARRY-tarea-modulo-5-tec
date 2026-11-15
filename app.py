"""
ARRY Trading Desk  —  Streamlit dashboard (technical + fundamental + forecast + options)
Run:  streamlit run app.py

Data sources
  * Yahoo Finance (yfinance): 3y daily OHLCV, statements, option chains, ratios   (primary)
  * Finviz snapshot (scraped, best-effort): profile, sector, peers, analyst ratings, short float
  * Google News RSS: headlines (Yahoo news endpoint is flaky)
  * Company filings in `source files/` (Q3-2026 press release, 10-Q, presentation): segment data,
    orderbook, guidance, leverage  ->  hard-coded in FACTS below so the app works offline for them.

The pure-python analytics (indicators, candlestick signals, ARIMA/GARCH forecast, options ideas,
horizon scoring) live at the top of this file and are re-used by build_html.py to generate the
static dashboard (dashboard.html).  UI code is inside main().
Not investment advice.
"""
from __future__ import annotations

import json
import math
import re
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"}

# ----------------------------------------------------------------------------------------------
# 1. COMPANY FACTS taken from the Q3-2026 source files (press release / presentation / 10-Q)
# ----------------------------------------------------------------------------------------------
FACTS = {
    "asof": "Q3 2026 (quarter ended Sep 30, 2026 — reported Nov 4, 2026)",
    "orderbook": {"labels": ["3Q25", "4Q25", "1Q26", "2Q26", "3Q26"], "usd_b": [1.9, 2.2, 2.4, 2.5, 2.6]},
    "book_to_bill_ttm": 1.6, "new_orders_ttm_b": 1.8, "gw_delivered": 105, "orderbook_yoy": 0.42,
    "new_products_pct_orderbook": 0.60, "active_patents": 294,
    # quarterly, $M — presentation p.8/18 (4Q25 / 1Q26 revenue derived: 9M26 − 2Q − 3Q, 4Q25 from EBITDA margin)
    "q": {
        "labels": ["3Q25", "4Q25", "1Q26", "2Q26", "3Q26"],
        "revenue": [393.5, 224.8, 223.4, 342.1, 330.2],
        "adj_ebitda": [72.2, 11.2, 28.8, 63.3, 32.1],
        "adj_ebitda_margin": [18.3, 5.0, 12.9, 18.5, 9.7],
    },
    "guidance": {  # FY2026, $M (EPS in $)
        "labels": ["Revenue", "Adj. EBITDA", "Adj. EPS ($)"],
        "prev_lo": [1400, 210, 0.68], "prev_hi": [1500, 230, 0.75],
        "new_lo": [1340, 190, 0.57], "new_hi": [1400, 210, 0.69],
        "adj_gm": 0.27, "q4_rev_lo": 445, "q4_rev_hi": 505,
    },
    "segments": {  # $M
        "q3_26": {"Array Legacy (U.S.)": 296.2, "STI Operations (Intl.)": 34.0},
        "q3_25": {"Array Legacy (U.S.)": 353.4, "STI Operations (Intl.)": 40.1},
        "m9_26": {"Array Legacy (U.S.)": 833.9, "STI Operations (Intl.)": 61.8},
        "m9_25": {"Array Legacy (U.S.)": 858.5, "STI Operations (Intl.)": 199.6},
        "over_time_q3_26": 251.7, "point_in_time_q3_26": 78.5,
    },
    "liquidity": {"cash": 143.3, "rcf_available": 322.6, "total_liquidity": 465.9,
                  "total_debt": 673.2, "net_debt": 530.0, "ttm_adj_ebitda": 135.0,
                  "debt_ebitda": 5.0, "net_debt_ebitda": 3.9, "conv_2028": 325.0, "conv_2031": 345.0,
                  "ratings": "B1 (stable) / B+ (stable)"},
    "customers": "Top-2 U.S. customers = 14.2% and 11.8% of Q3-26 revenue (18.5% / 17.5% for 9M-26)",
    "rpo_m": 522.2,
    "merger": "Stock-for-stock merger with Shoals (SHLS) announced Nov 4, 2026: 1.6931 ARRY shares per SHLS share; "
              "SHLS holders ≈65% / ARRY holders ≈35% of combined company; Series A preferred minimum redemption cut $200M→$50M.",
}

# Reported statements ($ thousands) — press release Q3-2026
REPORTED = {
    "income": {  # label: [Q3-26, Q3-25, 9M-26, 9M-25]
        "Revenue": [330212, 393491, 895689, 1058097],
        "Cost of revenue": [260993, 287820, 663862, 778874],
        "Gross profit": [69219, 105671, 231827, 279223],
        "General & administrative": [58783, 52248, 163512, 141147],
        "Δ FV contingent consideration": [-10293, 1014, -10438, 1014],
        "Depreciation & amortization": [6789, 6958, 22939, 17951],
        "Income from operations": [13940, 45451, 55814, 119111],
        "Interest expense": [-5810, -5046, -17159, -21849],
        "Income tax expense": [1651, 9941, 11156, 30092],
        "Net income": [8197, 33503, 34540, 93511],
        "Preferred dividends & accretion": [16113, 15144, 47558, 44375],
        "Net income to common": [-7916, 18359, -13018, 49136],
        "Adj. gross profit": [75404, 110504, 249321, 291335],
        "Adj. EBITDA": [32061, 72194, 124148, 176336],
        "Adj. net income": [9447, 45762, 55327, 104235],
    },
    "balance": {  # label: [Sep 30 2026, Dec 31 2025]
        "Cash & equivalents": [143280, 244388], "Accounts receivable": [329456, 271578],
        "Inventories": [167494, 150374], "Total current assets": [735661, 869044],
        "PP&E, net": [67442, 58225], "Goodwill": [231765, 135173], "Other intangibles": [274638, 238579],
        "Total assets": [1584951, 1451792], "Accounts payable": [199966, 143994],
        "Deferred revenue (current)": [108484, 128433], "Total current liabilities": [453990, 377012],
        "Long-term debt": [658550, 658664], "Total liabilities": [1283330, 1191404],
        "Series A redeemable preferred": [509978, 466728], "Total stockholders' equity": [-208357, -206340],
    },
    "cashflow": {  # label: [Q3-26, Q3-25, 9M-26, 9M-25]
        "Net cash from operations": [4618, 27363, 96476, 58145],
        "Capex (PP&E)": [-5700, -5513, -20844, -14496],
        "Free cash flow": [-1082, 21850, 75632, 43649],
        "Acquisitions, net of cash": [-160869, -164916, -160869, -164916],
        "Series A preferred dividends paid": [-4308, 0, -4308, 0],
    },
}

PEERS_DEFAULT = ["NXT", "SHLS", "FSLR", "ENPH", "SEDG", "RUN", "CSIQ", "FLNC"]
BENCHMARKS = {"S&P 500 (^GSPC)": "^GSPC", "Nasdaq Comp. (^IXIC)": "^IXIC", "Dow Jones (^DJI)": "^DJI",
              "Russell 2000 (^RUT)": "^RUT", "Nasdaq-100 ETF (QQQ)": "QQQ", "Solar ETF (TAN)": "TAN"}


# ----------------------------------------------------------------------------------------------
# 2. DATA LAYER
# ----------------------------------------------------------------------------------------------
def get_history(ticker: str, years: float = 3.0) -> pd.DataFrame:
    import yfinance as yf
    period = f"{int(round(years))}y" if years >= 1 else "6mo"
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No price data for '{ticker}'")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df[~df.index.duplicated(keep="last")]


def get_info(ticker: str) -> dict:
    import yfinance as yf
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def get_statements(ticker: str) -> dict:
    import yfinance as yf
    t = yf.Ticker(ticker)
    out = {}
    for k, attr in (("income", "quarterly_income_stmt"), ("balance", "quarterly_balance_sheet"),
                    ("cashflow", "quarterly_cashflow"), ("income_a", "income_stmt"),
                    ("balance_a", "balance_sheet"), ("cashflow_a", "cashflow")):
        try:
            d = getattr(t, attr)
            out[k] = d if d is not None else pd.DataFrame()
        except Exception:
            out[k] = pd.DataFrame()
    return out


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()


def get_finviz(ticker: str) -> dict:
    """Best-effort scrape of the Finviz quote page (snapshot table, profile, sector, peers, ratings, news)."""
    out = {"snapshot": {}, "profile": "", "sector": "", "industry": "", "country": "", "peers": [],
           "ratings": [], "news": [], "ok": False}
    try:
        r = requests.get(f"https://finviz.com/quote.ashx?t={ticker}&p=d", headers=UA, timeout=20)
        if r.status_code != 200:
            return out
        t = r.text
        cells = [_clean(x) for x in re.findall(r"snapshot-td2[^>]*>(.*?)</td>", t, flags=re.S)]
        out["snapshot"] = dict(zip(cells[0::2], cells[1::2]))
        m = re.search(r'f=sec_[a-z_]+[^"]*"[^>]*>([^<]+)<', t)
        out["sector"] = m.group(1) if m else ""
        m = re.search(r'f=ind_[a-z_]+[^"]*"[^>]*>([^<]+)<', t)
        out["industry"] = m.group(1) if m else ""
        m = re.search(r'f=geo_[a-z_]+[^"]*"[^>]*>([^<]+)<', t)
        out["country"] = m.group(1) if m else ""
        tk = list(dict.fromkeys(re.findall(r'data-boxover-ticker="([A-Z\.\-]+)"', t)))
        if "TAN" in tk:   # Finviz lists peers first, then ETFs that hold the stock
            out["peers"], out["etfs"] = tk[:tk.index("TAN")], tk[tk.index("TAN"):tk.index("TAN") + 6]
        else:
            out["peers"], out["etfs"] = tk[:8], []
        out["peers"] = [x for x in out["peers"] if x != ticker]
        m = re.search(r'class="[^"]*"[^>]*>\s*([^<]{80,900}?(?:founded|headquartered)[^<]{0,200})<', t)
        if m:
            out["profile"] = _clean(m.group(1))
        else:
            j = t.find(" manufactures ")
            if j > 0:
                s = t[max(0, j - 60):j + 900]
                out["profile"] = _clean(s.split("{")[0])
        rt = re.findall(r'"date":"(\d{4}-\d\d-\d\d)","action":"([^"]+)","firm":"([^"]+)","rating":"([^"]+)"'
                        r'(?:,"priceTarget":"([^"]*)")?', t)
        out["ratings"] = [{"date": a, "action": b, "firm": c, "rating": d.encode().decode("unicode_escape"),
                           "pt": e} for a, b, c, d, e in rt[:15]]
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(t, "lxml")
            for a in soup.select("a.tab-link-news")[:15]:
                tr = a.find_parent("tr")
                when = tr.find("td").get_text(" ", strip=True) if tr is not None else ""
                href = a.get("href", "")
                if href.startswith("/"):
                    href = "https://finviz.com" + href
                out["news"].append({"title": a.get_text(" ", strip=True), "link": href, "when": when,
                                    "source": "Finviz"})
        except Exception:
            pass
        out["ok"] = bool(out["snapshot"])
    except Exception:
        pass
    return out


def get_news(ticker: str, company: str = "") -> list[dict]:
    """Google News RSS (robust), + Yahoo news if available."""
    items = []
    q = requests.utils.quote(f"{ticker} {company} stock".strip())
    try:
        r = requests.get(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
                         headers=UA, timeout=20)
        root = ET.fromstring(r.content)
        for it in root.findall(".//item")[:30]:
            title = it.findtext("title") or ""
            src = it.findtext("source") or ""
            items.append({"title": title, "link": it.findtext("link") or "", "when": it.findtext("pubDate") or "",
                          "source": src or "Google News"})
    except Exception:
        pass
    return items


def filter_relevant(items: list[dict], ticker: str, company: str = "") -> list[dict]:
    """Keep headlines that actually mention the ticker / company's first word; de-duplicate by title."""
    keys = [ticker.lower()] + ([company.split()[0].lower()] if company else [])
    pat = re.compile(r"\b(" + "|".join(map(re.escape, keys)) + r")\b", re.I)
    seen, out = set(), []
    for it in items:
        t = it["title"].strip()
        if pat.search(t) and t.lower() not in seen:
            seen.add(t.lower())
            out.append(it)
    return out


def score_sentiment(items: list[dict]) -> list[dict]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        an = SentimentIntensityAnalyzer()
        for it in items:
            it["sent"] = round(an.polarity_scores(it["title"])["compound"], 3)
    except Exception:
        pos = ("beat", "record", "surge", "upgrade", "growth", "raises", "strong", "buy", "outperform", "gain")
        neg = ("miss", "falls", "cut", "downgrade", "loss", "drop", "slide", "sinks", "weak", "short", "lawsuit")
        for it in items:
            tl = it["title"].lower()
            it["sent"] = round(max(-1, min(1, 0.35 * (sum(w in tl for w in pos) - sum(w in tl for w in neg)))), 3)
    return items


# ----------------------------------------------------------------------------------------------
# 3. INDICATORS
# ----------------------------------------------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 10) -> pd.Series:
    d = close.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    au, ad = up.ewm(alpha=1 / n, adjust=False).mean(), dn.ewm(alpha=1 / n, adjust=False).mean()
    out = 100 - 100 / (1 + au / ad.replace(0, np.nan))
    return out.where(ad != 0, 100.0)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    m = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return m, m + k * sd, m - k * sd


def macd(close: pd.Series, f=12, s=26, sig=9):
    line = ema(close, f) - ema(close, s)
    signal = ema(line, sig)
    return line, signal, line - signal


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame, rsi_n=10, bb_n=20, bb_k=2.0, emas=(7, 14, 28, 50, 200)) -> pd.DataFrame:
    d = df.copy()
    for n in emas:
        d[f"EMA{n}"] = ema(d["Close"], n)
    d["BBm"], d["BBu"], d["BBl"] = bollinger(d["Close"], bb_n, bb_k)
    d["RSI"] = rsi(d["Close"], rsi_n)
    d["MACD"], d["MACDs"], d["MACDh"] = macd(d["Close"])
    d["ATR"] = atr(d)
    d["VolSMA20"] = d["Volume"].rolling(20).mean()
    d["EMA28"] = d["EMA28"] if "EMA28" in d else ema(d["Close"], 28)
    return d


# ----------------------------------------------------------------------------------------------
# 4. CANDLESTICK PATTERNS + FILTERED BUY / SELL SIGNALS
# ----------------------------------------------------------------------------------------------
BUY_PATTERNS = {"hammer": "Hammer", "bull_engulf": "Bullish Engulfing", "morning_star": "Morning Star",
                "piercing": "Piercing Line"}
SELL_PATTERNS = {"shooting_star": "Shooting Star", "bear_engulf": "Bearish Engulfing",
                 "evening_star": "Evening Star", "dark_cloud": "Dark Cloud Cover"}


def candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    up = h - np.maximum(o, c)
    lo = np.minimum(o, c) - l
    ab = body.rolling(14, min_periods=5).mean()
    po, pc, pbody = o.shift(1), c.shift(1), body.shift(1)
    green, red, pgreen, pred = c > o, c < o, pc > po, pc < po
    o2, c2, b2, ab2 = o.shift(2), c.shift(2), body.shift(2), ab.shift(2)
    out = pd.DataFrame(index=df.index)
    out["doji"] = body <= 0.1 * rng
    out["hammer"] = (lo >= 2 * body) & (up <= np.maximum(0.5 * body, 0.1 * rng)) & (body >= 0.03 * rng)
    out["shooting_star"] = (up >= 2 * body) & (lo <= np.maximum(0.5 * body, 0.1 * rng)) & (body >= 0.03 * rng)
    out["bull_engulf"] = pred & green & (o <= pc) & (c >= po) & (body > pbody)
    out["bear_engulf"] = pgreen & red & (o >= pc) & (c <= po) & (body > pbody)
    out["morning_star"] = (c2 < o2) & (b2 >= ab2) & (body.shift(1) <= 0.5 * b2) & green & (c > (o2 + c2) / 2)
    out["evening_star"] = (c2 > o2) & (b2 >= ab2) & (body.shift(1) <= 0.5 * b2) & red & (c < (o2 + c2) / 2)
    out["piercing"] = pred & (pbody >= ab.shift(1)) & green & (o < pc) & (c > (po + pc) / 2) & (c < po)
    out["dark_cloud"] = pgreen & (pbody >= ab.shift(1)) & red & (o > pc) & (c < (po + pc) / 2) & (c > po)
    return out.fillna(False)


def build_signals(d: pd.DataFrame, vol_mult=1.2, min_conf=2, require_next=True,
                  rsi_buy=45.0, rsi_sell=55.0) -> pd.DataFrame:
    """d must contain indicators (EMA28, RSI, BBl/BBu, VolSMA20).
    Raw signal = reversal candlestick pattern.  Confirmation checks (score 0-3):
      1 trend context  : prior close below EMA28 for buys (above for sells)  — a reversal pattern needs a trend to reverse
      2 volume         : signal-bar volume >= vol_mult x 20d average
      3 location       : RSI <= rsi_buy (>= rsi_sell) OR candle touched lower (upper) Bollinger band
    + next-bar check   : buy needs next close > signal close and > signal low (sell mirrored); last bar = 'pending'.
    State: confirmed (score >= min_conf and next-bar ok) | pending | filtered (misleading)."""
    pats = candle_patterns(d)
    c, h, l = d["Close"], d["High"], d["Low"]
    rows = []
    prev_close, prev_ema = c.shift(1), d["EMA28"].shift(1)
    vol_ok = d["Volume"] >= vol_mult * d["VolSMA20"]
    nxt = c.shift(-1)
    for side, group in (("BUY", BUY_PATTERNS), ("SELL", SELL_PATTERNS)):
        any_p = pats[list(group)].any(axis=1)
        for dt in d.index[any_p]:
            i = d.index.get_loc(dt)
            names = [group[k] for k in group if pats.at[dt, k]]
            if side == "BUY":
                trend = bool(prev_close.iloc[i] < prev_ema.iloc[i])
                loc = bool(d["RSI"].iloc[i] <= rsi_buy or l.iloc[i] <= d["BBl"].iloc[i] * 1.01)
            else:
                trend = bool(prev_close.iloc[i] > prev_ema.iloc[i])
                loc = bool(d["RSI"].iloc[i] >= rsi_sell or h.iloc[i] >= d["BBu"].iloc[i] * 0.99)
            vol = bool(vol_ok.iloc[i])
            score = int(trend) + int(vol) + int(loc)
            last = i == len(d) - 1
            if last:
                next_ok = None
            elif side == "BUY":
                next_ok = bool(nxt.iloc[i] > c.iloc[i] and nxt.iloc[i] > l.iloc[i])
            else:
                next_ok = bool(nxt.iloc[i] < c.iloc[i] and nxt.iloc[i] < h.iloc[i])
            if score < min_conf:
                state = "filtered"
            elif last:
                state = "pending"
            elif require_next and not next_ok:
                state = "filtered"
            else:
                state = "confirmed"
            rows.append({"date": dt, "side": side, "pattern": " + ".join(names), "score": score, "trend": trend,
                         "volume": vol, "location": loc, "next_ok": next_ok, "state": state,
                         "price": float(c.iloc[i]), "high": float(h.iloc[i]), "low": float(l.iloc[i])})
    sig = pd.DataFrame(rows)
    if sig.empty:
        return pd.DataFrame(columns=["date", "side", "pattern", "score", "trend", "volume", "location", "next_ok",
                                     "state", "price", "high", "low"])
    # same-bar BUY+SELL conflict -> drop both
    dup = sig.groupby("date")["side"].nunique()
    sig = sig[~sig["date"].isin(dup[dup > 1].index)]
    return sig.sort_values("date").reset_index(drop=True)


def signal_backtest(d: pd.DataFrame, sig: pd.DataFrame, hold=10) -> pd.DataFrame:
    """Forward return (side-adjusted). Raw: enter at signal close. Confirmed: enter at next-bar close (when known)."""
    c = d["Close"].values
    idx = {dt: i for i, dt in enumerate(d.index)}
    res = []
    for label, sub, off in (("Raw pattern (all)", sig, 0), ("Confirmed", sig[sig["state"] == "confirmed"], 1)):
        rets = []
        for _, r in sub.iterrows():
            i = idx[r["date"]] + off
            if i + hold < len(c):
                ret = c[i + hold] / c[i] - 1
                rets.append(ret if r["side"] == "BUY" else -ret)
        rets = np.array(rets)
        res.append({"Filter": label, "Signals": len(rets),
                    "Win rate": float((rets > 0).mean()) if len(rets) else np.nan,
                    "Avg return": float(rets.mean()) if len(rets) else np.nan,
                    "Median": float(np.median(rets)) if len(rets) else np.nan})
    return pd.DataFrame(res)


# ----------------------------------------------------------------------------------------------
# 5. FORECAST: ARIMA (statsmodels) + ARIMA-GARCH Monte-Carlo (arch)
# ----------------------------------------------------------------------------------------------
HORIZONS = {"1 week": 5, "3 months": 63, "6 months": 126, "12 months": 252}


def fit_arima(r: np.ndarray, max_p=3, max_q=3):
    from statsmodels.tsa.arima.model import ARIMA
    best = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                m = ARIMA(r, order=(p, 0, q), trend="c").fit()
                if best is None or m.aic < best[0]:
                    best = (m.aic, (p, q), m)
            except Exception:
                continue
    return best


def forecast_models(close: pd.Series, horizon=252, sims=2000, shrink=0.5, seed=7) -> dict:
    """ARIMA(p,1,q) on log-price == ARMA(p,q)+drift on log-returns. Order picked by AIC (p,q<=3).
    Drift is shrunk toward 0 (`shrink`) because extrapolating a 3y sample drift is unreliable.
    Fan-chart bands come from simulated paths. Second model: AR(1)-GARCH(1,1)-t (fat tails, vol clustering)."""
    from statsmodels.tsa.stattools import adfuller
    lp = np.log(close.astype(float))
    r = lp.diff().dropna().values
    S0 = float(close.iloc[-1])
    out = {"S0": S0, "horizon": horizon, "adf_price_p": float(adfuller(lp.values)[1]),
           "adf_ret_p": float(adfuller(r)[1])}
    aic, (p, q), res = fit_arima(r)
    params = np.array(res.params, dtype=float)
    params[0] *= shrink
    np.random.seed(seed)
    # `const` of sm.tsa.arima.ARIMA is the process mean -> shrink it by re-centring the simulated returns
    sim = np.asarray(res.simulate(nsimulations=horizon, repetitions=sims, anchor="end")).reshape(horizon, sims)
    sim = sim - (1 - shrink) * float(res.params[0])
    out["arima"] = {"order": f"ARIMA({p},1,{q})", "aic": float(aic), "drift_daily": float(params[0]),
                    "paths": _percentiles(S0 * np.exp(np.cumsum(sim, axis=0)))}
    out["arima"]["final"] = (S0 * np.exp(np.cumsum(sim, axis=0)))[-1]
    # GARCH
    try:
        from arch import arch_model
        am = arch_model(r * 100, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t")
        g = am.fit(disp="off")
        f = g.forecast(horizon=horizon, method="simulation", simulations=sims, reindex=False)
        sg = f.simulations.values[-1] / 100.0
        mu = g.params.get("Const", 0.0) / 100.0 / (1 - g.params.get("y[1]", 0.0))
        sg = sg - (1 - shrink) * mu
        gp = S0 * np.exp(np.cumsum(sg, axis=1)).T
        out["garch"] = {"order": "AR(1)-GARCH(1,1)-t", "aic": float(g.aic), "paths": _percentiles(gp),
                        "final": gp[-1], "cond_vol_ann": float(g.conditional_volatility[-1] * math.sqrt(252) / 100),
                        "nu": float(g.params.get("nu", np.nan))}
    except Exception as e:  # arch missing -> bootstrap residuals
        out["garch_error"] = str(e)
        boot = np.random.RandomState(seed).choice(r - r.mean(), size=(horizon, sims)) + params[0]
        gp = S0 * np.exp(np.cumsum(boot, axis=0))
        out["garch"] = {"order": "Bootstrap (arch not installed)", "aic": float("nan"), "paths": _percentiles(gp),
                        "final": gp[-1], "cond_vol_ann": float(r.std() * math.sqrt(252)), "nu": float("nan")}
    # consensus
    out["consensus"] = {k: (out["arima"]["paths"][k] + out["garch"]["paths"][k]) / 2 for k in out["arima"]["paths"]}
    out["prob"] = {}
    for name, h in HORIZONS.items():
        pa = out["arima"]["paths"]["p50"][h - 1]
        ends = np.concatenate([_end_at(out, "arima", h), _end_at(out, "garch", h)])
        out["prob"][name] = {"median": float(np.median(ends)), "p5": float(np.percentile(ends, 5)),
                             "p95": float(np.percentile(ends, 95)), "p_up": float((ends > S0).mean()),
                             "p_up20": float((ends > S0 * 1.2).mean()), "p_dn20": float((ends < S0 * 0.8).mean()),
                             "ret_median": float(np.median(ends) / S0 - 1), "arima_median": float(pa)}
    return out


def _percentiles(paths: np.ndarray) -> dict:
    return {f"p{p}": np.percentile(paths, p, axis=1) for p in (5, 25, 50, 75, 95)}


def _end_at(out, model, h):
    """Quantile-reconstructed sample at horizon h from stored percentile paths (keeps memory small)."""
    pp = out[model]["paths"]
    qs = np.array([5, 25, 50, 75, 95])
    vals = np.array([pp[f"p{q}"][h - 1] for q in qs])
    grid = np.linspace(2, 98, 400)
    return np.interp(grid, qs, vals)


def walk_forward(close: pd.Series, order=(0, 0), test_days=252, step=21, hz=21, sims=300) -> dict:
    from statsmodels.tsa.arima.model import ARIMA
    lp = np.log(close.astype(float))
    r = lp.diff().dropna().values
    px = close.values
    n = len(r)
    errs_m, errs_n, dirs, cover = [], [], [], []
    for i in range(n - test_days, n - hz, step):
        try:
            m = ARIMA(r[:i], order=(order[0], 0, order[1]), trend="c").fit()
            np.random.seed(i)
            sim = np.asarray(m.simulate(nsimulations=hz, repetitions=sims, anchor="end")).reshape(hz, sims)
        except Exception:
            continue
        S0 = px[i]       # price index offset: r[i-1] is the return into px[i]
        ends = S0 * np.exp(sim.sum(axis=0))
        actual = px[i + hz]
        med = np.median(ends)
        errs_m.append(abs(med - actual) / actual)
        errs_n.append(abs(S0 - actual) / actual)
        dirs.append(np.sign(med - S0) == np.sign(actual - S0))
        cover.append(np.percentile(ends, 5) <= actual <= np.percentile(ends, 95))
    if not errs_m:
        return {}
    return {"n": len(errs_m), "mape_model": float(np.mean(errs_m)), "mape_naive": float(np.mean(errs_n)),
            "dir_acc": float(np.mean(dirs)), "coverage90": float(np.mean(cover)), "hz": hz}


# ----------------------------------------------------------------------------------------------
# 6. OPTIONS (Black-Scholes greeks + strategy ideas)
# ----------------------------------------------------------------------------------------------
def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs(S, K, T, r, sigma, kind="c"):
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if kind == "c" else max(K - S, 0)
        return {"price": intrinsic, "delta": 1.0 if (kind == "c" and S > K) else (-1.0 if kind == "p" and S < K else 0.0),
                "gamma": 0.0, "theta": 0.0, "vega": 0.0, "d2": 0.0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "c":
        price = S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
        delta = _ncdf(d1)
        theta = (-S * _npdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * _ncdf(d2)) / 365
    else:
        price = K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)
        delta = _ncdf(d1) - 1
        theta = (-S * _npdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365
    return {"price": price, "delta": delta, "gamma": _npdf(d1) / (S * sigma * math.sqrt(T)),
            "theta": theta, "vega": S * _npdf(d1) * math.sqrt(T) / 100, "d2": d2}


def prob_above(S, X, T, r, sigma):
    if T <= 0 or sigma <= 0 or X <= 0:
        return float(S > X)
    d2 = (math.log(S / X) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _ncdf(d2)


def get_options(ticker: str, targets=(7, 91, 182, 365)) -> dict:
    """Pick, for each target horizon, the listed expiry nearest to it and download that chain."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    exps = list(t.options or [])
    today = pd.Timestamp.today().normalize()
    out = {"expiries": exps, "chains": {}}
    for tgt in targets:
        if not exps:
            break
        e = min(exps, key=lambda x: abs((pd.Timestamp(x) - today).days - tgt))
        if e in out["chains"]:
            continue
        ch = t.option_chain(e)
        out["chains"][e] = {"calls": ch.calls, "puts": ch.puts, "dte": max((pd.Timestamp(e) - today).days, 1)}
    return out


def _prep(df: pd.DataFrame, S, T, r, kind) -> pd.DataFrame:
    d = df.copy()
    d["mid"] = np.where((d["bid"] > 0) & (d["ask"] > 0), (d["bid"] + d["ask"]) / 2, d["lastPrice"])
    d = d[(d["mid"] > 0) & (d["impliedVolatility"] > 0.05)].copy()
    d["delta"] = [bs(S, k, T, r, iv, kind)["delta"] for k, iv in zip(d["strike"], d["impliedVolatility"])]
    d["spread_pct"] = np.where(d["mid"] > 0, (d["ask"] - d["bid"]) / d["mid"], np.nan)
    return d


def atm_iv(chains: dict, S) -> float:
    ivs = []
    for e, ch in chains.items():
        for side in ("calls", "puts"):
            d = ch[side]
            d = d[(d["impliedVolatility"] > 0.05)]
            if len(d):
                row = d.iloc[(d["strike"] - S).abs().argsort()[:1]]
                ivs.append(float(row["impliedVolatility"].iloc[0]))
    return float(np.median(ivs)) if ivs else float("nan")


def payoff_curve(kind, S, legs, premium_net, lo=0.4, hi=1.8, n=60):
    xs = np.linspace(S * lo, S * hi, n)
    pl = []
    for x in xs:
        v = 0.0
        for leg in legs:
            q = leg["qty"]
            if leg["type"] == "stock":
                v += q * (x - S)
            elif leg["type"] == "call":
                v += q * (max(x - leg["K"], 0))
            else:
                v += q * (max(leg["K"] - x, 0))
        pl.append(v + premium_net)
    return xs.tolist(), pl


def option_ideas(chain: dict, S: float, r: float, bias: float, iv_hv: float, rsi_v: float,
                 cc_delta=0.30, csp_delta=0.25) -> list[dict]:
    T = chain["dte"] / 365
    calls, puts = _prep(chain["calls"], S, T, r, "c"), _prep(chain["puts"], S, T, r, "p")
    if calls.empty or puts.empty:
        return []
    atm = float(np.median([calls.iloc[(calls["strike"] - S).abs().argsort()[:1]]["impliedVolatility"].iloc[0],
                           puts.iloc[(puts["strike"] - S).abs().argsort()[:1]]["impliedVolatility"].iloc[0]]))
    otm_c = calls[calls["strike"] > S]
    otm_p = puts[puts["strike"] < S]
    ideas = []

    def pick(df, target):
        return df.iloc[(df["delta"].abs() - target).abs().argsort()[:1]].iloc[0] if len(df) else None

    def liq(rows):
        sp = max(float(x["spread_pct"]) if pd.notna(x["spread_pct"]) else 1.0 for x in rows)
        oi = min(float(x.get("openInterest", 0) or 0) for x in rows)
        return "Good" if sp < 0.15 and oi >= 100 else ("OK" if sp < 0.35 else "Thin / wide spread"), sp, oi

    def fit(name):
        sell_rich = iv_hv >= 1.05
        if name == "Covered call":
            if bias > 0.35:
                return "Neutral", "Strong bullish bias — a short call caps upside; use higher strike / fewer contracts."
            return ("Preferred" if sell_rich and bias > -0.45 else "Neutral",
                    f"IV/HV {iv_hv:.2f}: premium {'rich' if sell_rich else 'fair'}; bias {bias:+.2f}.")
        if name == "Cash-secured put":
            if bias < -0.4:
                return "Avoid", "Bearish bias — assignment risk against a falling stock."
            return ("Preferred" if sell_rich and bias >= -0.2 and rsi_v < 65 else "Neutral",
                    f"IV/HV {iv_hv:.2f}; paid to wait for a lower entry; bias {bias:+.2f}.")
        if name == "Put credit spread":
            return ("Preferred" if bias > 0 and sell_rich else "Neutral",
                    "Defined-risk bullish premium sale; fits a low-priced, high-IV stock with 30%+ short float.")
        if name == "Bull call spread":
            return ("Preferred" if bias > 0.3 and iv_hv < 1.0 else ("Neutral" if bias > 0.1 else "Avoid"),
                    f"Debit strategy; best when IV is cheap (IV/HV {iv_hv:.2f}) and bias is bullish ({bias:+.2f}).")
        if name == "Collar":
            return ("Preferred" if bias < -0.2 else "Neutral",
                    "Protects shares against gap risk (earnings, merger vote, short squeeze reversal).")
        return "Neutral", ""

    # Covered call
    c = pick(otm_c, cc_delta)
    if c is not None:
        prem, K = float(c["mid"]), float(c["strike"])
        lq, sp, oi = liq([c])
        lab, why = fit("Covered call")
        xs, pl = payoff_curve("cc", S, [{"type": "stock", "qty": 1}, {"type": "call", "K": K, "qty": -1}], prem)
        ideas.append({"name": "Covered call", "legs": f"Own 100 sh @ ${S:.2f} + Sell {K:g} Call @ ${prem:.2f}",
                      "credit": prem, "max_profit": (K - S + prem), "max_loss": S - prem,
                      "breakeven": S - prem, "ret_static": prem / S, "ret_called": (K - S + prem) / S,
                      "ret_ann": prem / S * 365 / chain["dte"], "pop": prob_above(S, S - prem, T, r, atm),
                      "delta": float(c["delta"]), "iv": float(c["impliedVolatility"]), "liq": lq, "spread": sp, "oi": oi,
                      "fit": lab, "why": why, "capital": 100 * S, "x": xs, "pl": pl, "strike": K})
    # CSP
    p = pick(otm_p, csp_delta)
    if p is not None:
        prem, K = float(p["mid"]), float(p["strike"])
        lq, sp, oi = liq([p])
        lab, why = fit("Cash-secured put")
        xs, pl = payoff_curve("csp", S, [{"type": "put", "K": K, "qty": -1}], prem)
        ideas.append({"name": "Cash-secured put", "legs": f"Sell {K:g} Put @ ${prem:.2f} (cash ${100 * K:,.0f})",
                      "credit": prem, "max_profit": prem, "max_loss": K - prem, "breakeven": K - prem,
                      "ret_static": prem / K, "ret_called": prem / K, "ret_ann": prem / K * 365 / chain["dte"],
                      "pop": prob_above(S, K - prem, T, r, atm), "delta": float(p["delta"]),
                      "iv": float(p["impliedVolatility"]), "liq": lq, "spread": sp, "oi": oi, "fit": lab, "why": why,
                      "capital": 100 * K, "x": xs, "pl": pl, "strike": K,
                      "discount": (S - (K - prem)) / S})
        # put credit spread
        lower = otm_p[otm_p["strike"] < K].sort_values("strike", ascending=False)
        if len(lower):
            w = lower.iloc[min(1, len(lower) - 1)]
            cr = prem - float(w["mid"])
            width = K - float(w["strike"])
            if cr > 0 and width > 0:
                lq2, sp2, oi2 = liq([p, w])
                lab, why = fit("Put credit spread")
                xs, pl = payoff_curve("pcs", S, [{"type": "put", "K": K, "qty": -1},
                                                 {"type": "put", "K": float(w["strike"]), "qty": 1}], cr)
                ideas.append({"name": "Put credit spread",
                              "legs": f"Sell {K:g}P / Buy {float(w['strike']):g}P for ${cr:.2f} credit",
                              "credit": cr, "max_profit": cr, "max_loss": width - cr, "breakeven": K - cr,
                              "ret_static": cr / (width - cr), "ret_called": cr / (width - cr),
                              "ret_ann": cr / (width - cr) * 365 / chain["dte"],
                              "pop": prob_above(S, K - cr, T, r, atm), "delta": float(p["delta"] - w["delta"]),
                              "iv": float(p["impliedVolatility"]), "liq": lq2, "spread": sp2, "oi": oi2,
                              "fit": lab, "why": why, "capital": 100 * (width - cr), "x": xs, "pl": pl, "strike": K})
    # Bull call spread
    if len(calls):
        long = calls.iloc[(calls["strike"] - S).abs().argsort()[:1]].iloc[0]
        short = pick(otm_c[otm_c["strike"] > float(long["strike"])], 0.25) if len(otm_c) else None
        if short is not None:
            debit = float(long["mid"]) - float(short["mid"])
            width = float(short["strike"]) - float(long["strike"])
            if debit > 0 and width > debit:
                lq2, sp2, oi2 = liq([long, short])
                lab, why = fit("Bull call spread")
                xs, pl = payoff_curve("bcs", S, [{"type": "call", "K": float(long["strike"]), "qty": 1},
                                                 {"type": "call", "K": float(short["strike"]), "qty": -1}], -debit)
                ideas.append({"name": "Bull call spread",
                              "legs": f"Buy {float(long['strike']):g}C / Sell {float(short['strike']):g}C for ${debit:.2f} debit",
                              "credit": -debit, "max_profit": width - debit, "max_loss": debit,
                              "breakeven": float(long["strike"]) + debit, "ret_static": (width - debit) / debit,
                              "ret_called": (width - debit) / debit, "ret_ann": float("nan"),
                              "pop": prob_above(S, float(long["strike"]) + debit, T, r, atm),
                              "delta": float(long["delta"] - short["delta"]), "iv": float(long["impliedVolatility"]),
                              "liq": lq2, "spread": sp2, "oi": oi2, "fit": lab, "why": why,
                              "capital": 100 * debit, "x": xs, "pl": pl, "strike": float(long["strike"])})
    # Collar
    cp, cc_ = pick(otm_p, 0.20), pick(otm_c, cc_delta)
    if cp is not None and cc_ is not None:
        net = float(cc_["mid"]) - float(cp["mid"])
        lq2, sp2, oi2 = liq([cp, cc_])
        lab, why = fit("Collar")
        xs, pl = payoff_curve("collar", S, [{"type": "stock", "qty": 1}, {"type": "put", "K": float(cp["strike"]), "qty": 1},
                                            {"type": "call", "K": float(cc_["strike"]), "qty": -1}], net)
        ideas.append({"name": "Collar",
                      "legs": f"Own shares + Buy {float(cp['strike']):g}P / Sell {float(cc_['strike']):g}C (net ${net:+.2f})",
                      "credit": net, "max_profit": float(cc_["strike"]) - S + net,
                      "max_loss": S - float(cp["strike"]) - net, "breakeven": S - net, "ret_static": net / S,
                      "ret_called": (float(cc_["strike"]) - S + net) / S, "ret_ann": net / S * 365 / chain["dte"],
                      "pop": prob_above(S, S - net, T, r, atm), "delta": float(cc_["delta"] + cp["delta"]),
                      "iv": float(cc_["impliedVolatility"]), "liq": lq2, "spread": sp2, "oi": oi2, "fit": lab,
                      "why": why, "capital": 100 * S, "x": xs, "pl": pl, "strike": float(cc_["strike"])})
    for i in ideas:
        i["dte"] = chain["dte"]
        i["atm_iv"] = atm
    return ideas


# ----------------------------------------------------------------------------------------------
# 7. SCORING & HORIZON INSIGHTS
# ----------------------------------------------------------------------------------------------
def tech_score(d: pd.DataFrame, sig: pd.DataFrame) -> tuple[float, list[str]]:
    x, notes, s = d.iloc[-1], [], 0.0
    c = x["Close"]
    for n, w in ((50, 15), (200, 15)):
        up = c > x[f"EMA{n}"]
        s += w if up else -w
        notes.append(f"Price {'above' if up else 'below'} EMA{n} ({x[f'EMA{n}']:.2f})")
    stack_up, stack_dn = x["EMA7"] > x["EMA14"] > x["EMA28"], x["EMA7"] < x["EMA14"] < x["EMA28"]
    if stack_up:
        s += 10; notes.append("Short-term EMAs stacked bullish (7>14>28)")
    elif stack_dn:
        s -= 10; notes.append("Short-term EMAs stacked bearish (7<14<28)")
    s += 10 if x["MACDh"] > 0 else -10
    notes.append(f"MACD histogram {'positive' if x['MACDh'] > 0 else 'negative'}")
    rv = x["RSI"]
    if rv > 70:
        s -= 10; notes.append(f"RSI {rv:.0f} overbought")
    elif rv < 30:
        s += 10; notes.append(f"RSI {rv:.0f} oversold")
    else:
        s += (rv - 50) / 5
        notes.append(f"RSI {rv:.0f} neutral-{'bullish' if rv > 50 else 'bearish'}")
    pb = (c - x["BBl"]) / (x["BBu"] - x["BBl"]) if x["BBu"] > x["BBl"] else 0.5
    if pb < 0.1:
        s += 10; notes.append("Price at lower Bollinger band (mean-reversion bid)")
    elif pb > 0.9:
        s -= 10; notes.append("Price at upper Bollinger band (stretched)")
    if len(sig):
        rec = sig[(sig["state"].isin(["confirmed", "pending"])) & (sig["date"] >= d.index[-10])]
        if len(rec):
            last = rec.iloc[-1]
            s += 20 if last["side"] == "BUY" else -20
            notes.append(f"Recent {last['state']} {last['side']} signal: {last['pattern']} ({last['date']:%b %d})")
    return float(max(-100, min(100, s))), notes


def fundamental_score(info: dict, finviz: dict, facts=FACTS) -> tuple[float, list[str]]:
    s, notes = 0.0, []
    sn = finviz.get("snapshot", {}) if finviz else {}

    def num(v):
        try:
            return float(str(v).replace("%", "").replace(",", ""))
        except Exception:
            return None
    fpe = info.get("forwardPE") or num(sn.get("Forward P/E"))
    ps = info.get("priceToSalesTrailing12Months") or num(sn.get("P/S"))
    if fpe:
        s += 25 if fpe < 8 else 15 if fpe < 12 else -15 if fpe > 25 else 0
        notes.append(f"Forward P/E {fpe:.1f}x")
    if ps:
        s += 15 if ps < 1 else 0 if ps < 2.5 else -10
        notes.append(f"P/S {ps:.2f}x")
    s += 15; notes.append(f"Record orderbook ${facts['orderbook']['usd_b'][-1]}B (+{facts['orderbook_yoy']:.0%} YoY)")
    s += 10; notes.append(f"Book-to-bill {facts['book_to_bill_ttm']}x")
    nd = facts["liquidity"]["net_debt_ebitda"]
    s += -15 if nd > 3.5 else 0
    notes.append(f"Net debt / EBITDA {nd}x (elevated)")
    s += -10; notes.append("GAAP net loss to common in 3Q26; EBITDA margin 9.7% vs 18.3% a year ago")
    tgt, px = info.get("targetMeanPrice"), info.get("currentPrice") or info.get("regularMarketPrice")
    if tgt and px:
        up = tgt / px - 1
        s += 15 if up > 0.2 else 5 if up > 0 else -10
        notes.append(f"Analyst mean target ${tgt:.2f} ({up:+.0%})")
    sf = info.get("shortPercentOfFloat")
    if sf:
        notes.append(f"Short interest {sf:.0%} of float (squeeze fuel, but bearish conviction)")
        s += -5 if sf > 0.25 else 0
    s += -5; notes.append("FY26 guidance cut (revenue $1.34-1.40B vs $1.4-1.5B)")
    return float(max(-100, min(100, s))), notes


def news_score(items: list[dict]) -> float:
    if not items:
        return 0.0
    return float(max(-40, min(40, 100 * np.mean([i.get("sent", 0) for i in items[:15]]))))


HORIZON_W = {"1 week": (0.55, 0.20, 0.10, 0.15), "3 months": (0.25, 0.25, 0.35, 0.15),
             "6 months": (0.15, 0.25, 0.50, 0.10), "12 months": (0.05, 0.20, 0.70, 0.05)}


def rating(score: float) -> str:
    return ("Bullish" if score >= 25 else "Mild bullish" if score >= 10 else "Neutral" if score > -10
            else "Mild bearish" if score > -25 else "Bearish")


def horizon_insights(d, sig, fc, info, finviz, news, hv30) -> dict:
    ts, tn = tech_score(d, sig)
    fs, fn = fundamental_score(info, finviz)
    ns = news_score(news)
    S = float(d["Close"].iloc[-1])
    out = {}
    for h, (wt, wf, wv, wn) in HORIZON_W.items():
        pr = fc["prob"][h]
        fsc = 0.5 * max(-100, min(100, (pr["p_up"] - 0.5) * 200)) + 0.5 * max(-100, min(100, pr["ret_median"] / 0.2 * 100))
        comp = wt * ts + wf * fsc + wv * fs + wn * ns
        out[h] = {"score": float(comp), "rating": rating(comp), "tech": ts, "forecast": float(fsc), "fund": fs,
                  "news": ns, "bias": float(max(-1, min(1, comp / 50))), **pr,
                  "exp_move_1sd": float(S * hv30 * math.sqrt(HORIZONS[h] / 252))}
    return {"horizons": out, "tech_notes": tn, "fund_notes": fn, "S": S}


def realized_vol(close: pd.Series, n=30) -> float:
    return float(np.log(close).diff().rolling(n).std().iloc[-1] * math.sqrt(252))


def hv_percentile(close: pd.Series, n=30) -> float:
    hv = np.log(close).diff().rolling(n).std().dropna() * math.sqrt(252)
    return float((hv < hv.iloc[-1]).mean())


# ----------------------------------------------------------------------------------------------
# 8. FUNDAMENTAL SERIES
# ----------------------------------------------------------------------------------------------
def _row(df: pd.DataFrame, names: list[str]):
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def quarterly_series(stm: dict, shares: float | None) -> pd.DataFrame:
    inc, cf, bs_ = stm.get("income", pd.DataFrame()), stm.get("cashflow", pd.DataFrame()), stm.get("balance", pd.DataFrame())
    if inc.empty:
        return pd.DataFrame()
    cols = sorted(inc.columns)
    q = pd.DataFrame(index=pd.to_datetime(cols))
    mp = {"Revenue": ["Total Revenue", "Operating Revenue"], "GrossProfit": ["Gross Profit"],
          "OpIncome": ["Operating Income"], "NetIncome": ["Net Income Common Stockholders", "Net Income"],
          "EPS": ["Diluted EPS", "Basic EPS"], "EBITDA": ["EBITDA", "Normalized EBITDA"]}
    for k, names in mp.items():
        r = _row(inc, names)
        if r is not None:
            q[k] = r.reindex(cols).values
    for k, names, src in (("OCF", ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], cf),
                          ("Capex", ["Capital Expenditure", "Purchase Of PPE"], cf),
                          ("FCF", ["Free Cash Flow"], cf)):
        if src is not None and not src.empty:
            r = _row(src, names)
            if r is not None:
                q[k] = r.reindex(cols).values
    if bs_ is not None and not bs_.empty:
        for k, names in (("Cash", ["Cash And Cash Equivalents"]), ("Debt", ["Total Debt"]),
                         ("Shares", ["Ordinary Shares Number", "Share Issued"])):
            r = _row(bs_, names)
            if r is not None:
                q[k] = r.reindex(cols).values
    sh = q["Shares"].fillna(shares or np.nan) if "Shares" in q else pd.Series(shares, index=q.index)
    q["Shares"] = sh
    if "Revenue" in q:
        q["GM"] = q.get("GrossProfit", np.nan) / q["Revenue"]
        q["OM"] = q.get("OpIncome", np.nan) / q["Revenue"]
        q["Rev_YoY"] = q["Revenue"].pct_change(4)
    if "OCF" in q:
        if "FCF" not in q or q["FCF"].isna().all():
            q["FCF"] = q["OCF"] + q.get("Capex", 0)
        q["OCF_ps"], q["FCF_ps"] = q["OCF"] / q["Shares"], q["FCF"] / q["Shares"]
    return q


def valuation_history(close: pd.Series, q: pd.DataFrame, shares: float | None) -> pd.DataFrame:
    """P/E (TTM, only where TTM EPS>0), P/S (TTM revenue / share) as-of each trading day (45d reporting lag)."""
    if q.empty or "Revenue" not in q:
        return pd.DataFrame(index=close.index)
    t = pd.DataFrame(index=q.index)
    t["rev_ttm"] = q["Revenue"].rolling(4).sum()
    if "EPS" in q:
        t["eps_ttm"] = q["EPS"].rolling(4).sum()
    t["shares"] = q["Shares"]
    t.index = t.index + pd.Timedelta(days=45)
    t = t.dropna(subset=["rev_ttm"])
    if t.empty:
        return pd.DataFrame(index=close.index)
    t = t.reindex(close.index, method="ffill")
    v = pd.DataFrame(index=close.index)
    v["PS"] = close * t["shares"] / t["rev_ttm"]
    if "eps_ttm" in t:
        v["PE"] = (close / t["eps_ttm"]).where(t["eps_ttm"] > 0)
    return v


def monthly_returns_heatmap(close: pd.Series) -> pd.DataFrame:
    m = close.resample("ME").last().pct_change().dropna() if hasattr(close, "resample") else None
    t = pd.DataFrame({"y": m.index.year, "m": m.index.month, "r": m.values})
    return t


def comparison_table(tickers: list[str], finviz_fn=get_finviz) -> pd.DataFrame:
    rows = []
    for t in tickers:
        fv = finviz_fn(t).get("snapshot", {})
        if fv:
            rows.append({"Ticker": t, **{k: fv.get(k, "-") for k in
                                         ["Market Cap", "P/E", "Forward P/E", "PEG", "P/S", "P/B", "EV/EBITDA", "ROE",
                                          "ROIC", "Gross Margin", "Oper. Margin", "Profit Margin", "Debt/Eq",
                                          "Short Float", "Perf Month", "Perf YTD", "Perf Year", "RSI (14)", "Beta"]}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# 9. ECHARTS OPTION BUILDERS (Streamlit)
# ----------------------------------------------------------------------------------------------
TRADER = {"bg": "#0b0f14", "panel": "#111821", "grid": "#1f2a37", "text": "#c9d4e0", "amber": "#ffb000",
          "green": "#00d26a", "red": "#ff4d4f", "cyan": "#39c5ff", "violet": "#b18cff", "muted": "#7d8b9c"}
EMA_COLORS = {7: "#39c5ff", 14: "#7fffd4", 28: "#ffb000", 50: "#ff7ac6", 200: "#b18cff"}


def _f(x, nd=3):
    return None if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else round(float(x), nd)


def tech_option(d: pd.DataFrame, sig: pd.DataFrame, emas, show_bb, show_vol, show_macd, rsi_n, rsi_ob, rsi_os,
                show_filtered=False, title="") -> dict:
    dates = [x.strftime("%Y-%m-%d") for x in d.index]
    candles = [[_f(o), _f(c), _f(l), _f(h)] for o, c, l, h in zip(d["Open"], d["Close"], d["Low"], d["High"])]
    n_panels = 2 + int(show_vol) + int(show_macd)
    heights = [58] + ([12] if show_vol else []) + [14] + ([14] if show_macd else [])
    tops, y = [], 4
    for hgt in heights:
        tops.append(y); y += hgt + 3
    grids = [{"left": 55, "right": 25, "top": f"{t}%", "height": f"{hh}%"} for t, hh in zip(tops, heights)]
    xaxes = [{"type": "category", "data": dates, "gridIndex": i, "boundaryGap": True,
              "axisLine": {"lineStyle": {"color": TRADER["grid"]}}, "axisLabel": {"show": i == len(grids) - 1,
              "color": TRADER["muted"]}, "axisTick": {"show": False}, "splitLine": {"show": False}} for i in range(len(grids))]
    yaxes = [{"scale": True, "gridIndex": i, "splitNumber": 4, "axisLabel": {"color": TRADER["muted"]},
              "splitLine": {"lineStyle": {"color": TRADER["grid"]}}} for i in range(len(grids))]
    yaxes[0].update({"min": round(float(d["Low"].min()) * 0.9, 2), "max": round(float(d["High"].max()) * 1.08, 2)})
    series = [{"name": "Price", "type": "candlestick", "data": candles, "xAxisIndex": 0, "yAxisIndex": 0,
               "itemStyle": {"color": TRADER["green"], "color0": TRADER["red"], "borderColor": TRADER["green"],
                             "borderColor0": TRADER["red"]}}]
    for n in emas:
        series.append({"name": f"EMA {n}", "type": "line", "data": [_f(v) for v in d[f"EMA{n}"]], "xAxisIndex": 0,
                       "yAxisIndex": 0, "showSymbol": False, "smooth": False,
                       "lineStyle": {"width": 1.3, "color": EMA_COLORS.get(n, "#fff")}})
    if show_bb:
        for nm, col, dash in (("BB upper", "BBu", "dashed"), ("BB mid", "BBm", "dotted"), ("BB lower", "BBl", "dashed")):
            series.append({"name": nm, "type": "line", "data": [_f(v) for v in d[col]], "xAxisIndex": 0,
                           "yAxisIndex": 0, "showSymbol": False,
                           "lineStyle": {"width": 1, "color": "#8aa4c8", "type": dash, "opacity": .8}})
    # signals
    if len(sig):
        for state, side, color, sym, rot, size in (("confirmed", "BUY", TRADER["green"], "triangle", 0, 16),
                                                   ("confirmed", "SELL", TRADER["red"], "triangle", 180, 16),
                                                   ("pending", "BUY", TRADER["green"], "diamond", 0, 12),
                                                   ("pending", "SELL", TRADER["red"], "diamond", 0, 12)):
            sub = sig[(sig["state"] == state) & (sig["side"] == side)]
            if sub.empty:
                continue
            data = [{"value": [x.strftime("%Y-%m-%d"), _f(r.low * 0.965 if side == "BUY" else r.high * 1.035)],
                     "label": r.pattern, "sc": int(r.score)} for x, r in zip(sub["date"], sub.itertuples())]
            series.append({"name": f"{side} ({state})", "type": "scatter", "data": data, "symbol": sym,
                           "symbolRotate": rot, "symbolSize": size, "xAxisIndex": 0, "yAxisIndex": 0,
                           "itemStyle": {"color": color, "borderColor": "#fff", "borderWidth": 1},
                           "tooltip": {"formatter": "{a}<br/>{c}"}, "z": 10})
        if show_filtered:
            sub = sig[sig["state"] == "filtered"]
            if len(sub):
                data = [[x.strftime("%Y-%m-%d"), _f(r.low * 0.985 if r.side == "BUY" else r.high * 1.015)]
                        for x, r in zip(sub["date"], sub.itertuples())]
                series.append({"name": "Filtered (misleading)", "type": "scatter", "data": data, "symbol": "circle",
                               "symbolSize": 6, "xAxisIndex": 0, "yAxisIndex": 0,
                               "itemStyle": {"color": "transparent", "borderColor": TRADER["muted"], "borderWidth": 1}})
    p = 1
    if show_vol:
        series.append({"name": "Volume", "type": "bar", "xAxisIndex": p, "yAxisIndex": p,
                       "data": [{"value": int(v), "itemStyle": {"color": TRADER["green"] if c >= o else TRADER["red"]}}
                                for v, o, c in zip(d["Volume"], d["Open"], d["Close"])]})
        series.append({"name": "Vol SMA20", "type": "line", "xAxisIndex": p, "yAxisIndex": p, "showSymbol": False,
                       "data": [_f(v, 0) for v in d["VolSMA20"]], "lineStyle": {"width": 1, "color": TRADER["amber"]}})
        p += 1
    series.append({"name": f"RSI({rsi_n})", "type": "line", "xAxisIndex": p, "yAxisIndex": p, "showSymbol": False,
                   "data": [_f(v, 2) for v in d["RSI"]], "lineStyle": {"width": 1.4, "color": TRADER["amber"]},
                   "markLine": {"silent": True, "symbol": "none", "lineStyle": {"color": TRADER["muted"], "type": "dashed"},
                                "data": [{"yAxis": rsi_ob}, {"yAxis": rsi_os}]}})
    yaxes[p].update({"min": 0, "max": 100, "scale": False})
    p += 1
    if show_macd:
        series.append({"name": "MACD hist", "type": "bar", "xAxisIndex": p, "yAxisIndex": p,
                       "data": [{"value": _f(v, 4), "itemStyle": {"color": TRADER["green"] if (v or 0) >= 0 else TRADER["red"]}}
                                for v in d["MACDh"]]})
        series.append({"name": "MACD", "type": "line", "xAxisIndex": p, "yAxisIndex": p, "showSymbol": False,
                       "data": [_f(v, 4) for v in d["MACD"]], "lineStyle": {"width": 1, "color": TRADER["cyan"]}})
        series.append({"name": "Signal", "type": "line", "xAxisIndex": p, "yAxisIndex": p, "showSymbol": False,
                       "data": [_f(v, 4) for v in d["MACDs"]], "lineStyle": {"width": 1, "color": TRADER["amber"]}})
    return {"backgroundColor": TRADER["bg"], "animation": False, "title": {"text": title, "textStyle": {"color": TRADER["amber"]}},
            "legend": {"top": 0, "right": 10, "textStyle": {"color": TRADER["text"]}, "type": "scroll"},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
            "axisPointer": {"link": [{"xAxisIndex": "all"}]},
            "grid": grids, "xAxis": xaxes, "yAxis": yaxes, "series": series,
            "dataZoom": [{"type": "inside", "xAxisIndex": list(range(len(grids))), "start": 55, "end": 100},
                         {"type": "slider", "xAxisIndex": list(range(len(grids))), "bottom": 4, "height": 18,
                          "start": 55, "end": 100, "textStyle": {"color": TRADER["muted"]}}]}


def forecast_option(close: pd.Series, fc: dict, horizon: int, hist_days=180) -> dict:
    S0 = fc["S0"]
    hist = close.iloc[-hist_days:]
    hd = [x.strftime("%Y-%m-%d") for x in hist.index]
    fd = [x.strftime("%Y-%m-%d") for x in pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=horizon)]
    cons = fc["consensus"]
    pad = [None] * (len(hd) - 1)
    base = pad + [_f(S0)] + [_f(v) for v in cons["p5"][:horizon]]
    band = pad + [0] + [_f(a - b) for a, b in zip(cons["p95"][:horizon], cons["p5"][:horizon])]
    base50 = pad + [_f(S0)] + [_f(v) for v in cons["p25"][:horizon]]
    band50 = pad + [0] + [_f(a - b) for a, b in zip(cons["p75"][:horizon], cons["p25"][:horizon])]
    series = [
        {"name": "5-95% band (base)", "type": "line", "stack": "b90", "data": base, "lineStyle": {"opacity": 0}, "showSymbol": False, "tooltip": {"show": False}},
        {"name": "5–95% band", "type": "line", "stack": "b90", "data": band, "lineStyle": {"opacity": 0}, "showSymbol": False,
         "areaStyle": {"color": "rgba(57,197,255,0.15)"}},
        {"name": "25-75% band (base)", "type": "line", "stack": "b50", "data": base50, "lineStyle": {"opacity": 0}, "showSymbol": False, "tooltip": {"show": False}},
        {"name": "25–75% band", "type": "line", "stack": "b50", "data": band50, "lineStyle": {"opacity": 0}, "showSymbol": False,
         "areaStyle": {"color": "rgba(255,176,0,0.25)"}},
        {"name": "Price", "type": "line", "data": [_f(v) for v in hist.values] + [None] * horizon, "showSymbol": False,
         "lineStyle": {"color": TRADER["text"], "width": 1.6}},
        {"name": "ARIMA median", "type": "line", "data": pad + [_f(S0)] + [_f(v) for v in fc["arima"]["paths"]["p50"][:horizon]],
         "showSymbol": False, "lineStyle": {"color": TRADER["cyan"], "type": "dashed"}},
        {"name": "ARIMA-GARCH median", "type": "line", "data": pad + [_f(S0)] + [_f(v) for v in fc["garch"]["paths"]["p50"][:horizon]],
         "showSymbol": False, "lineStyle": {"color": TRADER["violet"], "type": "dashed"}},
        {"name": "Consensus median", "type": "line", "data": pad + [_f(S0)] + [_f(v) for v in cons["p50"][:horizon]],
         "showSymbol": False, "lineStyle": {"color": TRADER["amber"], "width": 2}},
    ]
    return {"backgroundColor": TRADER["bg"], "tooltip": {"trigger": "axis"},
            "legend": {"top": 0, "textStyle": {"color": TRADER["text"]},
                       "data": ["Price", "ARIMA median", "ARIMA-GARCH median", "Consensus median", "5–95% band", "25–75% band"]},
            "grid": {"left": 55, "right": 25, "top": 45, "bottom": 50},
            "xAxis": {"type": "category", "data": hd + fd, "axisLabel": {"color": TRADER["muted"]}},
            "yAxis": {"scale": True, "min": max(0, math.floor(min(float(hist.min()), float(cons["p5"][:horizon].min())) * 0.85)),
                      "axisLabel": {"color": TRADER["muted"]}, "splitLine": {"lineStyle": {"color": TRADER["grid"]}}},
            "dataZoom": [{"type": "inside"}], "series": series}


def bar_option(cats, series_map, title="", colors=None, stack=False, pct=False, horizontal=False) -> dict:
    colors = colors or [TRADER["amber"], TRADER["cyan"], TRADER["violet"], TRADER["green"], TRADER["red"]]
    ser = []
    for i, (n, v) in enumerate(series_map.items()):
        s = {"name": n, "type": "bar", "data": [_f(x, 3) for x in v], "itemStyle": {"color": colors[i % len(colors)]},
             "label": {"show": True, "position": "top" if not stack else "inside", "color": TRADER["text"],
                       "fontSize": 10, "formatter": "{c}%" if pct else "{c}"}}
        if stack:
            s["stack"] = "t"
        ser.append(s)
    cat_ax = {"type": "category", "data": list(cats), "axisLabel": {"color": TRADER["muted"]}}
    val_ax = {"type": "value", "axisLabel": {"color": TRADER["muted"]}, "splitLine": {"lineStyle": {"color": TRADER["grid"]}}}
    return {"backgroundColor": TRADER["bg"], "title": {"text": title, "textStyle": {"color": TRADER["amber"], "fontSize": 13}},
            "tooltip": {"trigger": "axis"}, "legend": {"top": 0, "right": 10, "textStyle": {"color": TRADER["text"]}},
            "grid": {"left": 55, "right": 20, "top": 50, "bottom": 30},
            "xAxis": val_ax if horizontal else cat_ax, "yAxis": cat_ax if horizontal else val_ax, "series": ser}


def pie_option(data: dict, title="") -> dict:
    return {"backgroundColor": TRADER["bg"], "title": {"text": title, "textStyle": {"color": TRADER["amber"], "fontSize": 13}},
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            "legend": {"bottom": 0, "textStyle": {"color": TRADER["text"]}},
            "color": [TRADER["amber"], TRADER["cyan"], TRADER["violet"], TRADER["green"], TRADER["red"]],
            "series": [{"type": "pie", "radius": ["38%", "68%"], "center": ["50%", "55%"],
                        "label": {"color": TRADER["text"], "formatter": "{b}\n{d}%"},
                        "data": [{"name": k, "value": round(v, 1)} for k, v in data.items()]}]}


def heatmap_option(hm: pd.DataFrame, title="Monthly returns (%)") -> dict:
    years = sorted(hm["y"].unique())
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    data = [[int(r.m) - 1, years.index(r.y), round(r.r * 100, 1)] for r in hm.itertuples()]
    mx = max(abs(x[2]) for x in data) if data else 10
    return {"backgroundColor": TRADER["bg"], "title": {"text": title, "textStyle": {"color": TRADER["amber"], "fontSize": 13}},
            "tooltip": {"formatter": "{c}"}, "grid": {"left": 55, "right": 20, "top": 40, "bottom": 50},
            "xAxis": {"type": "category", "data": months, "axisLabel": {"color": TRADER["muted"]}},
            "yAxis": {"type": "category", "data": [str(y) for y in years], "axisLabel": {"color": TRADER["muted"]}},
            "visualMap": {"min": -mx, "max": mx, "calculable": True, "orient": "horizontal", "left": "center", "bottom": 0,
                          "textStyle": {"color": TRADER["text"]},
                          "inRange": {"color": [TRADER["red"], "#2a1a1a", "#1a2a1a", TRADER["green"]]}},
            "series": [{"type": "heatmap", "data": data, "label": {"show": True, "color": "#fff", "fontSize": 10}}]}


def tree_option(facts=FACTS) -> dict:
    sg = facts["segments"]
    q26, q25, m26, m25 = sg["q3_26"], sg["q3_25"], sg["m9_26"], sg["m9_25"]
    tot26, tot25 = sum(q26.values()), sum(q25.values())
    t9_26, t9_25 = sum(m26.values()), sum(m25.values())

    leg, sti = "Array Legacy (U.S.)", "STI Operations (Intl.)"
    g = lambda a, b: f"{(a / b - 1):+.0%}"

    def N(id_, x, y, w, h, col, txt, fs=11):
        return {"name": id_, "value": [x, y], "symbol": "roundRect", "symbolSize": [w, h],
                "itemStyle": {"color": col, "borderColor": "#fff", "borderWidth": 1},
                "label": {"show": True, "position": "inside", "formatter": txt, "color": "#fff", "fontSize": fs,
                          "lineHeight": 15, "fontWeight": 600}}
    nodes = [
        N("root", 50, 10, 300, 74, "#b37a00", f"TOTAL REVENUE 3Q26\n${tot26:.1f}M  ({g(tot26, tot25)} YoY)\n9M-26: ${t9_26:.1f}M ({g(t9_26, t9_25)})", 12),
        N("leg", 20, 45, 230, 76, "#1f78a8", f"{leg}\n${q26[leg]:.1f}M ({g(q26[leg], q25[leg])} YoY)\n9M {g(m26[leg], m25[leg])} · {q26[leg] / tot26:.0%} of mix"),
        N("sti", 62, 45, 230, 76, "#7a58c2", f"{sti}\n${q26[sti]:.1f}M ({g(q26[sti], q25[sti])} YoY)\n9M {g(m26[sti], m25[sti])} · {q26[sti] / tot26:.0%} of mix"),
        N("rec", 90, 45, 200, 76, "#12804b", f"Recognition timing\nOver time ${sg['over_time_q3_26']}M (76%)\nPoint-in-time ${sg['point_in_time_q3_26']}M (24%)"),
        N("l1", 8, 82, 150, 78, "#2a6f97", "Volume −25%\nU.S. deliveries slip\n(permits, rates, tariffs)", 10),
        N("l2", 24, 82, 150, 78, "#2a6f97", "ASP +8%\nD2S, OmniTrack,\nAPA foundations", 10),
        N("l3", 40, 82, 150, 78, "#2a6f97", "AWM (closed Aug 31)\n1 month included\n→ BESS & data centers", 10),
        N("s1", 55, 82, 150, 78, "#5d449a", "Volume +71%\nSpain · Brazil ·\nAustralia", 10),
        N("s2", 70, 82, 150, 78, "#5d449a", "ASP −50%, freight ↑\ngross margin −2.7%", 10),
        N("r1", 90, 82, 170, 78, "#0e6a3e", "Backlog $2.6B (+42%)\nBook-to-bill 1.6x\n~80% delivers in 6 qtrs", 10)]
    links = [{"source": a, "target": b} for a, b in
             (("root", "leg"), ("root", "sti"), ("root", "rec"), ("leg", "l1"), ("leg", "l2"), ("leg", "l3"),
              ("sti", "s1"), ("sti", "s2"), ("rec", "r1"))]
    return {"backgroundColor": TRADER["bg"], "grid": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "xAxis": {"type": "value", "min": 0, "max": 100, "show": False},
            "yAxis": {"type": "value", "min": 0, "max": 100, "inverse": True, "show": False},
            "tooltip": {"show": False},
            "series": [{"type": "graph", "coordinateSystem": "cartesian2d", "data": nodes, "links": links,
                        "lineStyle": {"color": TRADER["muted"], "width": 1.6, "curveness": 0},
                        "edgeSymbol": ["none", "arrow"], "edgeSymbolSize": 7, "silent": True}]}


def payoff_option(idea: dict, S: float) -> dict:
    pl = [round(v * 100, 1) for v in idea["pl"]]
    return {"backgroundColor": TRADER["bg"], "title": {"text": f"{idea['name']} — P/L at expiry (per contract, $)",
            "textStyle": {"color": TRADER["amber"], "fontSize": 13}},
            "tooltip": {"trigger": "axis"}, "grid": {"left": 60, "right": 20, "top": 45, "bottom": 35},
            "xAxis": {"type": "category", "data": [f"{x:.2f}" for x in idea["x"]], "axisLabel": {"color": TRADER["muted"]}},
            "yAxis": {"type": "value", "axisLabel": {"color": TRADER["muted"]}, "splitLine": {"lineStyle": {"color": TRADER["grid"]}}},
            "series": [{"name": "Profit", "type": "line", "data": [max(v, 0) for v in pl], "showSymbol": False,
                        "lineStyle": {"color": TRADER["green"], "width": 2}, "areaStyle": {"color": TRADER["green"], "opacity": 0.25}},
                       {"name": "Loss", "type": "line", "data": [min(v, 0) for v in pl], "showSymbol": False,
                        "lineStyle": {"color": TRADER["red"], "width": 2}, "areaStyle": {"color": TRADER["red"], "opacity": 0.25}},
                       {"name": "Spot", "type": "scatter", "symbol": "diamond", "symbolSize": 14,
                        "itemStyle": {"color": TRADER["amber"], "borderColor": "#fff"},
                        "label": {"show": True, "formatter": f"Spot ${S:.2f}", "color": TRADER["amber"], "position": "top"},
                        "data": [[f"{idea['x'][k]:.2f}", pl[k]] for k in [int(np.argmin(np.abs(np.array(idea['x']) - S)))]]}]}


def gauge_option(value: float, title: str) -> dict:
    return {"backgroundColor": TRADER["bg"],
            "series": [{"type": "gauge", "min": -100, "max": 100, "splitNumber": 4, "radius": "92%",
                        "axisLine": {"lineStyle": {"width": 12, "color": [[0.375, TRADER["red"]], [0.45, "#ff9f43"],
                                                                         [0.55, TRADER["muted"]], [0.625, "#9be564"],
                                                                         [1, TRADER["green"]]]}},
                        "pointer": {"width": 4}, "axisLabel": {"color": TRADER["muted"], "fontSize": 9},
                        "detail": {"formatter": "{value}", "color": TRADER["amber"], "fontSize": 20, "offsetCenter": [0, "70%"]},
                        "title": {"color": TRADER["text"], "fontSize": 12, "offsetCenter": [0, "92%"]},
                        "data": [{"value": round(value, 0), "name": title}]}]}


def line_compare_option(df: pd.DataFrame, title="Rebased performance (start = 100)") -> dict:
    ser = [{"name": c, "type": "line", "showSymbol": False, "data": [_f(v, 2) for v in df[c]]} for c in df.columns]
    return {"backgroundColor": TRADER["bg"], "title": {"text": title, "textStyle": {"color": TRADER["amber"], "fontSize": 13}},
            "tooltip": {"trigger": "axis"}, "legend": {"top": 0, "right": 10, "textStyle": {"color": TRADER["text"]}, "type": "scroll"},
            "color": [TRADER["amber"], TRADER["cyan"], TRADER["green"], TRADER["red"], TRADER["violet"], "#ff7ac6", "#7fffd4", "#9aa5b1"],
            "grid": {"left": 55, "right": 20, "top": 50, "bottom": 50},
            "xAxis": {"type": "category", "data": [x.strftime("%Y-%m-%d") for x in df.index], "axisLabel": {"color": TRADER["muted"]}},
            "yAxis": {"type": "value", "scale": True, "axisLabel": {"color": TRADER["muted"]}, "splitLine": {"lineStyle": {"color": TRADER["grid"]}}},
            "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 16, "bottom": 6}], "series": ser}


# ----------------------------------------------------------------------------------------------
# 10. STREAMLIT UI
# ----------------------------------------------------------------------------------------------
CSS = f"""
<style>
.stApp {{background:{TRADER['bg']}; color:{TRADER['text']}; font-family:'Consolas','Menlo',monospace;}}
section[data-testid="stSidebar"] {{background:{TRADER['panel']}; border-right:1px solid {TRADER['grid']};}}
h1,h2,h3,h4 {{color:{TRADER['amber']} !important; letter-spacing:.5px; font-family:'Consolas','Menlo',monospace;}}
div[data-testid="stMetric"] {{background:{TRADER['panel']}; border:1px solid {TRADER['grid']}; border-left:3px solid {TRADER['amber']};
  padding:8px 12px; border-radius:4px;}}
div[data-testid="stMetricValue"] {{color:{TRADER['text']}; font-size:1.35rem;}}
div[data-testid="stMetricLabel"] p {{color:{TRADER['muted']}; text-transform:uppercase; font-size:.7rem;}}
button[data-baseweb="tab"] {{color:{TRADER['muted']}; font-weight:700;}}
button[data-baseweb="tab"][aria-selected="true"] {{color:{TRADER['amber']};}}
.card {{background:{TRADER['panel']}; border:1px solid {TRADER['grid']}; border-radius:4px; padding:12px 14px; margin-bottom:10px;}}
.pill {{padding:2px 8px; border-radius:3px; font-weight:700; font-size:.75rem;}}
.up {{color:{TRADER['green']};}} .dn {{color:{TRADER['red']};}} .mu {{color:{TRADER['muted']};}}
.ticker-tape {{background:#000; border-top:1px solid {TRADER['amber']}; border-bottom:1px solid {TRADER['amber']};
  padding:4px 10px; color:{TRADER['amber']}; white-space:nowrap; overflow:hidden; font-size:.8rem;}}
</style>
"""


def fmt_big(x):
    try:
        x = float(x)
    except Exception:
        return "–"
    for u, v in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= v:
            return f"{x / v:.2f}{u}"
    return f"{x:.2f}"


def main():  # pragma: no cover - UI
    import streamlit as st
    from streamlit_echarts import st_echarts

    st.set_page_config(page_title="ARRY Trading Desk", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    cache = lambda fn: st.cache_data(ttl=3600, show_spinner=False)(fn)
    c_hist, c_info, c_stm, c_fv, c_news = cache(get_history), cache(get_info), cache(get_statements), cache(get_finviz), cache(get_news)
    c_opt = st.cache_data(ttl=900, show_spinner=False)(get_options)
    c_fc = cache(lambda close_json, horizon, sims, shrink: forecast_models(pd.read_json(close_json, typ="series"), horizon, sims, shrink))

    # ------------------------------------------------ sidebar control panel
    sb = st.sidebar
    sb.markdown("### ⚙ CONTROL PANEL")
    page = sb.radio("Page", ["Dashboard", "Executive Summary", "Recommendations & Insights"], index=0)
    sb.markdown("---")
    ticker = sb.text_input("Ticker (any Yahoo symbol)", "ARRY").strip().upper()
    bm_label = sb.selectbox("Compare against index", list(BENCHMARKS) + ["Custom…"], index=0)
    bench = BENCHMARKS.get(bm_label) or sb.text_input("Custom benchmark", "^GSPC").upper()
    peers_in = sb.text_input("Compare tickers (comma-separated)", "NXT, SHLS, FSLR, ENPH")
    years = sb.slider("History (years)", 1, 5, 3)
    with sb.expander("📊 Technical studies", expanded=True):
        emas = sb.multiselect("EMAs", [7, 14, 28, 50, 200], [7, 14, 28, 50, 200])
        show_bb = sb.checkbox("Bollinger Bands", True)
        bb_n = sb.number_input("BB period", 5, 60, 20)
        bb_k = sb.number_input("BB std-dev", 1.0, 4.0, 2.0, 0.5)
        show_vol = sb.checkbox("Volume", True)
        show_macd = sb.checkbox("MACD", False)
        rsi_n = sb.number_input("RSI period (default 10)", 2, 50, 10)
        rsi_ob, rsi_os = sb.slider("RSI oversold / overbought", 10, 90, (30, 70))
    with sb.expander("🕯 Candlestick signals", expanded=False):
        min_conf = sb.slider("Min. confirmations (of 3)", 0, 3, 2)
        vol_mult = sb.slider("Volume multiple vs 20d avg", 0.8, 3.0, 1.2, 0.1)
        req_next = sb.checkbox("Require next-bar confirmation", True)
        show_filtered = sb.checkbox("Show filtered (misleading) signals", False)
        hold = sb.slider("Backtest holding period (bars)", 3, 30, 10)
    with sb.expander("🔮 Forecast", expanded=False):
        fc_h = sb.radio("Chart horizon", ["3 months", "6 months"], index=1, horizontal=True)
        shrink = sb.slider("Drift shrinkage (0 = no drift)", 0.0, 1.0, 0.5, 0.1)
        sims = sb.select_slider("Monte-Carlo paths", [500, 1000, 2000, 4000], 2000)
    with sb.expander("🎯 Options", expanded=False):
        cc_delta = sb.slider("Covered-call target delta", 0.10, 0.50, 0.30, 0.05)
        csp_delta = sb.slider("Cash-secured-put target |delta|", 0.10, 0.50, 0.25, 0.05)
        rf = sb.number_input("Risk-free rate", 0.0, 0.10, 0.042, 0.001, format="%.3f")
    sb.caption("Collapse this panel with the « arrow (top-right of the sidebar).")

    # ------------------------------------------------ data
    try:
        with st.spinner(f"Loading {ticker}…"):
            px = c_hist(ticker, years)
    except Exception as e:
        st.error(f"Could not load {ticker}: {e}")
        st.stop()
    info, stm, fv = c_info(ticker), c_stm(ticker), c_fv(ticker)
    news = score_sentiment(filter_relevant(fv.get("news", []) + c_news(ticker, info.get("shortName", "")),
                                           ticker, info.get("shortName", "")))
    d = add_indicators(px, rsi_n, bb_n, bb_k)
    d["EMA28"] = ema(d["Close"], 28)
    sig = build_signals(d, vol_mult, min_conf, req_next)
    S = float(d["Close"].iloc[-1])
    fc = c_fc(px["Close"].to_json(), 252, sims, shrink)
    hv = realized_vol(px["Close"])
    hv90 = realized_vol(px["Close"], 90)
    ins = horizon_insights(d, sig, fc, info, fv, news, hv)
    snap = fv.get("snapshot", {})

    st.markdown(f"<div class='ticker-tape'>● {ticker} {S:,.2f} &nbsp;|&nbsp; MKT CAP {fmt_big(info.get('marketCap'))} &nbsp;|&nbsp; "
                f"FWD P/E {info.get('forwardPE') or snap.get('Forward P/E', '–')} &nbsp;|&nbsp; SHORT FLOAT {snap.get('Short Float', '–')} "
                f"&nbsp;|&nbsp; TARGET {info.get('targetMeanPrice', snap.get('Target Price', '–'))} &nbsp;|&nbsp; "
                f"{info.get('sector', fv.get('sector', ''))} / {info.get('industry', '')}</div>", unsafe_allow_html=True)
    st.title(f"{info.get('shortName', ticker)} ({ticker}) — Trading Desk")

    def ec(opt, h=420, key=None):
        st_echarts(options=opt, height=f"{h}px", key=key)

    # ------------------------------------------------ pages
    if page == "Executive Summary":
        chg = d["Close"].pct_change().iloc[-1]
        c = st.columns(6)
        c[0].metric("Last", f"${S:,.2f}", f"{chg:+.1%}")
        c[1].metric("Market cap", fmt_big(info.get("marketCap")))
        c[2].metric("Fwd P/E", f"{info.get('forwardPE', float('nan')):.2f}x" if info.get("forwardPE") else snap.get("Forward P/E", "–"))
        c[3].metric("P/S", f"{info.get('priceToSalesTrailing12Months', float('nan')):.2f}x" if info.get("priceToSalesTrailing12Months") else snap.get("P/S", "–"))
        c[4].metric(f"RSI({rsi_n})", f"{d['RSI'].iloc[-1]:.0f}")
        c[5].metric("HV30 (ann.)", f"{hv:.0%}")
        a, b = st.columns([1, 2])
        with a:
            ec(gauge_option(ins["horizons"]["3 months"]["score"], "3-month composite"), 300)
        with b:
            st.subheader("Executive summary")
            hz = ins["horizons"]
            st.markdown(f"""
- **Thesis:** record **$2.6B orderbook (+42% YoY, 1.6x book-to-bill)** vs. margin compression (3Q26 adj. gross margin 22.8%, adj. EBITDA margin 9.7%) and a **FY26 guidance cut**; GAAP net loss to common of −$7.9M after $16.1M preferred dividends.
- **Catalyst/Risk:** stock-for-stock merger with **Shoals (SHLS)** ($1.6B tax shield + $237M NOLs; ~65% dilution to ARRY holders); net debt/EBITDA **3.9x**, ~$466M liquidity.
- **Technical:** {', '.join(ins['tech_notes'][:3])}.
- **Forecast (ARIMA + ARIMA-GARCH):** 3M median ${hz['3 months']['median']:.2f} ({hz['3 months']['ret_median']:+.0%}, P(up) {hz['3 months']['p_up']:.0%}); 6M median ${hz['6 months']['median']:.2f} (5–95%: ${hz['6 months']['p5']:.2f}–${hz['6 months']['p95']:.2f}).
- **Preliminary view:** 1W **{hz['1 week']['rating']}**, 3M **{hz['3 months']['rating']}**, 6M **{hz['6 months']['rating']}**, 12M **{hz['12 months']['rating']}** — speculative turnaround/value; size accordingly.""")
        ec(tech_option(d.iloc[-250:], sig[sig["date"] >= d.index[-250]] if len(sig) else sig, emas, show_bb, show_vol, False, rsi_n, rsi_ob, rsi_os), 560)

    elif page == "Recommendations & Insights":
        st.subheader("Recommendations & insights by horizon")
        st.caption("Composite = weighted blend of technical, forecast, fundamental/valuation and news-sentiment scores (−100…+100). Heuristic, not advice.")
        cols = st.columns(4)
        for col, (h, v) in zip(cols, ins["horizons"].items()):
            colr = TRADER["green"] if v["score"] > 10 else TRADER["red"] if v["score"] < -10 else TRADER["muted"]
            col.markdown(f"<div class='card'><div class='mu'>{h.upper()}</div><h2 style='margin:0;color:{colr}!important'>{v['rating']}</h2>"
                         f"<div>Score <b>{v['score']:+.0f}</b></div><div class='mu'>Median ${v['median']:.2f} ({v['ret_median']:+.0%})<br>"
                         f"5–95%: ${v['p5']:.2f} – ${v['p95']:.2f}<br>P(up) {v['p_up']:.0%} · P(+20%) {v['p_up20']:.0%} · P(−20%) {v['p_dn20']:.0%}<br>"
                         f"±1σ move ≈ ${v['exp_move_1sd']:.2f}</div></div>", unsafe_allow_html=True)
        st.markdown("#### Why")
        c1, c2 = st.columns(2)
        c1.markdown("**Technical drivers**\n\n" + "\n".join(f"- {n}" for n in ins["tech_notes"]))
        c2.markdown("**Fundamental / valuation drivers**\n\n" + "\n".join(f"- {n}" for n in ins["fund_notes"]))
        opts = c_opt(ticker)
        st.markdown("#### Options playbook by horizon")
        iv_hv_default = 1.0
        for h, tgt in (("1 week", 7), ("3 months", 91), ("6 months", 182), ("12 months", 365)):
            v = ins["horizons"][h]
            if not opts["chains"]:
                st.info("No listed options for this ticker."); break
            e = min(opts["chains"], key=lambda x: abs(opts["chains"][x]["dte"] - tgt))
            ch = opts["chains"][e]
            aiv = atm_iv({e: ch}, S)
            ivhv = aiv / hv90 if hv90 else iv_hv_default
            ideas = option_ideas(ch, S, rf, v["bias"], ivhv, float(d["RSI"].iloc[-1]), cc_delta, csp_delta)
            pref = [i for i in ideas if i["fit"] == "Preferred"] or [i for i in ideas if i["fit"] == "Neutral"][:2]
            with st.expander(f"{h} → expiry {e} ({ch['dte']} DTE) · ATM IV {aiv:.0%} vs HV90 {hv90:.0%} (IV/HV {ivhv:.2f}; HV30 {hv:.0%}) · {v['rating']}", expanded=(h == '3 months')):
                if not pref:
                    st.write("No liquid structure found.")
                for i in pref:
                    st.markdown(f"**{i['name']}** — {i['legs']}  \nCredit/debit ${i['credit']:+.2f} · max profit ${i['max_profit']*100:,.0f} · "
                                f"max loss ${i['max_loss']*100:,.0f} · break-even ${i['breakeven']:.2f} · POP ≈ {i['pop']:.0%} · "
                                f"liquidity: {i['liq']}  \n*{i['why']}*")

    else:
        tabs = st.tabs(["📈 Technical", "🔮 Forecast", "📊 Fundamentals & KPIs", "🧾 Financial statements", "🎯 Options",
                        "🆚 Compare & peers", "📰 News", "📑 Report analysis (Q3-26)", "🌳 Revenue tree"])
        # ---- technical
        with tabs[0]:
            ec(tech_option(d, sig, emas, show_bb, show_vol, show_macd, rsi_n, rsi_ob, rsi_os, show_filtered, f"{ticker} daily"), 760, "tech")
            c1, c2 = st.columns([1, 1])
            n_conf = int((sig["state"] == "confirmed").sum()) if len(sig) else 0
            n_filt = int((sig["state"] == "filtered").sum()) if len(sig) else 0
            c1.metric("Confirmed signals", n_conf, help="Pattern + ≥N confirmations + next-bar follow-through")
            c2.metric("Filtered out (misleading)", n_filt)
            st.markdown("**Signal log** (▲ BUY / ▼ SELL confirmed; ◆ pending next bar)")
            if len(sig):
                show = sig[sig["state"] != "filtered"].tail(25).copy() if not show_filtered else sig.tail(40).copy()
                show["date"] = show["date"].dt.strftime("%Y-%m-%d")
                st.dataframe(show[["date", "side", "pattern", "state", "score", "trend", "volume", "location", "next_ok", "price"]],
                             width="stretch", hide_index=True)
                bt = signal_backtest(d, sig, hold)
                st.markdown(f"**Does filtering help?** Forward {hold}-bar side-adjusted return")
                st.dataframe(bt.style.format({"Win rate": "{:.0%}", "Avg return": "{:+.1%}", "Median": "{:+.1%}"}),
                             width="stretch", hide_index=True)
            hm = monthly_returns_heatmap(px["Close"])
            ec(heatmap_option(hm, f"{ticker} seasonality — monthly returns (%)"), 300, "hm")
        # ---- forecast
        with tabs[1]:
            hzn = 126 if fc_h == "6 months" else 63
            st.caption(f"Chosen model: **{fc['arima']['order']}** on log-price (AIC grid p,q ≤ 3; ADF p-value price {fc['adf_price_p']:.2f} → non-stationary, d=1; "
                       f"returns {fc['adf_ret_p']:.3f} → stationary) + **{fc['garch']['order']}** for fat-tailed/volatility-clustered paths. "
                       f"Horizon 3–6 months: ARIMA is mean-reverting in returns so it's close to a drift random walk — the information is in the *bands*, not the line.")
            ec(forecast_option(px["Close"], fc, hzn), 480, "fc")
            rows = []
            for h, pr in fc["prob"].items():
                rows.append({"Horizon": h, "Median": pr["median"], "5th pct": pr["p5"], "95th pct": pr["p95"],
                             "P(up)": pr["p_up"], "P(+20%)": pr["p_up20"], "P(−20%)": pr["p_dn20"]})
            st.dataframe(pd.DataFrame(rows).style.format({"Median": "${:.2f}", "5th pct": "${:.2f}", "95th pct": "${:.2f}",
                                                          "P(up)": "{:.0%}", "P(+20%)": "{:.0%}", "P(−20%)": "{:.0%}"}),
                         width="stretch", hide_index=True)
            wf = walk_forward(px["Close"], tuple(int(x) for x in re.findall(r"\d", fc["arima"]["order"])[::2]))
            if wf:
                st.markdown(f"**Walk-forward back-test (last year, {wf['n']} origins, {wf['hz']}-day horizon):** model MAPE {wf['mape_model']:.1%} vs naive {wf['mape_naive']:.1%} · "
                            f"directional accuracy {wf['dir_acc']:.0%} · 90% band coverage {wf['coverage90']:.0%}")
            st.warning("12-month figures are shown for completeness; uncertainty is very wide and event risk (SHLS merger, tax-credit policy) is not in the model.")
        # ---- fundamentals
        with tabs[2]:
            q = quarterly_series(stm, info.get("sharesOutstanding"))
            k = st.columns(6)
            k[0].metric("Orderbook", "$2.6B", "+42% YoY")
            k[1].metric("Book-to-bill (TTM)", "1.6x")
            k[2].metric("Adj. EBITDA 3Q26", "$32.1M", "−56% YoY")
            k[3].metric("Net debt / EBITDA", "3.9x")
            k[4].metric("Liquidity", "$465.9M")
            k[5].metric("FCF 9M-26", "$75.6M", "+73% YoY")
            c1, c2 = st.columns(2)
            with c1:
                ec(bar_option(FACTS["orderbook"]["labels"], {"Orderbook ($B)": FACTS["orderbook"]["usd_b"]}, "Orderbook ($B)"), 300, "ob")
            with c2:
                fq = FACTS["q"]
                ec(bar_option(fq["labels"], {"Revenue ($M)": fq["revenue"], "Adj. EBITDA ($M)": fq["adj_ebitda"]}, "Revenue vs Adj. EBITDA ($M)"), 300, "re")
            c1, c2 = st.columns(2)
            with c1:
                ec(pie_option(FACTS["segments"]["q3_26"], "3Q26 revenue by segment ($M)"), 300, "pie")
            with c2:
                if not q.empty and "FCF_ps" in q:
                    ec(bar_option([x.strftime("%b-%y") for x in q.index], {"OCF / share": q["OCF_ps"].round(3).tolist(),
                                                                         "FCF / share": q["FCF_ps"].round(3).tolist()}, "Cash generation per share ($)"), 300, "cps")
            vh = valuation_history(px["Close"], q, info.get("sharesOutstanding"))
            if not vh.empty:
                c1, c2 = st.columns(2)
                with c1:
                    ec(line_compare_option(vh[["PS"]].dropna().rename(columns={"PS": "P/S (TTM)"}), "P/S ratio history"), 300, "ps")
                with c2:
                    if vh["PE"].notna().sum() > 5:
                        ec(line_compare_option(vh[["PE"]].dropna().rename(columns={"PE": "P/E (TTM, EPS>0)"}), "P/E ratio history"), 300, "pe")
                    else:
                        st.info("GAAP TTM EPS is negative → trailing P/E undefined. Use forward P/E "
                                f"({snap.get('Forward P/E', info.get('forwardPE', '–'))}x) and guidance-implied adj. P/E = price / $0.63 = {S / 0.63:.1f}x.")
            st.markdown("#### Valuation multiples & PE growth")
            g = FACTS["guidance"]
            vals = {"Price": f"${S:.2f}", "Trailing P/E": snap.get("P/E", "–"), "Forward P/E": snap.get("Forward P/E", info.get("forwardPE", "–")),
                    "PEG": snap.get("PEG", "–"), "P/S": snap.get("P/S", "–"), "P/FCF": snap.get("P/FCF", "–"), "P/C": snap.get("P/C", "–"),
                    "EV/Sales": snap.get("EV/Sales", "–"), "EV/EBITDA (Finviz)": snap.get("EV/EBITDA", "–"),
                    "Adj. P/E on FY26 guide mid ($0.63)": f"{S / 0.63:.1f}x", "EPS next Y": snap.get("EPS next Y", "–"),
                    "EPS next 5Y": snap.get("EPS next 5Y", "–")}
            st.dataframe(pd.DataFrame(vals.items(), columns=["Metric", "Value"]), width="stretch", hide_index=True)
            ec(bar_option(g["labels"][:2], {"Prev. guide mid": [(g["prev_lo"][i] + g["prev_hi"][i]) / 2 for i in range(2)],
                                           "New guide mid": [(g["new_lo"][i] + g["new_hi"][i]) / 2 for i in range(2)]}, "FY26 guidance revision ($M, midpoint)"), 300, "gd")
            if fv.get("profile"):
                st.markdown(f"**Profile:** {fv['profile']}")
        # ---- statements
        with tabs[3]:
            st.markdown("#### Reported — Q3-2026 press release ($ thousands)")
            a, b, c3 = st.tabs(["Income statement", "Balance sheet", "Cash flow / capex"])
            a.dataframe(pd.DataFrame(REPORTED["income"], index=["3Q26", "3Q25", "9M-26", "9M-25"]).T.style.format("{:,.0f}"), width="stretch")
            b.dataframe(pd.DataFrame(REPORTED["balance"], index=["Sep-30-26", "Dec-31-25"]).T.style.format("{:,.0f}"), width="stretch")
            c3.dataframe(pd.DataFrame(REPORTED["cashflow"], index=["3Q26", "3Q25", "9M-26", "9M-25"]).T.style.format("{:,.0f}"), width="stretch")
            st.markdown("#### Financial ratios (computed from reported Q3-26)")
            st.dataframe(reported_ratios(), width="stretch", hide_index=True)
            st.markdown("#### Yahoo Finance — quarterly statements")
            for k_, t_ in (("income", "Income statement"), ("balance", "Balance sheet"), ("cashflow", "Cash flow")):
                with st.expander(t_):
                    df_ = stm.get(k_, pd.DataFrame())
                    st.dataframe(df_.dropna(how="all") if not df_.empty else df_, width="stretch")
        # ---- options
        with tabs[4]:
            opts = c_opt(ticker)
            if not opts["chains"]:
                st.info("No listed options for this ticker.")
            else:
                hv_p = hv_percentile(px["Close"])
                st.caption(f"HV30 {hv:.0%} (percentile vs {years}y: {hv_p:.0%}) · HV90 {hv90:.0%} · IV = ATM implied vol from chain · IV/HV90 > 1.05 → premium selling favoured")
                sel = st.selectbox("Expiry", list(opts["chains"]), index=min(1, len(opts["chains"]) - 1),
                                   format_func=lambda e: f"{e} ({opts['chains'][e]['dte']} DTE)")
                ch = opts["chains"][sel]
                aiv = atm_iv({sel: ch}, S)
                bias = ins["horizons"]["3 months"]["bias"]
                ideas = option_ideas(ch, S, rf, bias, aiv / hv90 if hv90 else 1.0, float(d["RSI"].iloc[-1]), cc_delta, csp_delta)
                m = st.columns(4)
                m[0].metric("ATM IV", f"{aiv:.0%}"); m[1].metric("HV30 / HV90", f"{hv:.0%} / {hv90:.0%}"); m[2].metric("IV / HV90", f"{aiv / hv90:.2f}")
                m[3].metric("Bias (3M)", f"{bias:+.2f}")
                if ideas:
                    tbl = pd.DataFrame([{"Strategy": i["name"], "Structure": i["legs"], "Fit": i["fit"], "Net $/sh": i["credit"],
                                         "Max profit $": i["max_profit"] * 100, "Max loss $": i["max_loss"] * 100,
                                         "Break-even": i["breakeven"], "Static ret": i["ret_static"], "Ann. ret": i["ret_ann"],
                                         "POP": i["pop"], "Liquidity": i["liq"], "Capital $": i["capital"]} for i in ideas])
                    st.dataframe(tbl.style.format({"Net $/sh": "{:+.2f}", "Max profit $": "{:,.0f}", "Max loss $": "{:,.0f}", "Break-even": "{:.2f}",
                                                   "Static ret": "{:.1%}", "Ann. ret": "{:.0%}", "POP": "{:.0%}", "Capital $": "{:,.0f}"}),
                                 width="stretch", hide_index=True)
                    pick = st.selectbox("Payoff diagram", [i["name"] for i in ideas])
                    idea = next(i for i in ideas if i["name"] == pick)
                    ec(payoff_option(idea, S), 340, "payoff")
                    st.markdown(f"*{idea['why']}*")
                st.markdown("**Other ideas:** Wheel (CSP → assignment → covered calls); calendar/diagonal around the SHLS merger vote; "
                            "protective put before binary events. Wide bid/ask on this low-priced name — use limit orders at mid.")
                with st.expander("Raw chain"):
                    st.dataframe(ch["calls"][["strike", "bid", "ask", "lastPrice", "impliedVolatility", "openInterest", "volume"]], width="stretch")
                    st.dataframe(ch["puts"][["strike", "bid", "ask", "lastPrice", "impliedVolatility", "openInterest", "volume"]], width="stretch")
        # ---- compare
        with tabs[5]:
            tk = [t.strip().upper() for t in peers_in.split(",") if t.strip()]
            names = [ticker, bench] + [t for t in tk if t != ticker]
            closes = {}
            for t in names:
                try:
                    closes[t] = c_hist(t, years)["Close"]
                except Exception:
                    st.warning(f"No data for {t}")
            if len(closes) > 1:
                df = pd.DataFrame(closes).dropna(how="all").ffill().dropna()
                ec(line_compare_option(df / df.iloc[0] * 100), 420, "cmp")
                rets = df.pct_change().dropna()
                if bench in rets:
                    beta = rets.apply(lambda s_: np.cov(s_, rets[bench])[0, 1] / rets[bench].var())
                    corr = rets.corrwith(rets[bench])
                    st.dataframe(pd.DataFrame({"Total return": df.iloc[-1] / df.iloc[0] - 1, "Ann. vol": rets.std() * math.sqrt(252),
                                               f"Beta vs {bench}": beta, f"Corr vs {bench}": corr}).style.format(
                        {"Total return": "{:+.0%}", "Ann. vol": "{:.0%}", f"Beta vs {bench}": "{:.2f}", f"Corr vs {bench}": "{:.2f}"}),
                        width="stretch")
                    roll = (rets[ticker].rolling(60).corr(rets[bench])).dropna().to_frame(f"60d corr {ticker}/{bench}")
                    ec(line_compare_option(roll, "Rolling 60-day correlation"), 260, "roll")
            st.markdown("#### Peer valuation (Finviz)")
            peers = fv.get("peers") or PEERS_DEFAULT
            tbl = comparison_table([ticker] + [p for p in dict.fromkeys(tk + peers) if p != ticker][:8], c_fv)
            st.dataframe(tbl, width="stretch", hide_index=True)
        # ---- news
        with tabs[6]:
            allnews = news[:40]
            avg = np.mean([n.get("sent", 0) for n in news]) if news else 0
            st.metric("Headline sentiment (VADER, −1…+1)", f"{avg:+.2f}")
            for n in allnews:
                col = TRADER["green"] if n.get("sent", 0) > 0.15 else TRADER["red"] if n.get("sent", 0) < -0.15 else TRADER["muted"]
                st.markdown(f"<div class='card'><span style='color:{col}'>●</span> <a href='{n['link']}' target='_blank'>{n['title']}</a>"
                            f"<div class='mu'>{n.get('source', '')} · {n.get('when', '')}</div></div>", unsafe_allow_html=True)
            if fv.get("ratings"):
                st.markdown("#### Analyst actions (Finviz)")
                st.dataframe(pd.DataFrame(fv["ratings"]), width="stretch", hide_index=True)
        # ---- report analysis
        with tabs[7]:
            st.markdown(REPORT_ANALYSIS_MD)
        # ---- revenue tree
        with tabs[8]:
            ec(tree_option(), 520, "tree")
            st.markdown(REVENUE_TREE_SUMMARY)

    st.caption("Educational tool — not investment advice. Data: Yahoo Finance, Finviz, Google News, company filings (Q3-2026).")


# ----------------------------------------------------------------------------------------------
# 11. NARRATIVE (follows the two course prompts)
# ----------------------------------------------------------------------------------------------
def reported_ratios() -> pd.DataFrame:
    b, i, cf = REPORTED["balance"], REPORTED["income"], REPORTED["cashflow"]
    ca, cl, inv = b["Total current assets"][0], b["Total current liabilities"][0], b["Inventories"][0]
    cash, ar = b["Cash & equivalents"][0], b["Accounts receivable"][0]
    ebit9, int9 = i["Income from operations"][2], -i["Interest expense"][2]
    ni9, assets = i["Net income"][2], b["Total assets"][0]
    ic = 661794 + b["Series A redeemable preferred"][0] + b["Total stockholders' equity"][0]
    nopat = ebit9 * 4 / 3 * (1 - 0.25)
    rows = [
        ("Liquidity", "Current ratio", ca / cl, "x"), ("Liquidity", "Quick ratio (CA − inventory)/CL", (ca - inv) / cl, "x"),
        ("Liquidity", "Cash ratio", cash / cl, "x"), ("Leverage", "Total debt / TTM adj. EBITDA", 673.2 / 135.0, "x"),
        ("Leverage", "Net debt / TTM adj. EBITDA", 530.0 / 135.0, "x"),
        ("Leverage", "Interest coverage (9M EBIT / interest)", ebit9 / int9, "x"),
        ("Leverage", "Interest coverage (9M adj. EBITDA / interest)", i["Adj. EBITDA"][2] / int9, "x"),
        ("Leverage", "Liabilities / assets", b["Total liabilities"][0] / assets, "%"),
        ("Leverage", "Debt / (stockholders' equity)", "n.m. (equity < 0)", ""),
        ("Profitability", "Gross margin 3Q26", i["Gross profit"][0] / i["Revenue"][0], "%"),
        ("Profitability", "Adj. gross margin 3Q26", i["Adj. gross profit"][0] / i["Revenue"][0], "%"),
        ("Profitability", "Operating margin 3Q26", i["Income from operations"][0] / i["Revenue"][0], "%"),
        ("Profitability", "Adj. EBITDA margin 3Q26", i["Adj. EBITDA"][0] / i["Revenue"][0], "%"),
        ("Profitability", "Net margin to common 3Q26", i["Net income to common"][0] / i["Revenue"][0], "%"),
        ("Returns", "ROA (9M NI annualised / assets)", ni9 * 4 / 3 / assets, "%"),
        ("Returns", "ROE", "n.m. (negative equity)", ""),
        ("Returns", "ROIC (NOPAT 25% tax / debt+pref+equity)", nopat / ic, "%"),
        ("Efficiency", "DSO (AR / 3Q rev × 92)", ar / i["Revenue"][0] * 92, "d"),
        ("Efficiency", "Inventory days (inv / COGS × 92)", inv / i["Cost of revenue"][0] * 92, "d"),
        ("Cash", "FCF margin 9M-26", cf["Free cash flow"][2] / i["Revenue"][2], "%"),
        ("Cash", "Capex / revenue 9M-26", -cf["Capex (PP&E)"][2] / i["Revenue"][2], "%"),
        ("Cash", "Capex 9M-26 vs 9M-25", cf["Capex (PP&E)"][2] / cf["Capex (PP&E)"][3] - 1, "%"),
    ]
    out = []
    for g, n, v, u in rows:
        if isinstance(v, str):
            out.append({"Group": g, "Ratio": n, "Value": v})
        elif u == "%":
            out.append({"Group": g, "Ratio": n, "Value": f"{v:.1%}"})
        elif u == "d":
            out.append({"Group": g, "Ratio": n, "Value": f"{v:.0f} days"})
        else:
            out.append({"Group": g, "Ratio": n, "Value": f"{v:.2f}x"})
    return pd.DataFrame(out)


REVENUE_TREE_SUMMARY = """
**Executive summary (revenue tree)**
- **Main revenue engine:** U.S. utility-scale solar trackers (Array Legacy) — $296.2M, ~90% of 3Q26 revenue; revenue = GW delivered × ASP, recognised mostly over time.
- **Most relevant segment:** Array Legacy; STI (international) is only ~10% and shrinking (−69% 9M) on lower ASP.
- **Key growth driver:** $2.6B orderbook (+42%), ~60% from new products (D2S, OmniTrack, APA foundations, AWM wire management) — 80% delivers within 6 quarters; 4Q26 guided $445–505M.
- **Near-term drag:** project-timing slips (permitting, rates, tariffs) cut volume 25% and compressed margins; FY26 revenue guide cut to $1.34–1.40B.
- **Strategic lever:** SHLS merger + AWM extend content-per-MW (balance-of-system) beyond trackers.
"""

REPORT_ANALYSIS_MD = """
### 1. General information, strategy & macro (Q3-2026 10-Q / press release)
**Opportunities:** record orderbook **$2.6B (+42% YoY; 4th straight record)**, TTM book-to-bill **1.6x**, >$1.8B new orders TTM; platform expansion (APA foundations → Atlas™ I/II; AWM wire management → BESS & data centres); new Albuquerque plant (216k sq ft, 100% domestic content, higher 45X capture); stock-for-stock merger with **Shoals** (tax shield ≈$1.6B amortisation + $237M NOLs; Series A minimum redemption cut $200M→$50M).
**Risks:** policy & financing uncertainty extending project timelines (permitting touches ~30% of U.S. pipeline; Section 232 polysilicon tariffs; FCC inverter ban; higher rates); **45X receivable reserve $10.9M**; customer concentration (top-2 = 26% of 3Q revenue); convertible debt ($325M 2028, $345M 2031) and Series A preferred ($510M liquidation preference); integration risk (APA, AWM, SHLS); short interest of 26–35% of float depending on source (Finviz 26.3%, source documents ~35.5%).
**Strategy shift:** from "tracker company" → "integrated energy-infrastructure provider" (foundation-to-tracker + wire management); international footprint is being de-emphasised (STI −69% 9M).
**Macro factors:** interest rates/tax-equity availability, OBBB-era ITC sunset (construction-start rules), tariffs, Middle-East/Red-Sea logistics, BRL volatility.

### 2. Financial results
| ($M) | 3Q26 | 3Q25 | YoY | 2Q26 |
|---|---|---|---|---|
| Revenue | 330.2 | 393.5 | −16% | 342.1 |
| – Array Legacy (U.S.) | 296.2 | 353.4 | −16% | |
| – STI Operations | 34.0 | 40.1 | −15% | |
| Gross margin (adj.) | 22.8% | 28.1% | −520 bps | 30.8% |
| Adj. EBITDA (margin) | 32.1 (9.7%) | 72.2 (18.3%) | −56% | 63.3 (18.5%) |
| Net income to common | −7.9 | 18.4 | n.m. | 8.4 |
| Adj. EPS (diluted) | $0.06 | $0.30 | −80% | $0.24 |
| FCF | −1.1 | 21.9 | n.m. | 113.6 |

- **Trend:** 9M revenue −15% ($895.7M vs $1,058.1M); volume −25% (U.S.), ASP +8%. YTD adj. gross margin held at **27.8% (+30 bps)** — the 3Q dip is mostly the 45X reserve (−330 bps), input costs and mix. G&A up (adj. G&A 13.9% of 3Q revenue vs 9.9%) — **cost trend to watch**.
- **Returns:** reported ROE n.m. (negative common equity after preferred); ROA ≈2.9% (9M annualised); ROIC ≈6% (approx., NOPAT at 25% tax). Finviz peer set: see *Compare & peers* tab.
- **Leverage:** total debt $673M = **5.0x** TTM adj. EBITDA ($135M); net debt **$530M = 3.9x**; 9M EBIT/interest 3.3x; management expects leverage to rebound after the 4Q26 delivery peak.
- **Liquidity:** cash $143.3M + undrawn RCF → **$465.9M**; current ratio 1.62x, quick 1.25x. Cash fell $101M YTD, mainly the $161M AWM acquisition; operating cash flow 9M +66% to $96.5M.
- **Outlook / CAPEX:** FY26 guide: revenue $1.34–1.40B, adj. EBITDA $190–210M, adj. EPS $0.57–0.69, adj. GM ~27%; 4Q26 revenue $445–505M (implies a record quarter). **CAPEX:** 9M-26 $20.8M (+44% vs $14.5M; 2.3% of revenue, Albuquerque ramp); no explicit FY27 capex guidance is given in the provided documents.

### Conclusions
- **Strengths:** record backlog & book-to-bill; asset-light model (capex ≈2% of sales); strong 9M cash conversion; domestic-content moat; merger tax shield; depressed multiples (fwd P/E ≈5.6x, P/S ≈0.67x).
- **Risks:** guidance cut and timing slippage; margin and EBITDA compression; leverage 3.9x net with converts/preferred ahead of common; customer concentration; policy dependence; merger dilution/execution; high short interest/volatility (beta ≈1.8).
- **Risk-return:** asymmetric but event-heavy — valuation discounts a lot of bad news, yet earnings visibility depends on a 4Q26 ramp and policy timing. Versus peers (e.g. NXT) ARRY trades at a steep discount with weaker margins and a more levered balance sheet.
- **Preliminary recommendation (not a buy/sell call):** **attractive enough to study further** — speculative value/turnaround; confirm 4Q26 delivery ramp, 45X recoveries, merger terms/vote and de-leveraging before sizing; use options structures (cash-secured puts / put spreads) to define risk.
"""


if __name__ == "__main__":
    main()
