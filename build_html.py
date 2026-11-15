"""
build_html.py — generates the static, self-contained dashboard.html from the SAME analytics used by app.py.

    python build_html.py            # needs internet (Yahoo Finance, Finviz, Google News) + requirements.txt

It downloads data for ARRY + peers + indices, runs the ARIMA / ARIMA-GARCH forecast, option-chain analytics and
horizon scoring, embeds everything as JSON in dashboard_template.html and inlines Apache ECharts so the output
file works offline.  Indicators, candlestick signals (and their filters), options ideas and charts are
re-computed live in the browser, so the left control panel is fully interactive.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import app as A

HERE = Path(__file__).parent
TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "ARRY"
ECHARTS_URL = "https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"


def pack(df: pd.DataFrame) -> dict:
    return {"d": [x.strftime("%Y-%m-%d") for x in df.index], "o": df["Open"].round(4).tolist(),
            "h": df["High"].round(4).tolist(), "l": df["Low"].round(4).tolist(),
            "c": df["Close"].round(4).tolist(), "v": df["Volume"].astype(int).tolist()}


def clean(o):
    """JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.strftime("%Y-%m-%d")
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def md_to_html(md: str) -> str:
    """Tiny markdown subset (###, bullets, tables, **bold**, *italic*)."""
    def inline(t):
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        return re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", t)
    out, lines, i = [], md.strip("\n").split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("### "):
            out.append(f"<h3>{inline(ln[4:])}</h3>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")]); i += 1
            i -= 1
            head, body = rows[0], [r for r in rows[2:]]
            out.append("<div class='tw'><table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>" +
                       "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body) + "</tbody></table></div>")
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(inline(lines[i][2:])); i += 1
            i -= 1
            out.append("<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>")
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "\n".join(out)


def frame_to_table(df: pd.DataFrame, scale=1e6) -> dict:
    if df is None or df.empty:
        return {"cols": [], "rows": []}
    df = df.dropna(how="all")
    cols = [pd.Timestamp(c).strftime("%Y-%m-%d") for c in df.columns]
    rows = []
    for lab, r in df.iterrows():
        vals = []
        for v in r.values:
            try:
                f = float(v)
                vals.append(None if np.isnan(f) else (round(f / scale, 3) if abs(f) > 1000 else round(f, 4)))
            except Exception:
                vals.append(None)
        rows.append([str(lab)] + vals)
    return {"cols": cols, "rows": rows}


def main():
    t0 = time.time()
    peers_default = A.PEERS_DEFAULT
    universe = [TICKER] + peers_default
    benches = list(A.BENCHMARKS.values())
    print("prices…")
    prices, bad = {}, []
    for t in universe + benches:
        try:
            prices[t] = pack(A.get_history(t, 3))
        except Exception as e:
            bad.append(t); print("  skip", t, e)
    px = A.get_history(TICKER, 3)
    info = A.get_info(TICKER)
    print("statements / finviz / news…")
    stm = A.get_statements(TICKER)
    fv = A.get_finviz(TICKER)
    peers = [p for p in dict.fromkeys(fv.get("peers", []) + peers_default) if p != TICKER][:9]
    peer_snap = {}
    for p in [TICKER] + peers:
        peer_snap[p] = A.get_finviz(p).get("snapshot", {}) if p != TICKER else fv.get("snapshot", {})
        time.sleep(0.4)
    company = info.get("shortName", "Array Technologies")
    news = A.score_sentiment(A.filter_relevant(fv.get("news", []) + A.get_news(TICKER, company), TICKER, company))
    print("indicators / signals / forecast…")
    d = A.add_indicators(px, 10)
    d["EMA28"] = A.ema(d["Close"], 28)
    sig = A.build_signals(d)
    fc = A.forecast_models(px["Close"], 252, 3000, 0.5)
    order = tuple(int(x) for x in re.findall(r"\d", fc["arima"]["order"])[::2])
    wf = A.walk_forward(px["Close"], order)
    hv, hv90, hvp = A.realized_vol(px["Close"]), A.realized_vol(px["Close"], 90), A.hv_percentile(px["Close"])
    ins = A.horizon_insights(d, sig, fc, info, fv, news, hv)
    S = float(px["Close"].iloc[-1])
    print("options…")
    opts = A.get_options(TICKER, targets=(7, 91, 182, 365))
    chains = {}
    for e, ch in opts["chains"].items():
        def rows(df):
            return df[["strike", "bid", "ask", "lastPrice", "impliedVolatility", "openInterest", "volume"]].fillna(0).round(4).values.tolist()
        chains[e] = {"dte": ch["dte"], "calls": rows(ch["calls"]), "puts": rows(ch["puts"])}
    playbook = {}
    for h, tgt in (("1 week", 7), ("3 months", 91), ("6 months", 182), ("12 months", 365)):
        if not opts["chains"]:
            break
        e = min(opts["chains"], key=lambda x: abs(opts["chains"][x]["dte"] - tgt))
        ch = opts["chains"][e]
        aiv = A.atm_iv({e: ch}, S)
        ivhv = aiv / hv90 if hv90 else 1.0
        ideas = A.option_ideas(ch, S, 0.042, ins["horizons"][h]["bias"], ivhv, float(d["RSI"].iloc[-1]))
        for i in ideas:
            i.pop("x", None); i.pop("pl", None)
        playbook[h] = {"expiry": e, "dte": ch["dte"], "atm_iv": aiv, "iv_hv": ivhv, "ideas": ideas}
    q = A.quarterly_series(stm, info.get("sharesOutstanding"))
    vh = A.valuation_history(px["Close"], q, info.get("sharesOutstanding"))
    qd = {"d": [x.strftime("%b-%y") for x in q.index]}
    for k in ("Revenue", "GrossProfit", "OpIncome", "NetIncome", "EPS", "OCF", "FCF", "OCF_ps", "FCF_ps", "GM", "OM"):
        if k in q:
            qd[k] = q[k].tolist()
    vhd = {"d": [x.strftime("%Y-%m-%d") for x in vh.index]}
    for k in vh.columns:
        vhd[k] = vh[k].round(3).tolist()
    hz = ins["horizons"]
    summary = [
        "<b>Thesis:</b> record <b>$2.6B orderbook (+42% YoY, 1.6x book-to-bill)</b> against margin compression (3Q26 adj. gross margin 22.8%, adj. EBITDA margin 9.7%) and a <b>FY26 guidance cut</b>; GAAP net loss to common of −$7.9M after $16.1M of preferred dividends.",
        "<b>Catalyst / risk:</b> stock-for-stock merger with <b>Shoals (SHLS)</b> — $1.6B tax shield + $237M NOLs, ~65% dilution to ARRY holders; net debt/EBITDA <b>3.9x</b>, ~$466M liquidity.",
        "<b>Technical:</b> " + "; ".join(ins["tech_notes"][:4]) + ".",
        f"<b>Forecast (ARIMA + ARIMA-GARCH):</b> 3M median ${hz['3 months']['median']:.2f} ({hz['3 months']['ret_median']:+.0%}, P(up) {hz['3 months']['p_up']:.0%}); 6M median ${hz['6 months']['median']:.2f} (5–95%: ${hz['6 months']['p5']:.2f}–${hz['6 months']['p95']:.2f}).",
        f"<b>Preliminary view:</b> 1W <b>{hz['1 week']['rating']}</b> · 3M <b>{hz['3 months']['rating']}</b> · 6M <b>{hz['6 months']['rating']}</b> · 12M <b>{hz['12 months']['rating']}</b> — speculative turnaround / value; size accordingly.",
    ]
    fc_out = {"S0": fc["S0"], "adf_price_p": fc["adf_price_p"], "adf_ret_p": fc["adf_ret_p"], "prob": fc["prob"],
              "arima": {"order": fc["arima"]["order"], "aic": fc["arima"]["aic"],
                        "paths": {k: v.round(3).tolist() for k, v in fc["arima"]["paths"].items()}},
              "garch": {"order": fc["garch"]["order"], "cond_vol_ann": fc["garch"]["cond_vol_ann"], "nu": fc["garch"]["nu"],
                        "paths": {k: v.round(3).tolist() for k, v in fc["garch"]["paths"].items()}},
              "consensus": {k: v.round(3).tolist() for k, v in fc["consensus"].items()}, "wf": wf}
    keep = ["shortName", "longName", "marketCap", "forwardPE", "trailingPE", "priceToSalesTrailing12Months", "targetMeanPrice",
            "targetHighPrice", "targetLowPrice", "recommendationKey", "numberOfAnalystOpinions", "shortPercentOfFloat", "beta",
            "sector", "industry", "fullTimeEmployees", "sharesOutstanding", "enterpriseValue", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"]
    payload = {
        "ticker": TICKER, "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "S": S,
        "prices": prices, "benchmarks": {k: v for k, v in A.BENCHMARKS.items() if v in prices},
        "peers": peers, "info": {k: info.get(k) for k in keep}, "finviz": {k: fv.get(k) for k in ("profile", "sector", "country", "ratings", "etfs")},
        "snap": peer_snap, "facts": A.FACTS, "reported": A.REPORTED, "ratios": A.reported_ratios().to_dict("records"),
        "quarterly": qd, "valhist": vhd, "forecast": fc_out, "options": {"chains": chains, "playbook": playbook},
        "hv": {"hv30": hv, "hv90": hv90, "pct": hvp}, "insights": ins, "summary": summary, "news": news[:45],
        "stmts": {k: frame_to_table(stm.get(k)) for k in ("income", "balance", "cashflow")},
        "report_html": md_to_html(A.REPORT_ANALYSIS_MD), "tree_summary_html": md_to_html(A.REVENUE_TREE_SUMMARY),
        "py_signals": [{"date": r.date.strftime("%Y-%m-%d"), "side": r.side, "state": r.state} for r in sig.itertuples()],
    }
    print("echarts…")
    try:
        echarts = urllib.request.urlopen(urllib.request.Request(ECHARTS_URL, headers=A.UA), timeout=60).read().decode("utf-8")
        echarts_tag = f"<script>{echarts}</script>"
    except Exception as e:
        print("  could not inline echarts (", e, ") -> CDN tag")
        echarts_tag = f'<script src="{ECHARTS_URL}"></script>'
    tpl = (HERE / "dashboard_template.html").read_text(encoding="utf-8")
    data_js = "const D = " + json.dumps(clean(payload), separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/") + ";"
    html = tpl.replace("<!--__ECHARTS__-->", echarts_tag).replace("/*__PAYLOAD__*/", data_js)
    (HERE / "dashboard.html").write_text(html, encoding="utf-8")
    print(f"dashboard.html written ({len(html) / 1e6:.2f} MB) in {time.time() - t0:.0f}s; missing tickers: {bad}")


if __name__ == "__main__":
    main()
