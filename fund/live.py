"""
A live paper portfolio: the backtest's rules run forward, one month at a time.

    python -m fund.live

In the first days of each month, the agents score the nine sectors on the
last month-end close, and both combinations (the plain average and the ridge)
pick their top three. The picks go into results/live_picks.csv before the
month's returns exist. Later runs fill in each month's return for the picks
and for an equal-weight holding of all nine. The backtest found no reliable
signal, so the honest expectation is that the picks roughly match the
benchmark; this records whether that holds out of sample.
"""
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge

from fund.agents import ALL_AGENTS
from fund.backtest import (RIDGE_ALPHA, RIDGE_WINDOW, TICKERS, TOP_N, build_panel,
                           month_ends, rank_z)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "live_picks.csv"
README = ROOT / "README.md"
START, END = "<!-- live:start -->", "<!-- live:end -->"
PICK_WINDOW_DAYS = 10     # picks are only made early in the month they are for


def latest_prices():
    import yfinance as yf
    px = yf.download(TICKERS, start="1999-01-01", auto_adjust=True,
                     progress=False)["Close"][TICKERS].dropna()
    # Keep only completed months, so every month end is a real close.
    this_month = pd.Timestamp(date.today()).to_period("M")
    return px, px[px.index.to_period("M") < this_month]


def picks(closed):
    names = [a.name for a in ALL_AGENTS]
    t = month_ends(closed)[-1]
    x_t = pd.DataFrame({a.name: rank_z(a.score(closed, t)) for a in ALL_AGENTS})
    panel = build_panel(closed)
    dates = panel.index.get_level_values("date").unique()[-RIDGE_WINDOW:]
    hist = panel.loc[dates]
    y = hist["fwd"] - hist.groupby(level="date")["fwd"].transform("mean")
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(hist[names], y)
    forecasts = {"equal_weight": x_t[names].mean(axis=1),
                 "ridge": pd.Series(ridge.predict(x_t[names]), index=x_t.index)}
    top = {k: " ".join(sorted(f.nlargest(TOP_N).index)) for k, f in forecasts.items()}
    return t, top


def score(ledger, px):
    """Return over the month the picks were held, from one month-end close to
    the next. Filled only once that month has closed."""
    closes = px.groupby(px.index.to_period("M")).tail(1)
    for i, r in ledger[ledger["all_nine"].isna()].iterrows():
        held = pd.Period(r["month"], "M")
        start = closes[closes.index.to_period("M") == held - 1]
        end = closes[closes.index.to_period("M") == held]
        if start.empty or end.empty or end.index[0].to_period("M") >= pd.Timestamp(date.today()).to_period("M"):
            continue
        ret = end.iloc[0] / start.iloc[0] - 1
        for col in ("equal_weight", "ridge"):
            ledger.loc[i, f"{col}_return"] = ret[r[col].split()].mean()
        ledger.loc[i, "all_nine"] = ret.mean()
    return ledger


def write_section(lines):
    text = README.read_text(encoding="utf-8")
    before, rest = text.split(START)
    README.write_text(before + START + "\n" + "\n".join(lines) + "\n" + END + rest.split(END)[1],
                      encoding="utf-8")


def update_readme(ledger):
    pct = lambda v: f"{v:+.1%}" if pd.notna(v) else "pending"
    if ledger.empty:
        nxt = (pd.Timestamp(date.today()).to_period("M") + 1).strftime("%B %Y")
        write_section([f"The first picks will be made in the first days of {nxt}."])
        return
    lines =["| Month held | Average picks | Return | Ridge picks | Return | All nine |",
             "|---|---|---|---|---|---|"]
    for _, r in ledger.iloc[::-1].iterrows():
        lines.append(f"| {r.month} | {r.equal_weight} | {pct(r.equal_weight_return)} | "
                     f"{r.ridge} | {pct(r.ridge_return)} | {pct(r.all_nine)} |")
    done = ledger.dropna(subset=["all_nine"])
    lines.append("")
    if len(done):
        grow = lambda s: (1 + s).prod() - 1
        lines.append(f"Over {len(done)} completed month{'s' if len(done) > 1 else ''}: average picks "
                     f"{grow(done.equal_weight_return):+.1%}, ridge picks {grow(done.ridge_return):+.1%}, "
                     f"all nine sectors {grow(done.all_nine):+.1%}, before costs.")
    else:
        lines.append("No month has finished yet.")
    write_section(lines)


def main():
    px, closed = latest_prices()
    cols = ["month", "scored_on", "equal_weight", "ridge",
            "equal_weight_return", "ridge_return", "all_nine"]
    ledger = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=cols)
    ledger = score(ledger, px)

    month = pd.Timestamp(date.today()).to_period("M")
    have = ledger["month"].astype(str).tolist()
    if str(month) not in have and date.today().day <= PICK_WINDOW_DAYS:
        t, top = picks(closed)
        new = pd.DataFrame([{"month": str(month), "scored_on": t.strftime("%Y-%m-%d"), **top}])
        ledger = new if ledger.empty else pd.concat([ledger, new], ignore_index=True)
        print(f"{month}: average {top['equal_weight']}, ridge {top['ridge']}")
    ledger.to_csv(LEDGER, index=False, float_format="%.5f")
    update_readme(ledger)


if __name__ == "__main__":
    main()
