"""
Monthly cross-sectional backtest of the signal agents on US sector ETFs.

At each month end every agent scores the nine original SPDR sector funds. The
portfolio manager turns the scores into one forecast in two ways: an equal
average of the agents' rank scores, and a ridge regression refitted every
month on the previous five years of scores and realised next-month returns.
The forecast picks the top three sectors to hold long, and for the
long-short book the bottom three to short.

    python -m fund.backtest

Writes results/agent_ic.csv, results/portfolios.csv and figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from fund.agents import ALL_AGENTS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "prices.csv"
# The nine sector funds that have traded since December 1998. Using funds
# that existed for the whole sample avoids survivorship bias.
TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
TX_COST = 0.001        # 10 bp per unit of turnover
TOP_N = 3
RIDGE_WINDOW = 60      # months of history for the learned combination
RIDGE_ALPHA = 1.0


def load_prices():
    if DATA.exists():
        return pd.read_csv(DATA, index_col=0, parse_dates=True)
    import yfinance as yf
    px = yf.download(TICKERS, start="1999-01-01", end="2025-10-01",
                     auto_adjust=True, progress=False)["Close"][TICKERS].dropna()
    DATA.parent.mkdir(exist_ok=True)
    px.to_csv(DATA)
    return px


def month_ends(prices):
    return prices.groupby(prices.index.to_period("M")).tail(1).index


def rank_z(s: pd.Series) -> pd.Series:
    """Cross-sectional ranks scaled to mean 0, standard deviation 1. Ranks keep
    one extreme asset from dominating a nine-asset cross-section."""
    r = s.rank()
    return (r - r.mean()) / r.std()


def build_panel(prices):
    """Scores at each month end and the return over the following month."""
    dates = month_ends(prices)
    rows = []
    for t, t_next in zip(dates[:-1], dates[1:]):
        fwd = prices.loc[t_next] / prices.loc[t] - 1
        scores = {a.name: a.score(prices, t) for a in ALL_AGENTS}
        if any(s.isna().any() for s in scores.values()):
            continue
        df = pd.DataFrame({k: rank_z(v) for k, v in scores.items()})
        df["fwd"] = fwd
        df["date"] = t
        rows.append(df)
    panel = pd.concat(rows)
    panel.index.name = "ticker"
    return panel.reset_index().set_index(["date", "ticker"])


def agent_ic(panel):
    names = [a.name for a in ALL_AGENTS]
    ic = panel.groupby(level="date").apply(
        lambda g: pd.Series({n: stats.spearmanr(g[n], g["fwd"])[0] for n in names}))
    n = len(ic)
    return pd.DataFrame({
        "mean_ic": ic.mean(),
        "ic_std": ic.std(),
        "ic_ir_annual": ic.mean() / ic.std() * np.sqrt(12),
        "t_stat": ic.mean() / (ic.std() / np.sqrt(n)),
        "hit_rate": (ic > 0).mean(),
        "months": n,
    }), ic


def forecasts(panel):
    """Equal-weight and ridge forecasts. At date t the ridge only sees rows
    whose forward return finished by t, that is, dates strictly before t."""
    names = [a.name for a in ALL_AGENTS]
    dates = panel.index.get_level_values("date").unique()
    eq, learned, coefs = [], [], {}
    for i, t in enumerate(dates):
        x_t = panel.loc[t, names]
        eq.append(x_t.mean(axis=1).rename(t))
        if i < RIDGE_WINDOW:
            continue
        hist = panel.loc[dates[i - RIDGE_WINDOW:i]]
        y = hist["fwd"] - hist.groupby(level="date")["fwd"].transform("mean")
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(hist[names], y)
        coefs[t] = pd.Series(model.coef_, index=names)
        learned.append(pd.Series(model.predict(x_t), index=x_t.index, name=t))
    return (pd.DataFrame(eq), pd.DataFrame(learned), pd.DataFrame(coefs).T)


def portfolio_returns(forecast, panel, long_short):
    fwd = panel["fwd"].unstack()
    weights = []
    for t, f in forecast.iterrows():
        w = pd.Series(0.0, index=f.index)
        order = f.sort_values()
        w[order.index[-TOP_N:]] = 1 / TOP_N
        if long_short:
            w[order.index[:TOP_N]] = -1 / TOP_N
        weights.append(w.rename(t))
    w = pd.DataFrame(weights)
    gross = (w * fwd.loc[w.index]).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    return gross - TX_COST * turnover, turnover


def summary(ret, turnover, bench=None):
    nav = (1 + ret).cumprod()
    out = {
        "annual_return": nav.iloc[-1] ** (12 / len(ret)) - 1,
        "annual_vol": ret.std() * np.sqrt(12),
        "return_to_vol": ret.mean() / ret.std() * np.sqrt(12),   # no risk-free deduction
        "max_drawdown": (1 - nav / nav.cummax()).max(),
        "annual_turnover": turnover.sum() * 12 / len(ret),
    }
    if bench is not None:
        active = ret - bench.loc[ret.index]
        out["active_return"] = active.mean() * 12
        out["tracking_error"] = active.std() * np.sqrt(12)
        out["information_ratio"] = active.mean() / active.std() * np.sqrt(12)
    return out


def main():
    prices = load_prices()
    panel = build_panel(prices)
    ic_table, ic_series = agent_ic(panel)

    eq, learned, coefs = forecasts(panel)
    eq = eq.loc[learned.index]          # compare both over the same months
    bench = panel["fwd"].unstack().mean(axis=1)

    rows, curves = {}, {}
    for label, f in (("equal weight", eq), ("ridge", learned)):
        for ls in (False, True):
            ret, to = portfolio_returns(f, panel, ls)
            key = f"{label}, {'long-short' if ls else 'long only'}"
            rows[key] = summary(ret, to, None if ls else bench)
            curves[key] = ret
    rows["sector equal weight"] = summary(bench.loc[learned.index],
                                          pd.Series(0.0, index=learned.index))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    ic_table.to_csv(out / "agent_ic.csv", float_format="%.4f")
    pd.DataFrame(rows).T.to_csv(out / "portfolios.csv", float_format="%.4f")
    coefs.to_csv(out / "ridge_coefficients.csv", float_format="%.5f")

    figs = ROOT / "figures"
    figs.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in ("equal weight, long only", "ridge, long only"):
        (1 + curves[k]).cumprod().plot(ax=ax, label=k)
    (1 + bench.loc[learned.index]).cumprod().plot(ax=ax, label="sector equal weight", color="black", lw=1)
    ax.set_yscale("log")
    ax.set_title("Top three sectors, net of 10 bp per unit of turnover")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "long_only.png", dpi=120)

    fig, ax = plt.subplots(figsize=(10, 4))
    coefs.plot(ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Ridge weight on each agent, refitted monthly on the prior 60 months")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "ridge_coefficients.png", dpi=120)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(ic_table)
    print()
    print(pd.DataFrame(rows).T)
    print(f"\n{learned.index[0]:%b %Y} to {learned.index[-1]:%b %Y}, {len(learned)} months")


if __name__ == "__main__":
    main()
