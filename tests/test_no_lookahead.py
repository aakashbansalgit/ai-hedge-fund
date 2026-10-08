"""No agent and no forecast at month t may depend on prices after t.

Each test changes every price after a chosen month end and checks that
nothing computed for that month end moves.

Run from the repo root: python -m pytest tests  (or python tests/test_no_lookahead.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fund import backtest as bt
from fund.agents import ALL_AGENTS


def _shock_after(prices, t):
    shocked = prices.copy()
    after = shocked.index > t
    rng = np.random.default_rng(0)
    shocked.loc[after] *= np.exp(rng.normal(0, 0.2, size=shocked.loc[after].shape))
    return shocked


def test_agents_use_only_past_prices():
    prices = bt.load_prices()
    for t in bt.month_ends(prices)[[24, 120, 250]]:
        shocked = _shock_after(prices, t)
        for agent in ALL_AGENTS:
            a, b = agent.score(prices, t), agent.score(shocked, t)
            assert np.allclose(a, b, equal_nan=True), (agent.name, t)


def test_ridge_forecast_uses_only_completed_months():
    prices = bt.load_prices().loc[:"2012-12-31"]
    t = bt.month_ends(prices)[-13]
    panel = bt.build_panel(prices)
    shocked = bt.build_panel(_shock_after(prices, t))
    _, learned, _ = bt.forecasts(panel)
    _, learned_shocked, _ = bt.forecasts(shocked)
    assert np.allclose(learned.loc[:t], learned_shocked.loc[:t])
    # and the shock does change later forecasts, so the test has teeth
    assert not np.allclose(learned.loc[t:].iloc[1:], learned_shocked.loc[t:].iloc[1:])


if __name__ == "__main__":
    test_agents_use_only_past_prices()
    test_ridge_forecast_uses_only_completed_months()
    print("no agent or forecast uses future prices")
