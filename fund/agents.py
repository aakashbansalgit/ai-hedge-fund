"""
Signal agents. Each one looks at daily prices up to a date and scores every
asset; higher means expected to do better next month. An agent never sees a
price after the date it is asked about, which tests/test_no_lookahead.py
checks by changing future prices and confirming no score moves.
"""
import numpy as np
import pandas as pd


class Agent:
    name = "agent"

    def score(self, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def _upto(prices, asof):
        return prices.loc[:asof]


class Momentum(Agent):
    """12-1 month momentum: the return from 12 months ago to 1 month ago.
    The latest month is skipped because one-month returns tend to reverse."""
    name = "momentum_12_1"

    def score(self, prices, asof):
        p = self._upto(prices, asof)
        if len(p) < 253:
            return pd.Series(np.nan, index=prices.columns)
        return p.iloc[-22] / p.iloc[-253] - 1


class ShortTermReversal(Agent):
    """Last month's losers tend to bounce: score is minus the 1-month return."""
    name = "reversal_1m"

    def score(self, prices, asof):
        p = self._upto(prices, asof)
        if len(p) < 22:
            return pd.Series(np.nan, index=prices.columns)
        return -(p.iloc[-1] / p.iloc[-22] - 1)


class LowVolatility(Agent):
    """Lower trailing volatility scores higher (the low-volatility anomaly)."""
    name = "low_vol_63d"

    def score(self, prices, asof):
        r = self._upto(prices, asof).pct_change().iloc[-63:]
        if len(r) < 63:
            return pd.Series(np.nan, index=prices.columns)
        return -r.std()


class Trend(Agent):
    """Distance of price above its 200-day average."""
    name = "trend_200d"

    def score(self, prices, asof):
        p = self._upto(prices, asof)
        if len(p) < 200:
            return pd.Series(np.nan, index=prices.columns)
        return p.iloc[-1] / p.iloc[-200:].mean() - 1


class RSIReversion(Agent):
    """Oversold on 14-day RSI scores higher: score is 50 minus RSI."""
    name = "rsi_14"

    def score(self, prices, asof):
        p = self._upto(prices, asof)
        if len(p) < 30:
            return pd.Series(np.nan, index=prices.columns)
        d = p.diff().iloc[-14:]
        gain = d.clip(lower=0).mean()
        loss = (-d.clip(upper=0)).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        return 50 - rsi


ALL_AGENTS = [Momentum(), ShortTermReversal(), LowVolatility(), Trend(), RSIReversion()]
