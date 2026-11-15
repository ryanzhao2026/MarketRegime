"""跨资产引擎：用「资产间相对表现」判断资金的风险偏好。

权重 15%。这是宏观基金最重视、而纯技术指标最缺失的维度。

四个特征：
  1. SPY / TLT — 股债相对强弱，风险偏好的第一顺位指标
  2. HYG / TLT — 信用 vs 久期，信用风险溢价的方向（最灵敏）
  3. GLD / SPY — 黄金 vs 股票，避险需求的直接读数（反向计分）
  4. DXY Trend — 美元趋势，全球流动性松紧（反向计分）

输出 risk_on_score ∈ [0, 100]：越高代表风险偏好越强。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import features as F
from data import (DOLLAR_CANDIDATES, EQUITY, GOLD, HIGH_YIELD, LONG_BOND,
                  MarketData)

FACTOR_WEIGHTS = {
    "spy_tlt": 0.30,
    "hyg_tlt": 0.30,
    "gld_spy": 0.20,
    "dxy": 0.20,
}

LOOKBACK = 63      # 季度视角，避免日内噪声
MA_FAST = 50


def _ratio_score(
    ratio: pd.Series,
    chg_points: list[tuple[float, float]],
    lookback: int = LOOKBACK,
    invert: bool = False,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """比值型特征打分：63 日变化（分段映射）+ 是否站上 50 日均线。"""
    chg = (ratio / ratio.shift(lookback) - 1.0) * 100.0
    base = F.piecewise(chg, chg_points)
    if invert:
        base = 100.0 - base
    ma = F.sma(ratio, MA_FAST)
    above = pd.Series(np.where(ma.notna(), (ratio > ma).astype(float), np.nan), index=ratio.index)
    if invert:
        above = 1.0 - above
    trend_score = (above * 100.0).clip(0.0, 100.0)
    return F.blend(base, trend_score, weights=[0.7, 0.3]), chg, ratio


def compute(md: MarketData) -> pd.DataFrame:
    idx = md.prices.index
    px = md.prices.ffill()

    def col(t: str) -> pd.Series | None:
        return px[t] if t in px.columns and px[t].notna().any() else None

    spy, tlt = col(EQUITY), col(LONG_BOND)
    hyg, gld = col(HIGH_YIELD), col(GOLD)
    # 美元指数列名随数据源不同（DXY / DINI / UUP），按候选顺序取第一个可用的
    dxy = next((c for c in (col(t) for t in DOLLAR_CANDIDATES) if c is not None), None)

    # ---------- 1. SPY / TLT
    if spy is not None and tlt is not None:
        ratio1 = F.safe_div(spy, tlt)
        s_spy_tlt, chg1, _ = _ratio_score(
            ratio1,
            [(-25.0, 0.0), (-12.0, 12.0), (-4.0, 32.0), (0.0, 50.0),
             (4.0, 68.0), (12.0, 88.0), (25.0, 100.0)],
        )
    else:
        ratio1 = chg1 = s_spy_tlt = pd.Series(np.nan, index=idx)

    # ---------- 2. HYG / TLT（信用 vs 久期）
    if hyg is not None and tlt is not None:
        ratio2 = F.safe_div(hyg, tlt)
        s_hyg_tlt, chg2, _ = _ratio_score(
            ratio2,
            [(-18.0, 0.0), (-9.0, 12.0), (-3.0, 34.0), (0.0, 50.0),
             (3.0, 68.0), (9.0, 88.0), (18.0, 100.0)],
        )
    else:
        ratio2 = chg2 = s_hyg_tlt = pd.Series(np.nan, index=idx)

    # ---------- 3. GLD / SPY（避险，反向）
    if gld is not None and spy is not None:
        ratio3 = F.safe_div(gld, spy)
        s_gld_spy, chg3, _ = _ratio_score(
            ratio3,
            [(-20.0, 0.0), (-10.0, 12.0), (-3.0, 32.0), (0.0, 50.0),
             (3.0, 68.0), (10.0, 88.0), (20.0, 100.0)],
            invert=True,
        )
    else:
        ratio3 = chg3 = s_gld_spy = pd.Series(np.nan, index=idx)

    # ---------- 4. DXY 趋势（反向：美元走强 = 收紧）
    if dxy is not None:
        chg4 = (dxy / dxy.shift(LOOKBACK) - 1.0) * 100.0
        base4 = F.piecewise(chg4, [(-8.0, 0.0), (-4.0, 15.0), (-1.0, 40.0), (0.0, 50.0),
                                   (1.0, 62.0), (4.0, 85.0), (8.0, 100.0)])
        ma50 = F.sma(dxy, MA_FAST)
        above = pd.Series(np.where(ma50.notna(), (dxy > ma50).astype(float), np.nan), index=dxy.index)
        s_dxy = F.blend(base4, (1.0 - above) * 100.0, weights=[0.65, 0.35])
    else:
        chg4 = s_dxy = pd.Series(np.nan, index=idx)

    # HYG 自身站上 MA50 的补充权重（信用市场的即时状态）
    factors = {
        "spy_tlt": s_spy_tlt,
        "hyg_tlt": s_hyg_tlt,
        "gld_spy": s_gld_spy,
        "dxy": s_dxy,
    }
    risk_on_score = F.weighted_mean(factors, FACTOR_WEIGHTS)

    out = pd.DataFrame(
        {
            "raw_spy_tlt": ratio1,
            "raw_spy_tlt_chg63": chg1,
            "raw_hyg_tlt": ratio2,
            "raw_hyg_tlt_chg63": chg2,
            "raw_gld_spy": ratio3,
            "raw_gld_spy_chg63": chg3,
            "raw_dxy_chg63": chg4,
            "spy_tlt_score": s_spy_tlt,
            "hyg_tlt_score": s_hyg_tlt,
            "gld_spy_score": s_gld_spy,
            "dxy_score": s_dxy,
            "risk_on_score": risk_on_score,
        },
        index=idx,
    )
    return out


def describe(row: pd.Series) -> dict:
    c1 = row.get("raw_spy_tlt_chg63", np.nan)
    c2 = row.get("raw_hyg_tlt_chg63", np.nan)
    c3 = row.get("raw_gld_spy_chg63", np.nan)
    c4 = row.get("raw_dxy_chg63", np.nan)
    score = row.get("risk_on_score", np.nan)

    def label(v, strong, weak):
        if pd.isna(v):
            return "未知"
        if v >= 3:
            return strong
        if v <= -3:
            return weak
        return "中性"

    return {
        "risk_on_score": float(score) if pd.notna(score) else None,
        "spy_tlt_chg63_pct": float(c1) if pd.notna(c1) else None,
        "credit_chg63_pct": float(c2) if pd.notna(c2) else None,
        "credit_label": label(c2, "信用走强", "信用走弱"),
        "gld_spy_chg63_pct": float(c3) if pd.notna(c3) else None,
        "dxy_chg63_pct": float(c4) if pd.notna(c4) else None,
        "dxy_label": ("美元走强(收紧)" if pd.notna(c4) and c4 >= 1.0
                      else "美元走弱(宽松)" if pd.notna(c4) and c4 <= -1.0 else "美元震荡"),
    }
