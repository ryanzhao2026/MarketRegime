"""跨资产引擎 v2：用「资产间相对表现」判断资金的风险偏好。

权重 15%（不变）。这是宏观基金最重视、而纯技术指标最缺失的维度。

四个子因子：
  1. SPY / TLT       — 股债相对强弱，风险偏好的第一顺位指标
  2. **信用价差**     — v2 核心替换（原 HYG/TLT）
  3. GLD / SPY       — 黄金 vs 股票，避险需求的直接读数（反向计分）
  4. 美元趋势         — 官方广义贸易加权美元指数（v2 优先），反向计分

v2 对信用维度做了什么
--------------------
v1 用 HYG/TLT 比值当信用代理，但 HYG 与 TLT 都含**久期暴露**：当利率下行时
两者同涨，比值变化主要反映久期而非信用。v2 改用**纯信用价差**：

    主口径：CREDIT_SPREAD = 穆迪 Baa 与 10Y 国债利差（FRED BAA10Y）
            已剥离国债基准，纯粹反映「企业还能不能借到钱」
    副口径：HY_OAS（ICE BofA 高收益 OAS），更贴高收益债但仅近 3 年

若两者都缺失，自动退回 v1 的 HYG/TLT 口径，并在解释层标注所用口径。

分档阈值的来源
------------
由 FRED BAA10Y 实测分位标定（2006-01-03 → 2026-11-12，5215 个交易日）：

    水平 p1=1.44 p5=1.55 p25=1.83 p50=2.25 p75=2.85 p95=3.46 p99=5.68
         2008-12 峰值 6.16，2020-03 峰值 4.31，2022-10 峰值 2.37
    63 日变化 p5=-0.51pp p50=-0.04pp p95=+0.58pp p99=+2.09pp

建模声明
--------
价差水平与「风险偏好」按线性反向映射（价差越窄 → Risk-On 越强）。
这意味着**极窄价差会被读作强风险偏好**，尽管它也可能代表自满。
这一方向选择与 v1 一致（v1 也是「信用走强 = Risk-On」），保持跨版本可比；
若改为「极窄=自满风险」，会反转该支柱的经济含义，属于另一次模型变更。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import features as F
from data import (CREDIT_SPREAD, DOLLAR_BROAD, DOLLAR_CANDIDATES, EQUITY,
                  GOLD, HIGH_YIELD, HY_OAS, LONG_BOND, MarketData)

FACTOR_WEIGHTS = {
    "spy_tlt": 0.30,
    "credit": 0.30,
    "gld_spy": 0.20,
    "dxy": 0.20,
}

LOOKBACK = 63      # 季度视角，避免日内噪声
MA_FAST = 50

# 信用价差「风险度」分档（0=极宽松，100=危机），由实测分位标定
CREDIT_LEVEL_POINTS = [
    (1.40, 0.0),      # p1 以下，历史极窄
    (1.55, 12.0),     # p5
    (1.83, 30.0),     # p25
    (2.25, 45.0),     # p50
    (2.85, 62.0),     # p75
    (3.46, 80.0),     # p95
    (4.31, 90.0),     # 2020-03 峰值
    (6.16, 100.0),    # 2008-12 峰值
]
CREDIT_CHG_POINTS = [
    (-0.60, 0.0),     # 快速收窄
    (-0.30, 25.0),
    (-0.04, 48.0),    # p50
    (0.30, 60.0),
    (0.58, 72.0),     # p95 走阔
    (1.00, 85.0),
    (2.09, 100.0),    # p99 急剧走阔
]


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


def _credit_factor(
    md: MarketData, idx: pd.Index, px: pd.DataFrame, hyg, tlt
) -> tuple[pd.Series, pd.Series, pd.Series, str]:
    """返回 (Risk-On 方向的信用分, 原始水平, 63 日变化, 实际使用的口径)。"""
    for name in (CREDIT_SPREAD, HY_OAS):
        if name in px.columns and px[name].notna().sum() >= 60:
            s = px[name].reindex(idx).ffill(limit=10)
            risk = F.piecewise(s, CREDIT_LEVEL_POINTS)
            chg = s - s.shift(LOOKBACK)          # 单位 pp，不是百分比
            chg_score = F.piecewise(chg, CREDIT_CHG_POINTS)
            risk_blend = F.blend(risk, chg_score, weights=[0.67, 0.33])
            return (100.0 - risk_blend).clip(0.0, 100.0), s, chg, name

    # 回退：v1 的 HYG/TLT 口径（含久期，仅作降级用）
    if hyg is not None and tlt is not None:
        ratio = F.safe_div(hyg, tlt)
        score, chg, _ = _ratio_score(
            ratio,
            [(-18.0, 0.0), (-9.0, 12.0), (-3.0, 34.0), (0.0, 50.0),
             (3.0, 68.0), (9.0, 88.0), (18.0, 100.0)],
        )
        return score, ratio, chg, "HYG/TLT(fallback)"

    nan = pd.Series(np.nan, index=idx)
    return nan, nan, nan, "unavailable"


def compute(md: MarketData) -> pd.DataFrame:
    idx = md.prices.index
    px = md.prices.ffill()

    def col(t: str) -> pd.Series | None:
        return px[t] if t in px.columns and px[t].notna().any() else None

    spy, tlt = col(EQUITY), col(LONG_BOND)
    hyg, gld = col(HIGH_YIELD), col(GOLD)
    # 美元口径优先级：官方广义指数 > DXY > DINI > UUP 代理
    dxy = next((c for c in (col(t) for t in DOLLAR_CANDIDATES) if c is not None), None)
    dxy_name = next((t for t in DOLLAR_CANDIDATES if col(t) is not None), "unavailable")

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

    # ---------- 2. 信用（v2：纯信用价差优先）
    s_credit, raw_credit, chg_credit, credit_source = _credit_factor(
        md, idx, px, hyg, tlt
    )

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

    # ---------- 4. 美元趋势（反向：美元走强 = 收紧）
    if dxy is not None:
        chg4 = (dxy / dxy.shift(LOOKBACK) - 1.0) * 100.0
        base4 = F.piecewise(chg4, [(-8.0, 0.0), (-4.0, 15.0), (-1.0, 40.0), (0.0, 50.0),
                                   (1.0, 62.0), (4.0, 85.0), (8.0, 100.0)])
        ma50 = F.sma(dxy, MA_FAST)
        above = pd.Series(np.where(ma50.notna(), (dxy > ma50).astype(float), np.nan), index=dxy.index)
        s_dxy = F.blend(base4, (1.0 - above) * 100.0, weights=[0.65, 0.35])
    else:
        chg4 = s_dxy = pd.Series(np.nan, index=idx)

    factors = {
        "spy_tlt": s_spy_tlt,
        "credit": s_credit,
        "gld_spy": s_gld_spy,
        "dxy": s_dxy,
    }
    risk_on_score = F.weighted_mean(factors, FACTOR_WEIGHTS)

    out = pd.DataFrame(
        {
            "raw_spy_tlt": ratio1,
            "raw_spy_tlt_chg63": chg1,
            "raw_credit": raw_credit,
            "raw_credit_chg63": chg_credit,
            "raw_gld_spy": ratio3,
            "raw_gld_spy_chg63": chg3,
            "raw_dxy_chg63": chg4,
            "spy_tlt_score": s_spy_tlt,
            "credit_score": s_credit,
            "gld_spy_score": s_gld_spy,
            "dxy_score": s_dxy,
            "risk_on_score": risk_on_score,
        },
        index=idx,
    )
    # 口径标记（字符串列，供解释层与归档使用）
    out["credit_source"] = pd.Series(credit_source, index=idx, dtype="object")
    out["dollar_source"] = pd.Series(dxy_name, index=idx, dtype="object")
    return out


def describe(row: pd.Series) -> dict:
    c1 = row.get("raw_spy_tlt_chg63", np.nan)
    c2 = row.get("raw_credit_chg63", np.nan)
    c3 = row.get("raw_gld_spy_chg63", np.nan)
    c4 = row.get("raw_dxy_chg63", np.nan)
    raw_credit = row.get("raw_credit", np.nan)
    src = row.get("credit_source", "unavailable")
    dsrc = row.get("dollar_source", "unavailable")
    score = row.get("risk_on_score", np.nan)

    def label(v, strong, weak):
        if pd.isna(v):
            return "未知"
        if v >= 3:
            return strong
        if v <= -3:
            return weak
        return "中性"

    # 信用口径不同，阈值含义不同：价差用 pp，比值用 %
    if src == CREDIT_SPREAD or src == HY_OAS:
        if pd.isna(c2):
            credit_label = "未知"
        elif c2 <= -0.15:
            credit_label = "信用走强(价差收窄)"
        elif c2 >= 0.15:
            credit_label = "信用走弱(价差走阔)"
        else:
            credit_label = "信用平稳"
    else:
        credit_label = label(c2, "信用走强", "信用走弱")

    return {
        "risk_on_score": float(score) if pd.notna(score) else None,
        "spy_tlt_chg63_pct": float(c1) if pd.notna(c1) else None,
        "credit_source": str(src),
        "credit_spread_pct": float(raw_credit) if pd.notna(raw_credit) else None,
        "credit_chg63": float(c2) if pd.notna(c2) else None,
        "credit_chg63_pct": float(c2) if pd.notna(c2) else None,
        "credit_label": credit_label,
        "gld_spy_chg63_pct": float(c3) if pd.notna(c3) else None,
        "dollar_source": str(dsrc),
        "dxy_chg63_pct": float(c4) if pd.notna(c4) else None,
        "dxy_label": ("美元走强(收紧)" if pd.notna(c4) and c4 >= 1.0
                      else "美元走弱(宽松)" if pd.notna(c4) and c4 <= -1.0 else "美元震荡"),
    }
