"""趋势引擎：判断市场是否存在「可交易的、方向一致的」趋势。

权重 25%。五个因子：
  1. ADX                     — 趋势强度（Wilder 14），方向由均线结构给出
  2. Trend Efficiency        — Kaufman 效率系数，净位移/路径长度
  3. EMA200 Distance         — 价格相对长期均线的偏离度
  4. Multi-Filter Agreement  — 多过滤器一致性（5 个布尔条件投票）
  5. Slope Consistency       — 滚动斜率的方向稳定度

输出 trend_score ∈ [0, 100]，50 为中性，>50 为多头趋势，<50 为空头趋势。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import features as F
from data import EQUITY, MarketData

# 子因子权重（合计 1.0）
FACTOR_WEIGHTS = {
    "adx": 0.25,
    "efficiency": 0.20,
    "ema200_distance": 0.20,
    "agreement": 0.20,
    "slope_consistency": 0.15,
}

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200
ADX_LEN = 14
EFF_LEN = 20
SLOPE_LEN = 20
SLOPE_LOOKBACK = 60


def _direction(price: pd.Series, ema_slow: pd.Series) -> pd.Series:
    """方向符号：价格在 EMA200 上方为 +1，下方为 -1。"""
    return np.where(price >= ema_slow, 1.0, -1.0)


def compute(md: MarketData, ticker: str = EQUITY) -> pd.DataFrame:
    """计算趋势因子原始值、子分数与合成趋势分。

    返回 DataFrame，列包含原始因子（raw_*）与子分数（*_score）以及 trend_score。
    """
    price = md.series(ticker)
    if price.empty:
        raise ValueError(f"{ticker} 无有效价格")

    ema_f, ema_m, ema_s = F.ema(price, EMA_FAST), F.ema(price, EMA_MID), F.ema(price, EMA_SLOW)
    direction = pd.Series(_direction(price, ema_s), index=price.index)

    # ---------- 1. ADX（有 OHLC 用真实 ADX，否则用收盘价构造的高低价代理）
    if md.has_ohlc():
        o = md.ohlc.reindex(price.index)
        adx_v, pdi, mdi = F.adx(o["high"], o["low"], o["close"], ADX_LEN)
    else:
        hi = price.rolling(2, min_periods=1).max()
        lo = price.rolling(2, min_periods=1).min()
        adx_v, pdi, mdi = F.adx(hi, lo, price, ADX_LEN)
    # ADX 本身无方向，用 DI 差值定方向；ADX 只贡献强度
    di_sign = pd.Series(np.where(pdi >= mdi, 1.0, -1.0), index=price.index)
    adx_strength = (adx_v / 40.0).clip(0.0, 1.0)          # 40 以上视为强趋势
    adx_score = (50.0 + adx_strength * 50.0 * di_sign).clip(0.0, 100.0)

    # ---------- 2. Trend Efficiency（0~1）× 方向
    eff = F.trend_efficiency(price, EFF_LEN)
    eff_score = (50.0 + (eff * direction).clip(-1.0, 1.0) * 50.0).clip(0.0, 100.0)

    # ---------- 3. EMA200 距离（百分比），绝对刻度 + 滚动分位混合
    dist = F.safe_div(price - ema_s, ema_s) * 100.0
    dist_abs = F.piecewise(
        dist,
        [(-12.0, 0.0), (-6.0, 15.0), (-2.0, 38.0), (0.0, 50.0),
         (2.0, 62.0), (6.0, 80.0), (12.0, 95.0), (20.0, 100.0)],
    )
    dist_rank = F.rolling_pct_rank(dist, 252, 120)
    dist_score = F.blend(dist_abs, dist_rank, weights=[0.7, 0.3])

    # ---------- 4. Multi-Filter Agreement：5 个布尔条件投票
    macd_line, macd_sig, macd_hist = F.macd(price)
    slope_m = F.rolling_slope(ema_m, SLOPE_LEN, normalize=True)
    filters = pd.DataFrame(
        {
            "price>ema200": (price > ema_s).astype(float),
            "ema20>ema50": (ema_f > ema_m).astype(float),
            "ema50>ema200": (ema_m > ema_s).astype(float),
            "macd>signal": (macd_line > macd_sig).astype(float),
            "ema50_slope>0": (slope_m > 0).astype(float),
        },
        index=price.index,
    )
    # 数据未成熟阶段（EMA200 缺失）不强行投票
    valid = ema_s.notna() & macd_sig.notna() & slope_m.notna()
    agreement_raw = filters.mean(axis=1).where(valid)
    agreement_score = (agreement_raw * 100.0).clip(0.0, 100.0)

    # ---------- 5. Slope Consistency：近 60 日内，20 日斜率为正的天数占比
    slope_pos = (F.rolling_slope(price, SLOPE_LEN, normalize=True) > 0).astype(float)
    slope_pos = slope_pos.where(price.rolling(SLOPE_LEN, min_periods=SLOPE_LEN).count() >= SLOPE_LEN)
    slope_consistency = slope_pos.rolling(SLOPE_LOOKBACK, min_periods=20).mean()
    slope_score = (slope_consistency * 100.0).clip(0.0, 100.0)

    factors = {
        "adx": adx_score,
        "efficiency": eff_score,
        "ema200_distance": dist_score,
        "agreement": agreement_score,
        "slope_consistency": slope_score,
    }
    trend_score = F.weighted_mean(factors, FACTOR_WEIGHTS)

    out = pd.DataFrame(
        {
            "raw_adx": adx_v,
            "raw_plus_di": pdi,
            "raw_minus_di": mdi,
            "raw_efficiency": eff,
            "raw_ema200_dist_pct": dist,
            "raw_agreement": agreement_raw,
            "raw_slope_consistency": slope_consistency,
            "raw_ema20": ema_f,
            "raw_ema50": ema_m,
            "raw_ema200": ema_s,
            "adx_score": adx_score,
            "efficiency_score": eff_score,
            "ema200_distance_score": dist_score,
            "agreement_score": agreement_score,
            "slope_consistency_score": slope_score,
            "trend_score": trend_score,
        },
        index=price.index,
    )
    return out


def describe(row: pd.Series) -> dict:
    """把最新一行因子翻译成人类可读标签（供解释层使用）。"""
    adx_v = row.get("raw_adx", np.nan)
    agreement = row.get("raw_agreement", np.nan)
    dist = row.get("raw_ema200_dist_pct", np.nan)
    eff = row.get("raw_efficiency", np.nan)

    if pd.isna(adx_v):
        strength = "未知"
    elif adx_v >= 40:
        strength = "极强趋势"
    elif adx_v >= 25:
        strength = "趋势成型"
    elif adx_v >= 18:
        strength = "趋势初现"
    else:
        strength = "无趋势/震荡"

    if pd.isna(agreement):
        align = "未知"
    elif agreement >= 0.8:
        align = "多过滤器高度一致"
    elif agreement <= 0.2:
        align = "多过滤器一致向下"
    else:
        align = "多过滤器分歧"

    return {
        "adx": float(adx_v) if pd.notna(adx_v) else None,
        "adx_label": strength,
        "agreement": float(agreement) if pd.notna(agreement) else None,
        "agreement_label": align,
        "ema200_dist_pct": float(dist) if pd.notna(dist) else None,
        "efficiency": float(eff) if pd.notna(eff) else None,
    }
