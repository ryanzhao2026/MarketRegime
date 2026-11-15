"""广度引擎：指数上涨是否由「多数股票」参与，还是少数权重股独撑。

权重 20%。这是最能区分「健康牛市」与「脆弱牛市」的维度。

四个因子：
  1. SPY > MA200 %       — 广度宇宙中价格站上 MA200 的比例
  2. Advance / Decline    — 日均上涨家数占比（20 日平滑）
  3. RSP / SPY Ratio      — 等权 vs 市值加权，衡量「宽度溢价」
  4. New Highs - New Lows — 创 252 日新高家数减创新低家数

退化说明：iFinD 未取到标普 500 全成分股日行情时，广度宇宙退化为
「4 只宽基 ETF + 4 只行业 ETF」共 8 个标的的截面统计。
这是**代理指标**，方向信号有效但绝对数值不如全成分股精确，已在输出中标注。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import features as F
from data import BREADTH_UNIVERSE, EQUAL_WEIGHT, EQUITY, MarketData

FACTOR_WEIGHTS = {
    "above_ma200": 0.30,
    "advance_decline": 0.25,
    "rsp_spy": 0.25,
    "net_new_high_low": 0.20,
}

MA_LEN = 200
AD_SMOOTH = 20
NHNL_LOOKBACK = 252


def universe_frame(md: MarketData) -> pd.DataFrame:
    """广度宇宙价格宽表（仅保留有数据的列）。"""
    cols = md.available(list(BREADTH_UNIVERSE))
    if not cols:
        raise ValueError("广度宇宙无可用数据")
    return md.prices[cols].ffill()


def compute(md: MarketData) -> pd.DataFrame:
    px = universe_frame(md)
    n_assets = px.shape[1]

    # ---------- 1. 站上 MA200 的比例（0~1 → 0~100）
    above = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    for c in px.columns:
        ma = F.sma(px[c], MA_LEN)
        above[c] = np.where(ma.notna(), (px[c] > ma).astype(float), np.nan)
    valid_cnt = above.notna().sum(axis=1)
    pct_above = (above.sum(axis=1, skipna=True) / valid_cnt.replace(0, np.nan))
    # 样本不足 MA200（前 100 天）时不给分，避免假信号
    pct_above = pct_above.where(valid_cnt >= max(3, n_assets - 2))
    above_ma200_score = (pct_above * 100.0).clip(0.0, 100.0)

    # ---------- 2. 上涨家数占比（20 日平滑）
    ret1 = px.pct_change()
    adv = (ret1 > 0).astype(float).where(ret1.notna())
    adv_ratio = adv.sum(axis=1) / ret1.notna().sum(axis=1).replace(0, np.nan)
    ad_ratio = adv_ratio.rolling(AD_SMOOTH, min_periods=5).mean()
    advance_decline_score = (ad_ratio * 100.0).clip(0.0, 100.0)

    # ---------- 3. RSP/SPY：等权相对强弱
    if EQUAL_WEIGHT in px.columns and EQUITY in px.columns:
        ratio = F.safe_div(px[EQUAL_WEIGHT], px[EQUITY])
        chg63 = (ratio / ratio.shift(63) - 1.0) * 100.0          # 季内相对涨跌幅 %
        ma50 = F.sma(ratio, 50)
        above_ma = np.where(ma50.notna(), (ratio > ma50).astype(float), np.nan)
        above_ma = pd.Series(above_ma, index=ratio.index)
        base = F.piecewise(chg63, [(-6.0, 0.0), (-2.0, 20.0), (0.0, 50.0),
                                   (2.0, 78.0), (5.0, 95.0), (9.0, 100.0)])
        trend_bonus = above_ma * 100.0
        rsp_spy_score = F.blend(base, trend_bonus, weights=[0.7, 0.3])
    else:
        ratio = pd.Series(np.nan, index=px.index)
        chg63 = pd.Series(np.nan, index=px.index)
        rsp_spy_score = pd.Series(np.nan, index=px.index)

    # ---------- 4. 净新高：创 252 日新高家数 - 创新低家数
    hi = px >= px.rolling(NHNL_LOOKBACK, min_periods=60).max()
    lo = px <= px.rolling(NHNL_LOOKBACK, min_periods=60).min()
    maturity = px.rolling(NHNL_LOOKBACK, min_periods=60).count() >= 60
    hi = hi.where(maturity)
    lo = lo.where(maturity)
    net = hi.sum(axis=1) - lo.sum(axis=1)
    net_norm = net / max(1, n_assets)                            # -1 ~ +1
    # 5 日平滑，抑制单日噪声
    net_norm = net_norm.rolling(5, min_periods=1).mean()
    nhnl_score = (50.0 + net_norm * 50.0).clip(0.0, 100.0)

    factors = {
        "above_ma200": above_ma200_score,
        "advance_decline": advance_decline_score,
        "rsp_spy": rsp_spy_score,
        "net_new_high_low": nhnl_score,
    }
    breadth_score = F.weighted_mean(factors, FACTOR_WEIGHTS)

    out = pd.DataFrame(
        {
            "raw_pct_above_ma200": pct_above,
            "raw_advance_ratio": ad_ratio,
            "raw_rsp_spy_ratio": ratio,
            "raw_rsp_spy_chg63": chg63,
            "raw_net_new_high_low": net_norm,
            "above_ma200_score": above_ma200_score,
            "advance_decline_score": advance_decline_score,
            "rsp_spy_score": rsp_spy_score,
            "net_new_high_low_score": nhnl_score,
            "breadth_score": breadth_score,
        },
        index=px.index,
    )
    return out


def describe(row: pd.Series, n_universe: int = 8) -> dict:
    pct = row.get("raw_pct_above_ma200", np.nan)
    adv = row.get("raw_advance_ratio", np.nan)
    chg = row.get("raw_rsp_spy_chg63", np.nan)
    net = row.get("raw_net_new_high_low", np.nan)
    score = row.get("breadth_score", np.nan)

    if pd.isna(pct):
        state = "未知"
    elif pct >= 0.75:
        state = "广度强劲"
    elif pct >= 0.5:
        state = "广度中性偏强"
    elif pct >= 0.25:
        state = "广度走弱"
    else:
        state = "广度恶化"

    return {
        "breadth_score": float(score) if pd.notna(score) else None,
        "breadth_state": state,
        "pct_above_ma200": float(pct) if pd.notna(pct) else None,
        "above_ma200_count": (f"{int(round(pct * n_universe))}/{n_universe}"
                              if pd.notna(pct) else None),
        "advance_ratio": float(adv) if pd.notna(adv) else None,
        "rsp_spy_chg63_pct": float(chg) if pd.notna(chg) else None,
        "net_new_high_low": float(net) if pd.notna(net) else None,
        "universe_size": n_universe,
    }
