"""波动率引擎：衡量「风险的价格」处于什么水平。

权重 20%。四个因子：
  1. VIX          — 隐含波动率绝对水平
  2. VVIX         — 波动率的波动率（尾部风险定价）
  3. Realized Vol — SPY 21 日已实现波动率（年化）
  4. Vol Expansion— 短期/长期波动率比值，捕捉波动率扩张

输出 vol_score ∈ [0, 100]：数值越高代表波动环境越危险。
在最终打分中进入 (100 - vol_score) 项。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import features as F
from data import EQUITY, VIX, VIX3M, VVIX, MarketData

FACTOR_WEIGHTS = {
    "vix": 0.30,
    "vvix": 0.15,
    "realized_vol": 0.20,
    "vol_expansion": 0.15,
    "term_structure": 0.20,
}

RV_SHORT = 21
RV_LONG = 63

# VVIX 合理性下界：VVIX 为 VIX 期权隐含波动率指数，长期量级远高于 VIX
# （历史区间约 70~150，约为 VIX 的 4~6 倍）。低于该下界的序列不视为 VVIX，
# 直接剔除并重归一化权重，避免用错误量级的数据污染打分。
VVIX_PLAUSIBLE_MIN = 50.0

# 波动率分档（vol_score 阈值 → 标签）
VOL_BANDS = (
    (25.0, "低波动"),
    (50.0, "正常波动"),
    (75.0, "高波动"),
    (100.1, "极端波动"),
)


def vol_label(score: float) -> str:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "未知"
    for upper, name in VOL_BANDS:
        if score < upper:
            return name
    return "极端波动"


def compute(md: MarketData) -> pd.DataFrame:
    idx = md.prices.index

    # ---------- 1. VIX 绝对水平：12 以下极度平静，35 以上恐慌
    if VIX in md.prices.columns and md.prices[VIX].notna().any():
        vix = md.prices[VIX].reindex(idx).ffill()
        vix_score = F.piecewise(
            vix,
            [(10.0, 0.0), (13.0, 12.0), (16.0, 30.0), (20.0, 48.0),
             (25.0, 68.0), (30.0, 82.0), (40.0, 95.0), (55.0, 100.0)],
        )
        vix_rank = F.rolling_pct_rank(vix, 252, 120)
        vix_score = F.blend(vix_score, vix_rank, weights=[0.75, 0.25])
    else:
        vix = pd.Series(np.nan, index=idx)
        vix_score = pd.Series(np.nan, index=idx)

    # ---------- 2. VVIX：波动率的波动率
    # 数据合理性校验：量级不符则视为不可用（保证不误用身份不明的序列）
    vvix_ok = False
    if VVIX in md.prices.columns and md.prices[VVIX].notna().any():
        raw = md.prices[VVIX].reindex(idx).ffill()
        med = float(raw.dropna().median()) if raw.notna().any() else np.nan
        if pd.notna(med) and med >= VVIX_PLAUSIBLE_MIN:
            vvix_ok = True
            vvix = raw
        else:
            vvix = pd.Series(np.nan, index=idx)
    else:
        vvix = pd.Series(np.nan, index=idx)

    if vvix_ok:
        vvix_score = F.piecewise(
            vvix,
            [(70.0, 0.0), (80.0, 20.0), (90.0, 42.0), (100.0, 60.0),
             (115.0, 80.0), (130.0, 93.0), (150.0, 100.0)],
        )
    else:
        vvix_score = pd.Series(np.nan, index=idx)

    # ---------- 3. 已实现波动率（年化，%）
    price = md.series(EQUITY)
    ret = price.pct_change()
    rv = F.realized_vol(ret, RV_SHORT) * 100.0
    rv_score = F.piecewise(
        rv,
        [(6.0, 0.0), (9.0, 15.0), (12.0, 32.0), (16.0, 50.0),
         (22.0, 70.0), (30.0, 86.0), (45.0, 100.0)],
    )

    # ---------- 4. 波动率扩张：短期 / 长期
    rv_s = F.realized_vol(ret, 5, annualize=False)
    rv_l = F.realized_vol(ret, RV_LONG, annualize=False)
    expansion = F.safe_div(rv_s, rv_l)
    expansion_score = F.piecewise(
        expansion,
        [(0.5, 0.0), (0.75, 18.0), (1.0, 45.0), (1.3, 68.0),
         (1.8, 86.0), (2.6, 100.0)],
    )

    # ---------- 5. VIX 期限结构（VIX / VIX3M）
    # contango（近月 < 远月，比值 < 1）代表市场平静；backwardation（比值 > 1）
    # 是恐慌见顶的经典特征——近端波动率被抢高。
    if VIX3M in md.prices.columns and md.prices[VIX3M].notna().any() \
            and vix.notna().any():
        vix3m = md.prices[VIX3M].reindex(idx).ffill(limit=5)
        term = F.safe_div(vix, vix3m)
        term_score = F.piecewise(
            term,
            [(0.70, 0.0), (0.82, 22.0), (0.92, 42.0), (1.00, 56.0),
             (1.06, 70.0), (1.15, 87.0), (1.30, 100.0)],
        )
    else:
        term = pd.Series(np.nan, index=idx)
        term_score = pd.Series(np.nan, index=idx)

    factors = {
        "vix": vix_score,
        "vvix": vvix_score,
        "realized_vol": rv_score,
        "vol_expansion": expansion_score,
        "term_structure": term_score,
    }
    vol_score = F.weighted_mean(factors, FACTOR_WEIGHTS)

    out = pd.DataFrame(
        {
            "raw_vix": vix,
            "raw_vvix": vvix,
            "raw_realized_vol_pct": rv,
            "raw_vol_expansion": expansion,
            "raw_term_structure": term,
            "vix_score": vix_score,
            "vvix_score": vvix_score,
            "realized_vol_score": rv_score,
            "vol_expansion_score": expansion_score,
            "term_structure_score": term_score,
            "vol_score": vol_score,
        },
        index=idx,
    )
    return out


def describe(row: pd.Series) -> dict:
    vix = row.get("raw_vix", np.nan)
    vvix = row.get("raw_vvix", np.nan)
    rv = row.get("raw_realized_vol_pct", np.nan)
    exp = row.get("raw_vol_expansion", np.nan)
    score = row.get("vol_score", np.nan)

    if pd.isna(vix):
        vix_state = "未知"
    elif vix < 14:
        vix_state = "极低（自满）"
    elif vix < 19:
        vix_state = "低位"
    elif vix < 25:
        vix_state = "中性偏高"
    elif vix < 32:
        vix_state = "紧张"
    else:
        vix_state = "恐慌"

    if pd.isna(exp):
        exp_state = "未知"
    elif exp >= 1.5:
        exp_state = "显著扩张"
    elif exp >= 1.15:
        exp_state = "温和扩张"
    elif exp <= 0.85:
        exp_state = "收缩"
    else:
        exp_state = "平稳"

    return {
        "vol_score": float(score) if pd.notna(score) else None,
        "vol_band": vol_label(score),
        "vix": float(vix) if pd.notna(vix) else None,
        "vix_state": vix_state,
        "vvix": float(vvix) if pd.notna(vvix) else None,
        "realized_vol_pct": float(rv) if pd.notna(rv) else None,
        "vol_expansion": float(exp) if pd.notna(exp) else None,
        "vol_expansion_state": exp_state,
    }
