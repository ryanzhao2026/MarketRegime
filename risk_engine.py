"""风险引擎：识别「尾部风险正在累积」的迹象，与波动率引擎互补。

权重 20%。波动率看的是「风险的价格」，风险引擎看的是「风险的形状」。

五个因子：
  1. Skew             — SPY 63 日收益偏度，左偏代表暴跌风险累积
  2. Kurtosis         — 63 日超额峰度，尖峰厚尾代表极端事件概率上升
  3. Drawdown         — 相对 252 日高点的回撤深度
  4. Correlation Regime — 风险资产两两相关性均值，趋近 1 意味着「无处可躲」
  5. Sector Dispersion  — 行业间收益离散度，过高代表轮动失序/主线不明

输出 crisis_score ∈ [0, 100]：越高代表危机特征越明显。
在最终打分中进入 (100 - crisis_score) 项。

说明：Sector Dispersion 按「分化加剧 → 风险上升」映射。这一方向选择是
有意的建模假设（轮动失序 = 领导力不稳定），并非唯一解读，见 README。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import features as F
from data import (CROSS_ASSET_UNIVERSE, EQUITY, HIGH_YIELD, NFCI, RRP,
                  SECTORS, SMALL_CAP, TECH, EQUAL_WEIGHT, WALCL, MarketData)

FACTOR_WEIGHTS = {
    "skew": 0.20,
    "kurtosis": 0.12,
    "drawdown": 0.22,
    "correlation": 0.18,
    "dispersion": 0.13,
    # v2 新增：流动性条件（原五因子按比例让渡 15% 权重，总量仍为 1.0）
    "liquidity": 0.15,
}

SKEW_LEN = 63
KURT_LEN = 63
DD_LOOKBACK = 252
CORR_LEN = 60
DISP_LEN = 21

# 相关性计算宇宙：股票类 + 信用 + 行业（不含债券/黄金/商品等天然负相关资产）
CORRELATION_UNIVERSE = (EQUITY, EQUAL_WEIGHT, TECH, SMALL_CAP, HIGH_YIELD) + SECTORS


def compute(md: MarketData) -> pd.DataFrame:
    idx = md.prices.index
    px = md.prices.ffill()

    spy = px[EQUITY] if EQUITY in px.columns else None
    if spy is None or spy.notna().sum() < 30:
        raise ValueError("风险引擎需要 SPY 数据")

    ret = spy.pct_change()

    # ---------- 1. 偏度（负偏 = 危机风险）
    skew = F.rolling_skew(ret, SKEW_LEN)
    # 偏度 -2.0 → 100 分（极左偏），+1.0 → 0 分
    skew_score = F.piecewise(
        skew,
        [(-2.0, 100.0), (-1.0, 82.0), (-0.5, 66.0), (0.0, 50.0),
         (0.5, 32.0), (1.5, 10.0), (2.5, 0.0)],
    )

    # ---------- 2. 峰度（厚尾 = 危机风险）
    kurt = F.rolling_kurt(ret, KURT_LEN)
    kurt_score = F.piecewise(
        kurt,
        [(-1.0, 0.0), (0.0, 15.0), (1.0, 32.0), (2.0, 50.0),
         (4.0, 72.0), (7.0, 90.0), (12.0, 100.0)],
    )

    # ---------- 3. 回撤（深度越深分越高）
    dd = F.drawdown_from_high(spy, DD_LOOKBACK)
    dd_score = F.piecewise(
        dd,
        [(-0.40, 100.0), (-0.25, 92.0), (-0.15, 80.0), (-0.10, 68.0),
         (-0.05, 52.0), (-0.02, 35.0), (0.0, 18.0)],
    )

    # ---------- 4. 相关性状态
    corr_cols = md.available(list(CORRELATION_UNIVERSE))
    if len(corr_cols) >= 3:
        corr_df = px[corr_cols].pct_change()
        avg_corr = F.avg_pairwise_corr(corr_df, CORR_LEN, 30)
    else:
        avg_corr = pd.Series(np.nan, index=idx)
    corr_score = F.piecewise(
        avg_corr,
        [(0.10, 0.0), (0.25, 20.0), (0.40, 42.0), (0.55, 60.0),
         (0.70, 80.0), (0.85, 95.0), (0.95, 100.0)],
    )

    # ---------- 5. 行业离散度（21 日收益的截面标准差）
    sec_cols = [c for c in SECTORS if c in px.columns and px[c].notna().any()]
    if len(sec_cols) >= 2:
        sec_ret = px[sec_cols].pct_change()
        turnover = sec_ret.rolling(DISP_LEN, min_periods=5).mean() * 100.0   # 月度收益 %
        dispersion = turnover.std(axis=1, ddof=1)
        disp_rank = F.rolling_pct_rank(dispersion, 252, 120)
        disp_abs = F.piecewise(
            dispersion,
            [(0.5, 0.0), (1.5, 20.0), (2.5, 42.0), (4.0, 62.0),
             (6.0, 82.0), (9.0, 100.0)],
        )
        disp_score = F.blend(disp_abs, disp_rank, weights=[0.6, 0.4])
    else:
        dispersion = pd.Series(np.nan, index=idx)
        disp_score = pd.Series(np.nan, index=idx)

    # ---------- 6. 流动性条件（v2 新增）
    # NFCI 是芝加哥联储的复合金融条件指数：0 为长期均值，正=偏紧。
    # 辅以美联储资产负债表（WALCL）与隔夜逆回购（RRP）判断「钱在不在系统里」。
    # 三者都是周频，缺失时整项为 NaN，权重自动重归一化。
    liquidity_score = pd.Series(np.nan, index=idx)
    nfci_now = pd.Series(np.nan, index=idx)
    nfci_chg = pd.Series(np.nan, index=idx)
    if NFCI in px.columns and px[NFCI].notna().sum() >= 20:
        nfci_now = px[NFCI].reindex(idx).ffill(limit=15)
        # 绝对水平：NFCI 长期均值 0，>0.3 明显偏紧，>1.0 为危机级
        lvl = F.piecewise(
            nfci_now,
            [(-1.20, 0.0), (-0.70, 15.0), (-0.50, 28.0), (-0.30, 45.0),
             (0.00, 62.0), (0.30, 78.0), (1.00, 92.0), (2.50, 100.0)],
        )
        # 变化方向：13 周（约一季度）加速收紧比绝对水平更早预警
        nfci_chg = nfci_now.diff(13)
        chg = F.piecewise(
            nfci_chg,
            [(-0.60, 0.0), (-0.25, 22.0), (0.0, 48.0), (0.25, 68.0),
             (0.60, 85.0), (1.50, 100.0)],
        )
        parts = {"nfci_level": lvl, "nfci_chg": chg}
        wts = {"nfci_level": 0.60, "nfci_chg": 0.40}

        # 资产负债表与逆回购（可选增强）：扩表/逆回购回落 = 流动性宽松
        if WALCL in px.columns and px[WALCL].notna().sum() >= 15:
            walcl = px[WALCL].reindex(idx).ffill(limit=15)
            wal_chg = (walcl / walcl.shift(13) - 1.0) * 100.0   # 13 周变化 %
            parts["walcl"] = F.piecewise(
                wal_chg,
                [(-4.0, 0.0), (-1.5, 20.0), (0.0, 45.0), (1.5, 62.0),
                 (4.0, 82.0), (8.0, 100.0)],
            )
            wts["walcl"] = 0.25
        if RRP in px.columns and px[RRP].notna().sum() >= 15:
            rrp = px[RRP].reindex(idx).ffill(limit=15)
            rrp_chg = (rrp / rrp.shift(13) - 1.0) * 100.0
            # 逆回购下降 = 资金回流体系（偏宽松），故反向计分
            parts["rrp"] = 100.0 - F.piecewise(
                rrp_chg,
                [(-60.0, 0.0), (-25.0, 25.0), (0.0, 50.0), (25.0, 72.0),
                 (60.0, 90.0), (150.0, 100.0)],
            )
            wts["rrp"] = 0.15
        tot = sum(wts.values())
        liquidity_score = F.weighted_mean(parts, {k: v / tot for k, v in wts.items()})

    factors = {
        "skew": skew_score,
        "kurtosis": kurt_score,
        "drawdown": dd_score,
        "correlation": corr_score,
        "dispersion": disp_score,
        "liquidity": liquidity_score,
    }
    crisis_score = F.weighted_mean(factors, FACTOR_WEIGHTS)

    out = pd.DataFrame(
        {
            "raw_skew": skew,
            "raw_kurtosis": kurt,
            "raw_drawdown": dd,
            "raw_avg_corr": avg_corr,
            "raw_dispersion": dispersion,
            "raw_nfci": nfci_now,
            "raw_nfci_chg13": nfci_chg,
            "liquidity_score": liquidity_score,
            "skew_score": skew_score,
            "kurtosis_score": kurt_score,
            "drawdown_score": dd_score,
            "correlation_score": corr_score,
            "dispersion_score": disp_score,
            "crisis_score": crisis_score,
        },
        index=idx,
    )
    return out


def describe(row: pd.Series) -> dict:
    skew = row.get("raw_skew", np.nan)
    kurt = row.get("raw_kurtosis", np.nan)
    dd = row.get("raw_drawdown", np.nan)
    corr = row.get("raw_avg_corr", np.nan)
    disp = row.get("raw_dispersion", np.nan)
    score = row.get("crisis_score", np.nan)

    if pd.isna(skew):
        skew_label = "未知"
    elif skew <= -0.8:
        skew_label = "明显左偏(下跌尾部厚)"
    elif skew <= -0.3:
        skew_label = "轻度左偏"
    else:
        skew_label = "偏度正常"

    if pd.isna(corr):
        corr_label = "未知"
    elif corr >= 0.65:
        corr_label = "高度同涨同跌"
    elif corr >= 0.45:
        corr_label = "相关性偏高"
    else:
        corr_label = "相关性分散"

    return {
        "crisis_score": float(score) if pd.notna(score) else None,
        "skew": float(skew) if pd.notna(skew) else None,
        "skew_label": skew_label,
        "kurtosis": float(kurt) if pd.notna(kurt) else None,
        "drawdown_pct": float(dd * 100.0) if pd.notna(dd) else None,
        "avg_corr": float(corr) if pd.notna(corr) else None,
        "corr_label": corr_label,
        "dispersion": float(disp) if pd.notna(disp) else None,
    }
