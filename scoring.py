"""打分层：把五个引擎的分数合成 Regime Score，映射档位，并生成解释层。

最终公式（机构版权重）::

    regime_score = 0.25 * trend_score
                 + 0.20 * breadth_score
                 + 0.15 * risk_on_score
                 + 0.20 * (100 - vol_score)
                 + 0.20 * (100 - crisis_score)

档位映射：
    80–100  Strong Bull
    60–80   Bull
    40–60   Neutral
    20–40   Bear
    0–20    Panic

解释层（比单一分数更重要）：
    Current Regime / Confidence / Drivers(+ 与 -) / 每个支柱的贡献度
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import breadth_engine
import cross_asset_engine
import risk_engine
import trend_engine
import volatility_engine
from data import EQUITY, MarketData

# ---------------------------------------------------------------- 权重
WEIGHTS = {
    "trend": 0.25,
    "breadth": 0.20,
    "risk_on": 0.15,
    "stability": 0.20,     # = 100 - vol_score
    "safety": 0.20,        # = 100 - crisis_score
}

PILLAR_LABELS = {
    "trend": "趋势 Trend",
    "breadth": "广度 Breadth",
    "risk_on": "跨资产 Risk-On",
    "stability": "低波动 Stability",
    "safety": "安全性 Safety",
}

REGIME_BANDS = (
    (80.0, 100.01, "Strong Bull"),
    (60.0, 80.0, "Bull"),
    (40.0, 60.0, "Neutral"),
    (20.0, 40.0, "Bear"),
    (-0.01, 20.0, "Panic"),
)

REGIME_CN = {
    "Strong Bull": "强势牛市",
    "Bull": "牛市",
    "Neutral": "中性震荡",
    "Bear": "熊市",
    "Panic": "恐慌",
}

# 档位配色（可视化统一使用）
REGIME_COLORS = {
    "Strong Bull": "#12b886",
    "Bull": "#40c057",
    "Neutral": "#adb5bd",
    "Bear": "#fd7e14",
    "Panic": "#e03131",
}


def map_regime(score: float) -> str:
    """分数 → 档位标签。"""
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "Unknown"
    for lo, hi, name in REGIME_BANDS:
        if lo <= score < hi:
            return name
    return "Unknown"


def regime_color(score: float) -> str:
    return REGIME_COLORS.get(map_regime(score), "#adb5bd")


# ---------------------------------------------------------------- 合成
def build_frame(md: MarketData) -> pd.DataFrame:
    """运行全部引擎，返回按日期对齐的完整因子与分数矩阵。"""
    idx = md.prices.index

    t = trend_engine.compute(md, ticker=EQUITY).reindex(idx)
    b = breadth_engine.compute(md).reindex(idx)
    v = volatility_engine.compute(md).reindex(idx)
    c = cross_asset_engine.compute(md).reindex(idx)
    r = risk_engine.compute(md).reindex(idx)

    df = pd.concat([t, b, v, c, r], axis=1)
    # 去掉重复列（各引擎索引相同，理论上无重复）
    df = df.loc[:, ~df.columns.duplicated()]

    # 支柱分
    df["pillar_trend"] = df["trend_score"]
    df["pillar_breadth"] = df["breadth_score"]
    df["pillar_risk_on"] = df["risk_on_score"]
    df["pillar_stability"] = 100.0 - df["vol_score"]
    df["pillar_safety"] = 100.0 - df["crisis_score"]

    # 最终打分（缺失支柱自动重归一化权重）
    pillars = {
        "trend": df["pillar_trend"],
        "breadth": df["pillar_breadth"],
        "risk_on": df["pillar_risk_on"],
        "stability": df["pillar_stability"],
        "safety": df["pillar_safety"],
    }
    df["regime_score"] = _weighted(pillars, WEIGHTS)
    df["regime"] = df["regime_score"].apply(map_regime)
    df["confidence"] = compute_confidence(df)
    return df


def _weighted(pillars: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    num = None
    den = None
    for k, s in pillars.items():
        w = weights[k]
        s = s.astype(float)
        if num is None:
            num = pd.Series(0.0, index=s.index)
            den = pd.Series(0.0, index=s.index)
        num = num + w * s.fillna(0.0)
        den = den + w * s.notna().astype(float)
    return (num / den.where(den > 0)).clip(0.0, 100.0)


def compute_confidence(df: pd.DataFrame) -> pd.Series:
    """置信度：模型内部一致性的度量，不是「预测准确率」。

    三个成分：
      1. 共识度 consensus  — 五个支柱彼此有多接近（分歧大 → 低置信）
      2. 边界距离 boundary — 离最近的档位分界有多远（贴边 → 低置信）
      3. 数据覆盖 coverage — 有效支柱占比
    """
    cols = ["pillar_trend", "pillar_breadth", "pillar_risk_on",
            "pillar_stability", "pillar_safety"]
    sub = df[cols]
    n_valid = sub.notna().sum(axis=1)

    std = sub.std(axis=1, ddof=0).where(n_valid >= 3)
    # 标准差 0 → 100；标准差 12 → 73；标准差 25 → 45；标准差 40 → 12
    consensus = (100.0 - std * 2.2).clip(0.0, 100.0)

    edges = np.array([20.0, 40.0, 60.0, 80.0])
    score = df["regime_score"].astype(float).values
    d = np.min(np.abs(score[:, None] - edges[None, :]), axis=1)
    # 距边界 8pp 及以上视为完全不模糊
    boundary = pd.Series(np.clip(d / 8.0 * 100.0, 0.0, 100.0), index=df.index)

    coverage = (n_valid / len(cols) * 100.0)

    conf = 0.50 * consensus.fillna(0.0) + 0.30 * boundary + 0.20 * coverage
    # 数据不足时强制压低置信度
    conf = conf.where(n_valid >= 4, conf * 0.6)
    return conf.clip(0.0, 100.0)


# ---------------------------------------------------------------- 解释层
def pillar_contributions(row: pd.Series) -> list[dict]:
    """把每个支柱的「相对中性的偏离」乘以权重，得到对总分的贡献（pp）。"""
    out = []
    for key, w in WEIGHTS.items():
        col = f"pillar_{key}"
        val = row.get(col, np.nan)
        if pd.isna(val):
            continue
        delta = float(val) - 50.0
        out.append(
            {
                "key": key,
                "label": PILLAR_LABELS[key],
                "score": float(val),
                "weight": w,
                "contribution": delta * w,       # 相对中性 ±50 的加权偏离
            }
        )
    return out


def explain(row: pd.Series, n_universe: int = 8) -> dict:
    """生成解释层字典：档位、置信度、正负驱动因子、各支柱读数。"""
    score = row.get("regime_score", np.nan)
    regime = map_regime(score)
    conf = row.get("confidence", np.nan)

    t = trend_engine.describe(row)
    b = breadth_engine.describe(row, n_universe)
    v = volatility_engine.describe(row)
    c = cross_asset_engine.describe(row)
    r = risk_engine.describe(row)

    contribs = pillar_contributions(row)

    # ---------- 正向 / 负向驱动
    positives: list[str] = []
    negatives: list[str] = []

    # 趋势
    ts = row.get("trend_score", np.nan)
    if pd.notna(ts):
        if ts >= 70:
            positives.append(f"趋势强劲（{ts:.0f}/100，{t['adx_label']}）")
        elif ts <= 30:
            negatives.append(f"趋势偏弱（{ts:.0f}/100，{t['adx_label']}）")

    # 广度
    bs = row.get("breadth_score", np.nan)
    if pd.notna(bs):
        if bs >= 65:
            positives.append(f"广度健康（{bs:.0f}/100，{b['breadth_state']}）")
        elif bs <= 35:
            negatives.append(f"广度恶化（{bs:.0f}/100，{b['breadth_state']}）")

    # 跨资产 / 信用
    rs = row.get("risk_on_score", np.nan)
    if pd.notna(rs):
        if rs >= 60:
            positives.append(f"跨资产 Risk-On（{rs:.0f}/100，{c['credit_label']}）")
        elif rs <= 40:
            negatives.append(f"跨资产 Risk-Off（{rs:.0f}/100，{c['credit_label']}）")

    # 波动率（注意方向：vol_score 高是坏消息）
    vs = row.get("vol_score", np.nan)
    if pd.notna(vs):
        if vs >= 60:
            negatives.append(f"波动率抬升（{vs:.0f}/100，VIX {v['vix']:.1f}）"
                             if v.get("vix") else f"波动率抬升（{vs:.0f}/100）")
        elif vs <= 40:
            positives.append(f"波动率受控（{vs:.0f}/100，{v['vol_band']}）")

    # 危机特征
    cs = row.get("crisis_score", np.nan)
    if pd.notna(cs):
        if cs >= 55:
            negatives.append(f"尾部风险累积（{cs:.0f}/100，{r['skew_label']}）")
        elif cs <= 35:
            positives.append(f"尾部风险低（{cs:.0f}/100，{r['corr_label']}）")

    contribs_sorted = sorted(contribs, key=lambda x: x["contribution"], reverse=True)

    # ---------- 补充：阈值驱动不足时，用相对贡献补齐（避免只输出一条驱动）
    mentioned_pos = " ".join(positives)
    mentioned_neg = " ".join(negatives)
    for pc in contribs_sorted:
        lbl = pc["label"].split()[0]
        if len(positives) < 2 and pc["contribution"] >= 1.0 and lbl not in mentioned_pos:
            positives.append(
                f"{lbl} 相对占优（{pc['score']:.0f}/100，贡献 {pc['contribution']:+.1f}pp）"
            )
            mentioned_pos += lbl
        elif len(negatives) < 2 and pc["contribution"] <= -1.0 and lbl not in mentioned_neg:
            negatives.append(
                f"{lbl} 相对拖累（{pc['score']:.0f}/100，贡献 {pc['contribution']:+.1f}pp）"
            )
            mentioned_neg += lbl

    return {
        "asof": str(row.name.date()) if hasattr(row.name, "date") else str(row.name),
        "regime_score": float(score) if pd.notna(score) else None,
        "regime": regime,
        "regime_cn": REGIME_CN.get(regime, "未知"),
        "confidence": float(conf) if pd.notna(conf) else None,
        "drivers_positive": positives[:4],
        "drivers_negative": negatives[:4],
        "contributions": contribs_sorted,
        "detail": {"trend": t, "breadth": b, "volatility": v,
                   "cross_asset": c, "risk": r},
    }


def format_report(res: dict) -> str:
    """把解释层渲染成机构风格的文本报告。"""
    L = []
    L.append("=" * 58)
    L.append(f"  MARKET REGIME MONITOR   ·   {res['asof']}")
    L.append("=" * 58)
    L.append("")
    L.append(f"  Current Regime :  {res['regime'].upper()}  ({res['regime_cn']})")
    L.append(f"  Regime Score   :  {res['regime_score']:.1f} / 100")
    L.append(f"  Confidence     :  {res['confidence']:.0f}%")
    L.append("")
    L.append("  Drivers")
    L.append("  " + "-" * 54)
    for p in res["drivers_positive"]:
        L.append(f"    + {p}")
    for n in res["drivers_negative"]:
        L.append(f"    - {n}")
    if not res["drivers_positive"] and not res["drivers_negative"]:
        L.append("    （各维度均处中性区间，无显著驱动）")
    L.append("")
    L.append("  Pillar Breakdown")
    L.append("  " + "-" * 54)
    L.append(f"    {'支柱':<18}{'得分':>8}{'权重':>8}{'贡献':>9}")
    for pc in res["contributions"]:
        L.append(
            f"    {pc['label']:<18}{pc['score']:>8.1f}"
            f"{pc['weight'] * 100:>7.0f}%{pc['contribution']:>+9.1f}"
        )
    L.append("")

    d = res["detail"]
    L.append("  Detail")
    L.append("  " + "-" * 54)
    t, b, v, c, r = d["trend"], d["breadth"], d["volatility"], d["cross_asset"], d["risk"]
    L.append(f"    趋势    ADX={_fmt(t['adx'])} 距EMA200={_pct(t['ema200_dist_pct'])} "
             f"效率={_fmt(t['efficiency'], 2)}")
    L.append(f"    广度    站上MA200 {b['above_ma200_count']}  "
             f"上涨占比={_pct((b['advance_ratio'] or 0) * 100, already=True)}")
    L.append(f"    波动    VIX={_fmt(v['vix'])}  VVIX={_fmt(v['vvix'])}  "
             f"RV={_pct(v['realized_vol_pct'], already=True)}"
             f"  扩张={_fmt(v['vol_expansion'], 2)}x")
    L.append(f"    跨资产  SPY/TLT={_pct(c['spy_tlt_chg63_pct'])}  "
             f"HYG/TLT={_pct(c['credit_chg63_pct'])}  DXY={_pct(c['dxy_chg63_pct'])}")
    L.append(f"    风险    偏度={_fmt(r['skew'], 2)}  回撤={_pct(r['drawdown_pct'], already=True)}  "
             f"相关={_fmt(r['avg_corr'], 2)}")
    L.append("")
    return "\n".join(L)


def _fmt(v, nd=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def _pct(v, nd=1, already=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:+.{nd}f}%" if not already else f"{v:.{nd}f}%"
