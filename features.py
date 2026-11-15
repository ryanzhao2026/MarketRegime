"""特征工程：指标计算、滚动归一化与稳健打分工具。

设计原则
--------
1. 所有函数都是「纯函数 + 时间序列输入 → 时间序列输出」，不做未来数据引用。
2. 每个因子先算原始量，再映射到 0~100 的分位尺度；
   绝对刻度（piecewise）保证可解释，滚动分位（pct_rank）保证自适应性，
   两者按比例混合以兼顾「有意义」与「可比」。
3. 缺失一律保留 NaN，由上层加权平均自动重新分配权重，不做前向填充造假。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------- 基础工具
def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """逐元素安全除法：分母为 0 时返回 NaN，避免 inf 污染打分。"""
    return num / den.replace(0.0, np.nan)


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=max(2, n // 2)).mean()


def rolling_slope(s: pd.Series, n: int = 20, normalize: bool = True) -> pd.Series:
    """滚动 OLS 斜率；normalize=True 时除以价格水平，得到「每根 K 线的百分比斜率」。"""
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    denom = float((x ** 2).sum())

    def _slope(win: np.ndarray) -> float:
        if np.isnan(win).any():
            return np.nan
        y = win - win.mean()
        return float((x * y).sum() / denom)

    slope = s.rolling(n, min_periods=n).apply(_slope, raw=True)
    return safe_div(slope, s) * 100.0 if normalize else slope


def realized_vol(returns: pd.Series, n: int = 21, annualize: bool = True) -> pd.Series:
    """滚动已实现波动率（收益标准差，年化）。"""
    v = returns.rolling(n, min_periods=max(5, n // 2)).std(ddof=1)
    return v * np.sqrt(TRADING_DAYS) if annualize else v


def drawdown_from_high(price: pd.Series, n: int = 252) -> pd.Series:
    """相对滚动 n 日高点的回撤，返回负值（-0.15 表示回撤 15%）。"""
    peak = price.rolling(n, min_periods=20).max()
    return safe_div(price, peak) - 1.0


def rolling_skew(r: pd.Series, n: int = 63) -> pd.Series:
    return r.rolling(n, min_periods=max(20, n // 2)).skew()


def rolling_kurt(r: pd.Series, n: int = 63) -> pd.Series:
    return r.rolling(n, min_periods=max(20, n // 2)).kurt()


def trend_efficiency(price: pd.Series, n: int = 20) -> pd.Series:
    """Kaufman 效率系数：净位移 / 路径总长，取值 0~1。"""
    net = (price - price.shift(n)).abs()
    path = price.diff().abs().rolling(n, min_periods=n).sum()
    return safe_div(net, path)


def avg_pairwise_corr(df: pd.DataFrame, n: int = 60, min_periods: int = 30) -> pd.Series:
    """滚动窗口内所有资产两两相关系数的均值（衡量「宏观因子主导度」）。"""
    cols = list(df.columns)
    if len(cols) < 2:
        return pd.Series(np.nan, index=df.index)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append(df[cols[i]].rolling(n, min_periods=min_periods).corr(df[cols[j]]))
    return pd.concat(pairs, axis=1).mean(axis=1)


# ---------------------------------------------------------------- 方向性指标
def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder ADX，返回 (adx, +DI, -DI)。ADX 只衡量趋势强度，不含方向。"""
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr_ = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    pdi = 100.0 * safe_div(plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean(), atr_)
    mdi = 100.0 * safe_div(minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean(), atr_)
    dx = 100.0 * safe_div((pdi - mdi).abs(), (pdi + mdi))
    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean(), pdi, mdi


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """返回 (macd 线, 信号线, 柱)。"""
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


# ---------------------------------------------------------------- 归一化
def rolling_pct_rank(series: pd.Series, window: int = 252, min_periods: int = 120) -> pd.Series:
    """滚动分位（0~100）：当前值在最近 window 个观测中的百分位。"""

    def _rank(arr: np.ndarray) -> float:
        last = arr[-1]
        valid = arr[~np.isnan(arr)]
        if np.isnan(last) or valid.size == 0:
            return np.nan
        return float((valid <= last).mean() * 100.0)

    return series.rolling(window, min_periods=min_periods).apply(_rank, raw=True)


def piecewise(values: pd.Series | np.ndarray, points: list[tuple[float, float]]) -> pd.Series:
    """分段线性映射：(x0,y0),(x1,y1)... 超出范围截断到端点，NaN 保持 NaN。"""
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    is_series = isinstance(values, pd.Series)
    arr = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        out = np.interp(arr, xs, ys)
    out = np.where(np.isnan(arr), np.nan, out)
    return pd.Series(out, index=values.index) if is_series else out


def blend(*score_series: pd.Series, weights: list[float] | None = None) -> pd.Series:
    """混合多个 0~100 分数（默认等权，自动忽略 NaN）。"""
    if not score_series:
        raise ValueError("blend 需要至少一个序列")
    w = weights if weights is not None else [1.0] * len(score_series)
    num = pd.Series(0.0, index=score_series[0].index)
    den = pd.Series(0.0, index=score_series[0].index)
    for s, wi in zip(score_series, w):
        s = s.astype(float)
        valid = s.notna()
        num = num + wi * s.fillna(0.0)
        den = den + wi * valid.astype(float)
    return (num / den.where(den > 0)).clip(0.0, 100.0)


def weighted_mean(factors: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """按权重合成 0~100 分数；缺失因子自动从分母剔除（权重重归一化）。"""
    num: pd.Series | None = None
    den: pd.Series | None = None
    for name, series in factors.items():
        w = float(weights.get(name, 0.0))
        if w <= 0 or series is None:
            continue
        s = series.astype(float)
        if num is None:
            num = pd.Series(0.0, index=s.index)
            den = pd.Series(0.0, index=s.index)
        num = num + w * s.fillna(0.0)
        den = den + w * s.notna().astype(float)
    if num is None or den is None:
        raise ValueError("weighted_mean 没有有效的因子输入")
    return (num / den.where(den > 0)).clip(0.0, 100.0)


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """按分位裁剪，抑制极端值对打分的支配。"""
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def zscore(s: pd.Series, window: int = 252, min_periods: int = 120) -> pd.Series:
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=1)
    return safe_div(s - mu, sd)


def signed_strength(value: pd.Series, strength: pd.Series, neutral: float = 50.0) -> pd.Series:
    """把「强度 0~1」与「方向 ±1」合成为 0~100 的带方向分数。"""
    return (neutral + (value * 1.0) * (strength * 100.0 - neutral)).clip(0.0, 100.0)


def last_valid(s: pd.Series, default: float = np.nan) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else default
