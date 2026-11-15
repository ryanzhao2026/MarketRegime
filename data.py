"""数据层：加载并对齐行情数据，定义标的角色与因子宇宙。

数据来源：同花顺 iFinD 历史日行情（收盘价，未复权用于价格序列展示；
ETF 分红对 MA200 等长周期指标影响有限，已在 README 中说明）。
频率：日频；样本区间 2024-01-02 至 2026-11-13。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_PRICE_PATH = os.path.join(DATA_DIR, "market_data.csv")
DEFAULT_OHLC_PATH = os.path.join(DATA_DIR, "spy_ohlc.csv")

TRADING_DAYS = 252

# ---------------------------------------------------------------- 标的角色
EQUITY = "SPY"            # 标普 500 市值加权
EQUAL_WEIGHT = "RSP"      # 标普 500 等权（广度代理）
TECH = "QQQ"              # 纳指 100
SMALL_CAP = "IWM"         # 罗素 2000
LONG_BOND = "TLT"         # 20 年期美债
HIGH_YIELD = "HYG"        # 高收益信用债
GOLD = "GLD"              # 黄金
COMMODITY = "DBC"         # 大宗商品
DOLLAR = "DXY"            # 美元指数（CI 环境用 UUP 代理，见 fetch_data.py）
DOLLAR_CANDIDATES = ("DXY", "DINI", "UUP")   # 兼容不同数据源列名
VIX = "VIX"               # 隐含波动率
VVIX = "VVIX"             # 波动率的波动率
VIX3M = "VIX3M"           # 3 个月 VIX（期限结构用，当前未进入打分）

SECTORS: tuple[str, ...] = ("XLK", "XLF", "XLE", "XLV")

# 广度宇宙：无法取得全成分股时退化为「核心宽基 + 行业代理」的固定样本
BREADTH_UNIVERSE: tuple[str, ...] = ("SPY", "RSP", "QQQ", "IWM") + SECTORS

# 跨资产与风险引擎使用的可交易风险资产（不含 VIX/VVIX 等指数）
CROSS_ASSET_UNIVERSE: tuple[str, ...] = (
    EQUITY, EQUAL_WEIGHT, TECH, SMALL_CAP, LONG_BOND, HIGH_YIELD, GOLD, COMMODITY,
)

# 仪表盘展示顺序与中文名
DISPLAY_NAMES = {
    "SPY": "标普500(SPY)", "RSP": "标普等权(RSP)", "QQQ": "纳指100(QQQ)",
    "IWM": "罗素2000(IWM)", "TLT": "长债(TLT)", "HYG": "高收益债(HYG)",
    "GLD": "黄金(GLD)", "DBC": "商品(DBC)", "DINI": "美元指数(DXY)",
    "VIX": "VIX", "VVIX": "VVIX", "XLK": "科技(XLK)", "XLF": "金融(XLF)",
    "XLE": "能源(XLE)", "XLV": "医疗(XLV)",
}


class DataError(RuntimeError):
    """数据缺失或格式错误。"""


@dataclass
class MarketData:
    """行情容器。

    prices : 收盘价宽表（index=交易日）
    ohlc   : SPY 的 OHLC 宽表，可能为空（此时 ADX 退化为代理指标）
    """

    prices: pd.DataFrame
    ohlc: pd.DataFrame | None = None

    # -------------------------------------------------- 便捷访问
    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.prices.index

    @property
    def last_date(self) -> pd.Timestamp:
        return self.prices.index[-1]

    def series(self, ticker: str) -> pd.Series:
        if ticker not in self.prices.columns:
            raise DataError(f"缺少标的 {ticker}")
        return self.prices[ticker].dropna()

    def available(self, universe: Sequence[str]) -> list[str]:
        return [t for t in universe if t in self.prices.columns and self.prices[t].notna().sum() > 0]

    # -------------------------------------------------- 回报率
    @property
    def returns(self) -> pd.DataFrame:
        return self.prices.pct_change()

    def has_ohlc(self) -> bool:
        return self.ohlc is not None and not self.ohlc.empty

    def slice_to(self, asof: str | pd.Timestamp | None) -> "MarketData":
        """截断到指定日期（含），用于无前视的时点复算。"""
        if asof is None:
            return self
        ts = pd.Timestamp(asof)
        prices = self.prices.loc[:ts]
        ohlc = None if not self.has_ohlc() else self.ohlc.loc[:ts]
        if prices.empty:
            raise DataError(f"{ts.date()} 之前没有可用行情")
        return MarketData(prices=prices, ohlc=ohlc)


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise DataError(f"找不到数据文件: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise DataError(f"{path} 缺少 date 列")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.apply(pd.to_numeric, errors="coerce")


def load_market_data(
    price_path: str | None = None,
    ohlc_path: str | None = None,
    asof: str | pd.Timestamp | None = None,
) -> MarketData:
    """读取本地行情 CSV，返回对齐后的 MarketData。"""
    prices = _read_csv(price_path or DEFAULT_PRICE_PATH)

    ohlc = None
    op = ohlc_path or DEFAULT_OHLC_PATH
    if os.path.exists(op):
        raw = _read_csv(op)
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in raw.columns]
        if {"high", "low", "close"}.issubset(keep):
            ohlc = raw[keep].reindex(prices.index).ffill()

    # 价格序列做基础清洗：非正价格视为缺失
    prices = prices.where(prices > 0)

    if prices.columns.size == 0:
        raise DataError("行情宽表为空")

    data = MarketData(prices=prices, ohlc=ohlc)
    return data.slice_to(asof)


def describe(data: MarketData) -> dict:
    """数据体检信息，供 main.py 打印与 README 引用。"""
    return {
        "start": str(data.prices.index[0].date()),
        "end": str(data.prices.index[-1].date()),
        "trading_days": int(len(data.prices)),
        "tickers": int(data.prices.shape[1]),
        "coverage": {c: int(data.prices[c].notna().sum()) for c in data.prices.columns},
        "has_ohlc": data.has_ohlc(),
    }
