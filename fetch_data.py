#!/usr/bin/env python3
"""取数层：抓取最新行情，生成 data/ 下的标准宽表。

数据源分层（按优先级自动回退）
------------------------------
1. **yfinance**（主源）
   一个库覆盖 ETF + 指数 + 美元指数，无需 API Key。
   但 Yahoo 对**数据中心 IP**（含 GitHub Actions runner、云主机）限流极严，
   常见 `YFRateLimitError(429)`。因此本层失败不致命，交给下层。

2. **Nasdaq 官方行情接口**（ETF 回退源）
   直接提供 OHLCV，对数据中心 IP 友好，实测稳定。

3. **CBOE 官方指数历史**（VIX / VVIX / VIX3M 回退源）
   权威来源，可修正 Yahoo 上指数序列的口径差异。

4. **仓库已有 CSV**（最后兜底）
   全部源都失败时沿用上次数据，保证站点不出现空白。

这种分层是刻意的：**yfinance 优先**满足易用性，**官方接口兜底**保证
每日定时任务在 CI 环境里也能真正更新，而不是依赖一个会随机 429 的源。

用法
----
    python fetch_data.py                     # 更新到最近交易日
    python fetch_data.py --start 2024-01-01  # 指定起始日全量重建
    python fetch_data.py --source yfinance   # 强制只用 yfinance
    python fetch_data.py --source auto       # 默认，分层回退
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from io import StringIO

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(BASE_DIR, "data")
DEFAULT_START = "2024-01-01"

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------- 标的映射
# 输出列名 -> yfinance ticker
YF_TICKERS: dict[str, str] = {
    # ETF
    "SPY": "SPY", "RSP": "RSP", "QQQ": "QQQ", "IWM": "IWM",
    "TLT": "TLT", "HYG": "HYG", "GLD": "GLD", "DBC": "DBC",
    "XLK": "XLK", "XLF": "XLF", "XLE": "XLE", "XLV": "XLV",
    # 指数
    "VIX": "^VIX", "VVIX": "^VVIX", "VIX3M": "^VIX3M",
    # 美元指数（ICE DXY）
    "DXY": "DX-Y.NYB",
}

ETF_COLS = ["SPY", "RSP", "QQQ", "IWM", "TLT", "HYG", "GLD", "DBC",
            "XLK", "XLF", "XLE", "XLV"]
INDEX_COLS = ["VIX", "VVIX", "VIX3M"]
DOLLAR_COL = "DXY"

# Nasdaq 回退用（仅 ETF）
NASDAK_URL = (
    "https://api.nasdaq.com/api/quote/{sym}/historical"
    "?assetclass=etf&fromdate={frm}&todate={to}&limit=5000"
)
# CBOE 回退用（VIX/VVIX/VIX3M 官方历史）
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
# 美元指数 yfinance 不可得时的代理
DOLLAR_PROXY_YF = "UUP"

RETRIES = 3
TIMEOUT = 45
POLITE_SLEEP = 0.35


# ================================================================ 工具
def _num(s) -> float:
    if s is None:
        return float("nan")
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).replace(",", "").replace("$", "").strip()
    if t in ("", "--", "N/A", "n/a", "None", "null"):
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _get(url: str, *, json_mode: bool = False, retries: int = RETRIES):
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA,
                              "Accept": "application/json, text/plain, */*",
                              "Accept-Language": "en-US,en;q=0.9"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                raw = r.read().decode("utf-8", errors="replace")
            return json.loads(raw) if json_mode else raw
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError, OSError) as e:
            last = e
            if attempt < retries:
                time.sleep(1.2 * attempt)
    print(f"    ! HTTP 失败({retries}次): {type(last).__name__}: {last}")
    return None


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ================================================================ 源 1：yfinance
def fetch_yfinance(cols: list[str], start: str, end: str) -> dict[str, pd.Series]:
    """批量下载，返回 {col: close Series}。整体失败返回 {}。"""
    try:
        import yfinance as yf
    except ImportError:
        print("    ! 未安装 yfinance（pip install yfinance），跳过此源")
        return {}

    tickers = {c: YF_TICKERS[c] for c in cols if c in YF_TICKERS}
    if not tickers:
        return {}

    # 批量拉取以减少请求数（限流时关键的优化）
    sym_list = list(dict.fromkeys(tickers.values()))
    raw = None
    for attempt in (1, 2):
        try:
            raw = yf.download(
                sym_list, start=start, end=end, interval="1d",
                auto_adjust=False, progress=False, threads=False,
                group_by="ticker", timeout=30,
            )
            if raw is not None and not raw.empty:
                break
        except Exception as e:                                # noqa: BLE001
            print(f"    ! yfinance 第{attempt}次失败: {type(e).__name__}: "
                  f"{str(e)[:90]}")
        raw = None
        if attempt == 1:
            time.sleep(20)          # 限流时给一点冷却再试一次

    if raw is None or raw.empty:
        print("    ! yfinance 无数据（通常是被限流 429）")
        return {}

    out: dict[str, pd.Series] = {}
    multi = isinstance(raw.columns, pd.MultiIndex)
    for col, sym in tickers.items():
        try:
            if multi:
                if sym not in raw.columns.get_level_values(0):
                    continue
                sub = raw[sym]
            else:
                sub = raw
            if "Close" not in sub.columns:
                continue
            s = pd.to_numeric(sub["Close"], errors="coerce").dropna()
            s = _norm(s.to_frame("close"))["close"]
            if not s.empty:
                out[col] = s
        except Exception:                                     # noqa: BLE001
            continue
    return out


# ================================================================ 源 2：Nasdaq
def fetch_nasdaq(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    data = _get(NASDAK_URL.format(sym=symbol, frm=start, to=end), json_mode=True)
    if not data:
        return None
    d = data.get("data") or {}
    rows = ((d.get("tradesTable") or {}).get("rows")) or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    keep = {}
    for src in ("open", "high", "low", "close", "volume"):
        if src in df.columns:
            keep[src] = df[src].map(_num)
    if "close" not in keep:
        return None
    out = pd.DataFrame(keep)
    out.index = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    out = out[out.index.notna()]
    return _norm(out)


# ================================================================ 源 3：CBOE
def fetch_cboe(symbol: str) -> pd.Series | None:
    raw = _get(CBOE_URL.format(sym=symbol))
    if not raw or not raw.strip():
        return None
    try:
        df = pd.read_csv(StringIO(raw))
    except Exception as e:                                    # noqa: BLE001
        print(f"    ! {symbol} CBOE CSV 解析失败: {e}")
        return None
    if "DATE" not in df.columns:
        return None
    col = next((c for c in ("CLOSE", "VVIX", "Close", symbol)
                if c in df.columns), None)
    if col is None:
        return None
    s = pd.Series(df[col].map(_num).values,
                  index=pd.to_datetime(df["DATE"], format="%m/%d/%Y",
                                       errors="coerce"))
    s = s[s.index.notna()]
    return _norm(s.to_frame("v"))["v"].dropna()


# ================================================================ 组装
def _load_existing(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()
    except Exception:                                         # noqa: BLE001
        return None


def build(start: str, out_dir: str, mode: str = "auto") -> int:
    today = pd.Timestamp.today().normalize()
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    os.makedirs(out_dir, exist_ok=True)

    price_path = os.path.join(out_dir, "market_data.csv")
    ohlc_path = os.path.join(out_dir, "spy_ohlc.csv")
    old = _load_existing(price_path)

    print(f"取数区间: {start} → {end}   模式: {mode}\n")

    closes: dict[str, pd.Series] = {}
    ledger: dict[str, str] = {}          # 列 -> 实际使用的源
    spy_ohlc: pd.DataFrame | None = None

    # ---------------- 1. yfinance（主源，批量）
    if mode in ("auto", "yfinance"):
        print("[源 1] yfinance（批量）")
        wanted = [c for c in (ETF_COLS + INDEX_COLS + [DOLLAR_COL])]
        yf_data = fetch_yfinance(wanted, start, end)
        if yf_data:
            for c, s in yf_data.items():
                closes[c] = s
                ledger[c] = "yfinance"
            print(f"    ✓ 成功 {len(yf_data)} 列: "
                  f"{', '.join(sorted(yf_data))}")
            print(f"    最新日 {max(s.index[-1] for s in yf_data.values()).date()}")
        else:
            print("    ✗ 无数据，回退到官方接口")

        # SPY OHLC 优先也取 yfinance
        if "SPY" in yf_data:
            try:
                import yfinance as yf
                o = yf.download("SPY", start=start, end=end, interval="1d",
                                auto_adjust=False, progress=False,
                                threads=False, timeout=30)
                if o is not None and not o.empty:
                    if isinstance(o.columns, pd.MultiIndex):
                        o.columns = o.columns.get_level_values(0)
                    o.columns = [str(c).lower() for c in o.columns]
                    keep = [c for c in ("open", "high", "low", "close", "volume")
                            if c in o.columns]
                    if {"high", "low", "close"}.issubset(keep):
                        spy_ohlc = _norm(o[keep])
            except Exception:                                 # noqa: BLE001
                pass

    # ---------------- 2. Nasdaq（ETF 回退）
    missing_etf = [c for c in ETF_COLS if c not in closes]
    if mode in ("auto", "nasdaq") and missing_etf:
        print(f"\n[源 2] Nasdaq 官方接口（补 {len(missing_etf)} 个 ETF）")
        for col in missing_etf:
            df = fetch_nasdaq(col, start, end)
            if df is None or df["close"].dropna().empty:
                print(f"    ! {col} 失败")
                continue
            closes[col] = df["close"].dropna()
            ledger[col] = "nasdaq"
            if col == "SPY":
                keep = [c for c in ("open", "high", "low", "close", "volume")
                        if c in df.columns]
                spy_ohlc = df[keep]
            print(f"    ✓ {col:<4} {len(df):>4} 行  "
                  f"{df.index[-1].date()}  {df['close'].iloc[-1]:.2f}")
            time.sleep(POLITE_SLEEP)

    # ---------------- 3. CBOE（指数回退）
    missing_idx = [c for c in INDEX_COLS if c not in closes]
    if mode in ("auto", "cboe") and missing_idx:
        print(f"\n[源 3] CBOE 官方指数（补 {len(missing_idx)} 个）")
        for col in missing_idx:
            s = fetch_cboe(col)
            if s is None or s.empty:
                print(f"    ! {col} 失败")
                continue
            closes[col] = s
            ledger[col] = "cboe"
            print(f"    ✓ {col:<5} {len(s):>5} 行  "
                  f"{s.index[-1].date()}  {s.iloc[-1]:.2f}")
            time.sleep(POLITE_SLEEP)

    # ---------------- 4. 美元指数兜底
    if DOLLAR_COL not in closes:
        # 先试 UUP 代理（yfinance 或 Nasdaq）
        proxy = None
        if mode in ("auto", "yfinance"):
            try:
                import yfinance as yf
                d = yf.download(DOLLAR_PROXY_YF, start=start, end=end,
                                interval="1d", auto_adjust=False,
                                progress=False, threads=False, timeout=30)
                if d is not None and not d.empty:
                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = d.columns.get_level_values(0)
                    s = pd.to_numeric(d["Close"], errors="coerce").dropna()
                    if not s.empty:
                        proxy = s
            except Exception:                                 # noqa: BLE001
                pass
        if proxy is None:
            df = fetch_nasdaq(DOLLAR_PROXY_YF, start, end)
            if df is not None and not df["close"].dropna().empty:
                proxy = df["close"].dropna()
        if proxy is not None:
            closes[DOLLAR_COL] = proxy
            ledger[DOLLAR_COL] = f"uup-proxy({DOLLAR_PROXY_YF})"
            print(f"\n[美元指数] 用 {DOLLAR_PROXY_YF} 代理，{len(proxy)} 行")
        else:
            print("\n[美元指数] 全部失败，跨资产引擎将自动降级")

    # ---------------- 5. 既有 CSV 兜底
    if old is not None:
        used_old = []
        for col in old.columns:
            if col not in closes and old[col].notna().any():
                closes[col] = old[col].dropna()
                ledger[col] = "existing-csv"
                used_old.append(col)
        if used_old:
            print(f"\n[源 4] 沿用已有历史: {', '.join(used_old)}")
        # 兼容旧列名 DINI
        if DOLLAR_COL not in closes and "DINI" in old.columns \
                and old["DINI"].notna().any():
            closes[DOLLAR_COL] = old["DINI"].dropna()
            ledger[DOLLAR_COL] = "existing-csv"

    if "SPY" not in closes or closes["SPY"].empty:
        print("\n✗ 关键标的 SPY 无数据，终止（避免产出空站点）", file=sys.stderr)
        return 1

    # ---------------- 组装宽表
    frame = pd.DataFrame(closes).sort_index()
    cal = closes["SPY"].index
    frame = frame.reindex(cal)
    ffill_cols = [c for c in frame.columns if c != "SPY"]
    frame[ffill_cols] = frame[ffill_cols].ffill(limit=5)

    for c in list(frame.columns):
        if frame[c].dropna().empty:
            print(f"    ! 列 {c} 全空，剔除（引擎自动降级）")
            frame = frame.drop(columns=[c])

    frame.index.name = "date"
    frame.to_csv(price_path, float_format="%.4f")
    print(f"\n  → {price_path}  {frame.shape[0]} 行 × {frame.shape[1]} 列")

    if spy_ohlc is not None and not spy_ohlc.empty:
        o = spy_ohlc.dropna(subset=[c for c in ("high", "low", "close")
                                    if c in spy_ohlc.columns])
        o.index.name = "date"
        o.to_csv(ohlc_path, float_format="%.4f")
        print(f"  → {ohlc_path}  {o.shape[0]} 行")

    # ---------------- 数据源台账
    print("\n  数据源台账")
    print("  " + "-" * 46)
    for c in frame.columns:
        print(f"    {c:<6} ← {ledger.get(c, '?')}")

    # ---------------- 新鲜度
    last = frame.index[-1]
    lag = (pd.Timestamp.today().normalize() - last).days
    print(f"\n最新交易日: {last.date()}（距今 {lag} 天）")
    if lag > 7:
        print("  ⚠ 数据可能滞后（含周末假日应 ≤4 天），请检查数据源",
              file=sys.stderr)

    # 供 CI 判断是否降到回退源
    used = set(ledger.values())
    if "yfinance" not in used:
        print("\n  注: yfinance 本次未成功（多为数据中心 IP 限流），"
              "已由官方接口接管，数据仍为最新。")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="抓取行情数据并生成标准宽表")
    p.add_argument("--start", default=DEFAULT_START, help="起始日期 YYYY-MM-DD")
    p.add_argument("--out-dir", default=DEFAULT_OUT, help="输出目录")
    p.add_argument("--source", default="auto",
                   choices=["auto", "yfinance", "nasdaq", "cboe"],
                   help="auto=分层回退（默认）；其余为强制单源")
    a = p.parse_args(argv)
    return build(a.start, a.out_dir, a.source)


if __name__ == "__main__":
    raise SystemExit(main())
