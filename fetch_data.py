#!/usr/bin/env python3
"""取数层 v2：抓取行情 + 宏观序列，生成 data/ 下的标准宽表。

数据源分层（按优先级自动回退，**逐列**判定）
------------------------------------------
1. **yfinance**（主源）
   ETF + VIX/VVIX/VIX9D/VIX3M/VIX6M + ICE 美元指数，一个库全覆盖。
   对数据中心 IP 限流严格（常见 429），失败不致命，交给下层。

2. **Nasdaq 官方行情**（ETF 回退源）
   OHLCV 直出。注意：部分网络环境下会整段不可达（HTTP 000）。

3. **CBOE 官方指数历史**（波动率回退源）
   VIX / VVIX / VIX9D / VIX3M / VIX6M 的权威源，含数十年历史。

4. **FRED 官方宏观**（宏观与信用，v2 新增）
   免 API Key（fredgraph.csv 端点）。
   · BAA10Y    穆迪 Baa 与 10Y 国债利差 —— 纯信用定价，2006 起全历史
   · BAMLH0A0HYM2  ICE BofA 高收益 OAS —— 受授权限制仅近 3 年，作补充
   · NFCI      芝加哥联储全国金融条件指数
   · WALCL     美联储总资产
   · RRPONTSYD 隔夜逆回购
   · DTWEXBGS  美联储广义贸易加权美元（官方口径，取代 UUP 代理）

5. **仓库既有 CSV**（最后兜底）
   保证站点不出现空白，但**会显式标记为降级**。

设计要点
--------
· 信用因子用 BAA10Y 而非 HY OAS：后者虽更"专业"，但免费端点只回溯 3 年，
  v2 的核心目标是把历史拉到 2006 以覆盖三轮压力期，故以长度为先，
  同时把 HY OAS 一并取回作交叉验证。
· 任何一列落到回退层或既有快照，都会在台账与 fetch_report.json 里标出
  `degraded`，绝不静默补数——悄悄用旧数据是最危险的失败模式。

用法
----
    python fetch_data.py                      # 增量更新到最近交易日
    python fetch_data.py --start 2006-01-03   # 全量重建（默认即 2006）
    python fetch_data.py --source yfinance    # 强制单源（调试用）
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
# v2：默认起点前推到 2006，覆盖 2008 / 2020 / 2022 三轮压力期
DEFAULT_START = "2006-01-03"

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------- 标的映射
YF_TICKERS: dict[str, str] = {
    # ETF
    "SPY": "SPY", "RSP": "RSP", "QQQ": "QQQ", "IWM": "IWM",
    "TLT": "TLT", "HYG": "HYG", "GLD": "GLD", "DBC": "DBC",
    "XLK": "XLK", "XLF": "XLF", "XLE": "XLE", "XLV": "XLV",
    # 波动率指数（含期限结构四端）
    "VIX": "^VIX", "VVIX": "^VVIX",
    "VIX9D": "^VIX9D", "VIX3M": "^VIX3M", "VIX6M": "^VIX6M",
    # ICE 美元指数
    "DXY": "DX-Y.NYB",
}

ETF_COLS = ["SPY", "RSP", "QQQ", "IWM", "TLT", "HYG", "GLD", "DBC",
            "XLK", "XLF", "XLE", "XLV"]
# CBOE 可提供的五个波动率序列
INDEX_COLS = ["VIX", "VVIX", "VIX9D", "VIX3M", "VIX6M"]
DOLLAR_COL = "DXY"

# FRED：输出列 -> (series_id, 说明)
FRED_SERIES: dict[str, tuple[str, str]] = {
    "CREDIT_SPREAD": ("BAA10Y", "穆迪 Baa 与 10Y 国债利差(%)"),
    "HY_OAS": ("BAMLH0A0HYM2", "ICE BofA 高收益 OAS(%)"),
    "NFCI": ("NFCI", "芝加哥联储金融条件指数"),
    "WALCL": ("WALCL", "美联储总资产(百万美元)"),
    "RRP": ("RRPONTSYD", "隔夜逆回购(十亿美元)"),
    "DXY_BROAD": ("DTWEXBGS", "广义贸易加权美元指数"),
}
# 宏观列不是价格：不做正数清洗，且允许更长的前向填充（周频序列）
MACRO_COLS = list(FRED_SERIES.keys())

NASDAK_URL = (
    "https://api.nasdaq.com/api/quote/{sym}/historical"
    "?assetclass=etf&fromdate={frm}&todate={to}&limit=5000"
)
CBOE_URL = ("https://cdn.cboe.com/api/global/us_indices/"
            "daily_prices/{sym}_History.csv")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={frm}"

DOLLAR_PROXY_YF = "UUP"

RETRIES = 3
TIMEOUT = 45
POLITE_SLEEP = 0.35
FRED_SLEEP = 1.5        # FRED 对突发请求敏感，放慢但不过度

# ---- FRED 全局预算（关键）
# FRED 一旦连不上，单序列最坏耗时 = curl 重试 + urllib 重试 + 四段窗口回退，
# 六序列叠加可超过一小时（实测 CI 曾卡住 40 分钟以上）。
# 因此设三重上限：单请求超时、整个 FRED 阶段的总预算、连续失败快速放弃。
FRED_CURL_TIMEOUT = 40      # 单次 curl 上限（秒）
FRED_CURL_RETRIES = 2
FRED_STAGE_BUDGET = 180     # 整个 FRED 阶段的墙钟预算（秒）
FRED_FAIL_FAST = 2          # 连续失败 N 个序列后放弃剩余


# ================================================================ 工具
def _num(s) -> float:
    if s is None:
        return float("nan")
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).replace(",", "").replace("$", "").strip()
    if t in ("", "--", ".", "N/A", "n/a", "None", "null"):
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _get(url: str, *, json_mode: bool = False, retries: int = RETRIES,
         sleep_base: float = 1.2, timeout: int = TIMEOUT):
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA,
                              "Accept": "application/json, text/plain, */*",
                              "Accept-Language": "en-US,en;q=0.9"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode("utf-8", errors="replace")
            return json.loads(raw) if json_mode else raw
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError, OSError) as e:
            last = e
            if attempt < retries:
                time.sleep(sleep_base * attempt)
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
            time.sleep(20)

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


# ================================================================ 源 4：FRED
# FRED 在数据中心网络下会以「慢速分块」方式返回，urllib 的 read 容易整体超时
# （实测 GitHub Actions 中 6 个序列全部 TimeoutError，而同时刻 curl 预检为 200）。
# 因此主路径改用系统 curl，并把长区间切成若干窗口分别取，减小单次响应体。
FRED_CHUNKS = [
    ("2006-01-01", "2011-12-31"),
    ("2012-01-01", "2017-12-31"),
    ("2018-01-01", "2023-12-31"),
    ("2024-01-01", "2026-12-31"),
]


def _curl_text(url: str, timeout: int = 90, retries: int = 3) -> str | None:
    """用系统 curl 取文本。比 urllib 更能适应慢速/分块响应与重定向。"""
    import shutil
    import subprocess

    if not shutil.which("curl"):
        return None
    last = ""
    for attempt in range(1, retries + 1):
        try:
            # --http1.1 是必需的：FRED 的 HTTP/2 服务端会返回
            # "HTTP/2 stream not closed cleanly: INTERNAL_ERROR"，
            # 导致响应体被截断或直接失败；退回 1.1 后稳定。
            p = subprocess.run(
                ["curl", "-sSL", "--http1.1", "--compressed",
                 "--max-time", str(timeout), "--retry", "2",
                 "--retry-delay", "2", "--retry-all-errors",
                 "-A", UA, "-H", "Accept: text/csv,*/*", url],
                capture_output=True, text=True, timeout=timeout + 30,
            )
            out = p.stdout or ""
            if p.returncode == 0 and out.strip():
                return out
            last = (p.stderr or "").strip()[:120] or f"rc={p.returncode}"
        except Exception as e:                                # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(2.0 * attempt)
    if last:
        print(f"    ! curl 失败: {last}")
    return None


def _parse_fred(raw: str | None) -> pd.Series | None:
    if not raw or not raw.strip():
        return None
    try:
        df = pd.read_csv(StringIO(raw))
    except Exception:                                         # noqa: BLE001
        return None
    if df.shape[1] < 2:
        return None
    s = pd.Series(
        pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
        index=pd.to_datetime(df.iloc[:, 0], errors="coerce"),
    )
    s = s[s.index.notna()].dropna()
    return s if not s.empty else None


def fetch_fred(series_id: str, start: str,
               deadline: float | None = None) -> pd.Series | None:
    """FRED 官方 CSV，无需 API Key。缺失值以 '.' 或空表示。

    取数顺序：整段(curl) → 整段(urllib) → 分段(curl) → 分段(urllib)。
    分段的意义：FRED 生成长区间 CSV 较慢，切窗后单次响应更小、更易成功。

    deadline 是绝对时间戳（time.monotonic 口径）。每次发起网络请求前检查，
    一旦超出预算立即返回 None，保证不会被单个序列拖死整条流水线。
    """
    def _expired() -> bool:
        return deadline is not None and time.monotonic() > deadline

    url = FRED_URL.format(sid=series_id, frm=start)

    if not _expired():
        s = _parse_fred(_curl_text(url, timeout=FRED_CURL_TIMEOUT,
                                   retries=FRED_CURL_RETRIES))
        if s is None and not _expired():
            s = _parse_fred(_get(url, retries=1, sleep_base=2.0, timeout=35))
        if s is not None:
            time.sleep(FRED_SLEEP)
            return _norm(s.to_frame("v"))["v"]

    parts: list[pd.Series] = []
    for a, b in FRED_CHUNKS:
        if b < start:
            continue
        if _expired():
            print(f"    ! {series_id} 超出 FRED 阶段预算，停止分段回退")
            break
        a2 = max(a, start)
        u = f"{FRED_URL.format(sid=series_id, frm=a2)}&coed={b}"
        p = _parse_fred(_curl_text(u, timeout=FRED_CURL_TIMEOUT, retries=1))
        if p is not None:
            parts.append(p)
        time.sleep(1.0)
    if not parts:
        return None
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return _norm(s.to_frame("v"))["v"]


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
    report_path = os.path.join(out_dir, "fetch_report.json")
    old = _load_existing(price_path)

    print(f"取数区间: {start} → {end}   模式: {mode}\n")

    closes: dict[str, pd.Series] = {}
    ledger: dict[str, str] = {}          # 列 -> 实际使用的源
    degraded: list[str] = []             # 落到回退层/旧快照的列
    spy_ohlc: pd.DataFrame | None = None

    # ---------------- 1. yfinance（主源，批量）
    if mode in ("auto", "yfinance"):
        print("[源 1] yfinance（批量）")
        wanted = ETF_COLS + INDEX_COLS + [DOLLAR_COL]
        yf_data = fetch_yfinance(wanted, start, end)
        if yf_data:
            for c, s in yf_data.items():
                closes[c] = s
                ledger[c] = "yfinance"
            print(f"    ✓ 成功 {len(yf_data)} 列: {', '.join(sorted(yf_data))}")
            print(f"    最新日 {max(s.index[-1] for s in yf_data.values()).date()}")
        else:
            print("    ✗ 无数据，回退到官方接口")

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
            degraded.append(col)
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
            degraded.append(col)
            print(f"    ✓ {col:<5} {len(s):>5} 行  "
                  f"{s.index[-1].date()}  {s.iloc[-1]:.2f}")
            time.sleep(POLITE_SLEEP)

    # ---------------- 4. FRED（宏观与信用，v2 新增）
    missing_macro = [c for c in MACRO_COLS if c not in closes]
    if mode in ("auto", "fred") and missing_macro:
        print(f"\n[源 4] FRED 官方宏观（补 {len(missing_macro)} 个）")
        # 阶段总预算 + 连续失败快速放弃：防止 FRED 整体不可用时把流水线拖死
        fred_deadline = time.monotonic() + FRED_STAGE_BUDGET
        consec_fail = 0
        for col in missing_macro:
            if time.monotonic() > fred_deadline:
                print(f"    ! FRED 阶段预算 {FRED_STAGE_BUDGET}s 用尽，"
                      f"放弃剩余 {len(missing_macro)} 个序列")
                break
            if consec_fail >= FRED_FAIL_FAST:
                print(f"    ! FRED 连续失败 {consec_fail} 次，快速放弃剩余序列")
                break
            sid, desc = FRED_SERIES[col]
            s = fetch_fred(sid, start, deadline=fred_deadline)
            if s is None or s.empty:
                print(f"    ! {col} ({sid}) 失败")
                consec_fail += 1
                time.sleep(FRED_SLEEP)
                continue
            consec_fail = 0
            # 合理性校验：序列量级必须与已知常识相符，防止端点改版返回脏数据
            med = float(s.median())
            if col == "CREDIT_SPREAD" and not (0.3 <= med <= 15):
                print(f"    ! {col} 中位数 {med:.2f} 异常，剔除")
                time.sleep(FRED_SLEEP)
                continue
            if col == "NFCI" and not (-2.0 <= med <= 3.0):
                print(f"    ! {col} 中位数 {med:.2f} 异常，剔除")
                time.sleep(FRED_SLEEP)
                continue
            closes[col] = s
            ledger[col] = f"fred:{sid}"
            degraded.append(col)
            print(f"    ✓ {col:<14} {len(s):>5} 行  {s.index.min().date()} → "
                  f"{s.index[-1].date()}  末值 {s.iloc[-1]:.2f}  ({desc})")
            time.sleep(FRED_SLEEP)

    # ---------------- 5. 美元指数兜底（官方广义 > DXY > UUP 代理）
    if "DXY_BROAD" not in closes and DOLLAR_COL not in closes:
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
            degraded.append(DOLLAR_COL)
            print(f"\n[美元指数] 用 {DOLLAR_PROXY_YF} 代理，{len(proxy)} 行")
        else:
            print("\n[美元指数] 全部失败，跨资产引擎将自动降级")

    # ---------------- 6. 既有 CSV 兜底
    if old is not None:
        used_old = []
        for col in old.columns:
            if col not in closes and old[col].notna().any():
                closes[col] = old[col].dropna()
                ledger[col] = "existing-csv"
                degraded.append(col)
                used_old.append(col)
        if used_old:
            print(f"\n[源 5] 沿用已有历史: {', '.join(used_old)}")
        if DOLLAR_COL not in closes and "DINI" in old.columns \
                and old["DINI"].notna().any():
            closes[DOLLAR_COL] = old["DINI"].dropna()
            ledger[DOLLAR_COL] = "existing-csv"
            degraded.append(DOLLAR_COL)

    if "SPY" not in closes or closes["SPY"].empty:
        print("\n✗ 关键标的 SPY 无数据，终止（避免产出空站点）", file=sys.stderr)
        return 1

    # ---------------- 组装宽表
    frame = pd.DataFrame(closes).sort_index()
    cal = closes["SPY"].index
    frame = frame.reindex(cal)

    # 价格列最多补 5 个交易日；宏观列多为周频，允许 10 天
    price_cols = [c for c in frame.columns if c not in MACRO_COLS and c != "SPY"]
    macro_cols = [c for c in frame.columns if c in MACRO_COLS]
    if price_cols:
        frame[price_cols] = frame[price_cols].ffill(limit=5)
    if macro_cols:
        frame[macro_cols] = frame[macro_cols].ffill(limit=10)

    for c in list(frame.columns):
        if frame[c].dropna().empty:
            print(f"    ! 列 {c} 全空，剔除（引擎自动降级）")
            frame = frame.drop(columns=[c])

    frame.index.name = "date"
    frame.to_csv(price_path, float_format="%.4f")
    print(f"\n → {price_path}  {frame.shape[0]} 行 × {frame.shape[1]} 列")

    if spy_ohlc is not None and not spy_ohlc.empty:
        o = spy_ohlc.dropna(subset=[c for c in ("high", "low", "close")
                                    if c in spy_ohlc.columns])
        o.index.name = "date"
        o.to_csv(ohlc_path, float_format="%.4f")
        print(f"  → {ohlc_path}  {o.shape[0]} 行")

    # ---------------- 数据源台账
    print("\n  数据源台账")
    print("  " + "-" * 52)
    for c in frame.columns:
        src = ledger.get(c, "?")
        flag = "  ⚠降级" if c in degraded else ""
        print(f"    {c:<14} ← {src}{flag}")

    # ---------------- 新鲜度
    last = frame.index[-1]
    lag = (pd.Timestamp.today().normalize() - last).days
    print(f"\n最新交易日: {last.date()}（距今 {lag} 天）")
    if lag > 7:
        print("  ⚠ 数据可能滞后（含周末假日应 ≤4 天），请检查数据源",
              file=sys.stderr)

    # ---------------- 结构化报告（供 CI 与仪表盘读取降级状态）
    report = {
        "fetched_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "requested_start": start,
        "latest_trading_day": str(last.date()),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "per_column_source": {c: ledger.get(c, "?") for c in frame.columns},
        "degraded_columns": sorted(set(degraded)),
        "primary_source_hit": any(v == "yfinance" for v in ledger.values()),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  → {report_path}")

    if report["degraded_columns"]:
        print(f"\n⚠ 降级列: {', '.join(report['degraded_columns'])}")
        print("  这些列未由主源提供，已在台账与 fetch_report.json 标记。")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="抓取行情与宏观数据并生成标准宽表")
    p.add_argument("--start", default=DEFAULT_START, help="起始日期 YYYY-MM-DD")
    p.add_argument("--out-dir", default=DEFAULT_OUT, help="输出目录")
    p.add_argument("--source", default="auto",
                   choices=["auto", "yfinance", "nasdaq", "cboe", "fred"],
                   help="auto=分层回退（默认）；其余为强制单源")
    a = p.parse_args(argv)
    return build(a.start, a.out_dir, a.source)


if __name__ == "__main__":
    raise SystemExit(main())
