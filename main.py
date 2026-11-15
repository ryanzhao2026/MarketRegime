#!/usr/bin/env python3
"""Market Regime Monitor — 主程序。

用法
----
    python main.py                      # 用最新数据出报告 + 生成仪表盘
    python main.py --asof 2026-10-31    # 指定时点复算（只用该日及之前数据，无前视）
    python main.py --no-png             # 跳过 PNG 渲染
    python main.py --export-csv         # 导出完整因子矩阵

流程：加载数据 → 运行五引擎 → 合成打分 → 解释层 → 渲染 Dashboard。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dashboard
import scoring
from data import EQUITY, MarketData, describe, load_market_data


def run(asof: str | None = None, make_png: bool = True, export_csv: bool = False) -> dict:
    """执行完整流程，返回结果字典。"""
    print("[1/5] 加载行情数据 ...")
    md = load_market_data(asof=asof)
    info = describe(md)
    print(f"      区间 {info['start']} → {info['end']}  "
          f"| {info['trading_days']} 交易日 | {info['tickers']} 标的"
          f" | OHLC={'有' if info['has_ohlc'] else '无'}")

    print("[2/5] 运行五引擎（趋势/广度/波动率/跨资产/风险）...")
    df = scoring.build_frame(md)
    last = df.dropna(subset=["regime_score"]).iloc[-1]
    print(f"      最新有效日 {last.name.date()}  regime_score={last['regime_score']:.2f}")

    print("[3/5] 生成解释层 ...")
    n_univ = int(len([c for c in md.prices.columns]))
    res = scoring.explain(last, n_universe=8)
    report = scoring.format_report(res)
    print()
    print(report)

    out_dir = dashboard.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("[4/5] 渲染 HTML 仪表盘 ...")
    html_path = dashboard.render_html(df, md, res)
    print(f"      → {html_path}")

    png_path = None
    if make_png:
        print("[5/5] 渲染 PNG 预览 ...")
        try:
            png_path = dashboard.render_png(df, md, res)
            print(f"      → {png_path}")
        except Exception as e:                                    # noqa: BLE001
            print(f"      PNG 渲染失败（不影响 HTML）: {e}")
    else:
        print("[5/5] 已跳过 PNG")

    if export_csv:
        csv_path = os.path.join(out_dir, "regime_history.csv")
        df.to_csv(csv_path, float_format="%.4f")
        print(f"      因子矩阵 → {csv_path}")

    txt_path = os.path.join(out_dir, "report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"      文本报告 → {txt_path}")

    return {"frame": df, "result": res, "report": report,
            "html": html_path, "png": png_path, "info": info}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Market Regime Monitor")
    p.add_argument("--asof", default=None, help="截止日期 YYYY-MM-DD（含当日）")
    p.add_argument("--no-png", action="store_true", help="跳过 PNG 渲染")
    p.add_argument("--export-csv", action="store_true", help="导出完整因子矩阵")
    a = p.parse_args(argv)

    run(asof=a.asof, make_png=not a.no_png, export_csv=a.export_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
