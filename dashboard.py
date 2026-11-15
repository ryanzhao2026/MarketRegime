"""Dashboard 层：把打分结果渲染成 6 面板机构风格视图。

输出两种形态：
  outputs/dashboard.html  — 自包含单文件（内联 SVG，无外部依赖，可直接分享）
  outputs/dashboard.png   — 静态预览图（matplotlib，用于快速审阅/截图）

六个面板：
  A  Price + Regime Color      价格叠加档位着色背景
  B  Regime & Confidence       当前档位与置信度 KPI
  C  Pillar Scores             五支柱得分条
  D  Regime Score History      总分历史 + 档位分界带
  E  Pillar History            五支柱时间序列
  F  Drivers & Metrics         解释层：正负驱动 + 关键读数表
"""

from __future__ import annotations

import html
import os
from datetime import datetime

import numpy as np
import pandas as pd

from data import DISPLAY_NAMES, EQUITY, MarketData
from scoring import (PILLAR_LABELS, REGIME_BANDS, REGIME_CN, REGIME_COLORS,
                     map_regime)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")

PILLAR_COLS = ["pillar_trend", "pillar_breadth", "pillar_risk_on",
               "pillar_stability", "pillar_safety"]
PILLAR_KEYS = ["trend", "breadth", "risk_on", "stability", "safety"]
PILLAR_COLOR_MAP = {
    "trend": "#4c6ef5",
    "breadth": "#12b886",
    "risk_on": "#f59f00",
    "stability": "#7950f2",
    "safety": "#e8590c",
}


# ============================================================ SVG 基元
def _x(i: int, n: int, w: float, pad_l: float, pad_r: float) -> float:
    if n <= 1:
        return pad_l
    return pad_l + (w - pad_l - pad_r) * i / (n - 1)


def _y(v: float, lo: float, hi: float, h: float, pad_t: float, pad_b: float) -> float:
    if hi <= lo:
        return h - pad_b
    r = (v - lo) / (hi - lo)
    return h - pad_b - (h - pad_t - pad_b) * r


def _line_path(vals, lo, hi, w, h, pad_l, pad_r, pad_t, pad_b):
    n = len(vals)
    d, pen = [], False
    for i, v in enumerate(vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            pen = False
            continue
        x = _x(i, n, w, pad_l, pad_r)
        y = _y(float(v), lo, hi, h, pad_t, pad_b)
        d.append(("L" if pen else "M") + f"{x:.1f},{y:.1f}")
        pen = True
    return " ".join(d)


def _grid(h, w, lo, hi, pad_l, pad_r, pad_t, pad_b, ticks=4, fmt="{:.0f}"):
    out = []
    for k in range(ticks + 1):
        v = lo + (hi - lo) * k / ticks
        y = _y(v, lo, hi, h, pad_t, pad_b)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" '
                   f'stroke="#e9ecef" stroke-width="1"/>')
        out.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
                   f'font-size="10" fill="#868e96">{fmt.format(v)}</text>')
    return "".join(out)


def _xlabels(idx: pd.DatetimeIndex, w, pad_l, pad_r, h, n_ticks=6):
    n = len(idx)
    if n == 0:
        return ""
    out = []
    step = max(1, n // n_ticks)
    for i in range(0, n, step):
        x = _x(i, n, w, pad_l, pad_r)
        out.append(f'<text x="{x:.1f}" y="{h - 6}" text-anchor="middle" '
                   f'font-size="10" fill="#868e96">{idx[i].strftime("%y-%m-%d")}</text>')
    return "".join(out)


# ============================================================ 面板 A
def panel_price(df: pd.DataFrame, md: MarketData, w=1180, h=300) -> str:
    spy = md.series(EQUITY).reindex(df.index).ffill()
    vals = spy.values.astype(float)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    pl, pr, pt, pb = 62, 18, 16, 42

    # 档位色背景带
    regimes = df["regime"].values
    bands = []
    n = len(regimes)
    start = 0
    for i in range(1, n + 1):
        if i == n or regimes[i] != regimes[start]:
            r = regimes[start]
            color = REGIME_COLORS.get(r, "#adb5bd")
            x0 = _x(start, n, w, pl, pr)
            x1 = _x(min(i, n - 1), n, w, pl, pr)
            bands.append(f'<rect x="{x0:.1f}" y="{pt}" width="{max(1.0, x1 - x0):.1f}" '
                         f'height="{h - pt - pb}" fill="{color}" opacity="0.18"/>')
            start = i
    # 档位换挡竖线
    seps = []
    for i in range(1, n):
        if regimes[i] != regimes[i - 1]:
            x = _x(i, n, w, pl, pr)
            seps.append(f'<line x1="{x:.1f}" y1="{pt}" x2="{x:.1f}" y2="{h - pb}" '
                        f'stroke="#495057" stroke-width="1" stroke-dasharray="3,3" opacity="0.45"/>')

    path = _line_path(vals, lo, hi, w, h, pl, pr, pt, pb)
    last = vals[-1]
    ly = _y(last, lo, hi, h, pt, pb)
    dot = (f'<circle cx="{_x(n - 1, n, w, pl, pr):.1f}" cy="{ly:.1f}" r="4" '
           f'fill="{REGIME_COLORS.get(regimes[-1], "#adb5bd")}" stroke="#fff" stroke-width="1.5"/>'
           f'<text x="{_x(n - 1, n, w, pl, pr) - 8:.1f}" y="{ly - 10:.1f}" text-anchor="end" '
           f'font-size="12" font-weight="700" fill="#212529">{last:.2f}</text>')

    # 图例
    legend_items = []
    lx = pl
    for _lo, _hi, r in REGIME_BANDS:
        legend_items.append(
            f'<rect x="{lx}" y="{pt - 4}" width="9" height="9" rx="2" '
            f'fill="{REGIME_COLORS[r]}" opacity="0.85"/>'
            f'<text x="{lx + 13}" y="{pt + 4}" font-size="10" fill="#495057">{r}</text>'
        )
        lx += 108

    return f"""
<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet" role="img">
  {''.join(bands)}{''.join(seps)}
  {_grid(h, w, lo, hi, pl, pr, pt, pb, 4, "{:.0f}")}
  <path d="{path}" fill="none" stroke="#1c1c1e" stroke-width="1.8" stroke-linejoin="round"/>
  {dot}
  <text x="{pl}" y="{pt + 22}" font-size="11" fill="#adb5bd">SPY</text>
  {''.join(legend_items)}
  {_xlabels(df.index, w, pl, pr, h, 6)}
</svg>"""


# ============================================================ 面板 D
def panel_regime_history(df: pd.DataFrame, w=1180, h=280) -> str:
    vals = df["regime_score"].values.astype(float)
    lo, hi = 0.0, 100.0
    pl, pr, pt, pb = 62, 18, 16, 42

    # 档位横带
    bands = []
    for lo_b, hi_b, name in REGIME_BANDS:
        y0 = _y(min(hi_b, 100.0), lo, hi, h, pt, pb)
        y1 = _y(max(lo_b, 0.0), lo, hi, h, pt, pb)
        bands.append(f'<rect x="{pl}" y="{y0:.1f}" width="{w - pl - pr}" '
                     f'height="{max(1.0, y1 - y0):.1f}" '
                     f'fill="{REGIME_COLORS[name]}" opacity="0.10"/>')
        label = f"{name}  {REGIME_CN[name]}"
        bands.append(f'<text x="{w - pr - 8}" y="{(y0 + y1) / 2 + 3:.1f}" text-anchor="end" '
                     f'font-size="9" fill="#adb5bd">{label}</text>')

    path = _line_path(vals, lo, hi, w, h, pl, pr, pt, pb)
    # 面积
    area = path + f" L{_x(len(vals) - 1, len(vals), w, pl, pr):.1f},{h - pb:.1f}" \
                  f" L{_x(0, len(vals), w, pl, pr):.1f},{h - pb:.1f} Z"
    last = vals[-1] if not np.isnan(vals[-1]) else 50.0
    ly = _y(last, lo, hi, h, pt, pb)
    dot = (f'<circle cx="{_x(len(vals) - 1, len(vals), w, pl, pr):.1f}" cy="{ly:.1f}" '
           f'r="4.5" fill="{REGIME_COLORS.get(map_regime(last), "#adb5bd")}" '
           f'stroke="#fff" stroke-width="1.5"/>'
           f'<text x="{_x(len(vals) - 1, len(vals), w, pl, pr) - 10:.1f}" y="{ly - 11:.1f}" '
           f'text-anchor="end" font-size="12" font-weight="700" fill="#212529">{last:.1f}</text>')

    return f"""
<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet" role="img">
  {''.join(bands)}
  {_grid(h, w, lo, hi, pl, pr, pt, pb, 4, "{:.0f}")}
  <path d="{area}" fill="url(#gRegime)" opacity="0.28"/>
  <defs><linearGradient id="gRegime" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#4c6ef5"/><stop offset="100%" stop-color="#4c6ef5" stop-opacity="0"/>
  </linearGradient></defs>
  <path d="{path}" fill="none" stroke="#4c6ef5" stroke-width="1.9"/>
  {dot}
  {_xlabels(df.index, w, pl, pr, h, 6)}
</svg>"""


# ============================================================ 面板 E
def panel_pillar_history(df: pd.DataFrame, w=1180, h=300) -> str:
    lo, hi = 0.0, 100.0
    pl, pr, pt, pb = 62, 18, 16, 46
    parts = [_grid(h, w, lo, hi, pl, pr, pt, pb, 4, "{:.0f}")]
    for key, col in zip(PILLAR_KEYS, PILLAR_COLS):
        vals = df[col].values.astype(float)
        d = _line_path(vals, lo, hi, w, h, pl, pr, pt, pb)
        parts.append(f'<path d="{d}" fill="none" stroke="{PILLAR_COLOR_MAP[key]}" '
                     f'stroke-width="1.6" opacity="0.92"/>')
    # 中性线
    yn = _y(50, lo, hi, h, pt, pb)
    parts.append(f'<line x1="{pl}" y1="{yn:.1f}" x2="{w - pr}" y2="{yn:.1f}" '
                 f'stroke="#868e96" stroke-width="1" stroke-dasharray="4,4" opacity="0.6"/>')

    legend = []
    lx = pl
    for key, label in zip(PILLAR_KEYS, PILLAR_LABELS.values()):
        legend.append(
            f'<rect x="{lx}" y="{h - 28}" width="14" height="3" rx="1.5" '
            f'fill="{PILLAR_COLOR_MAP[key]}"/>'
            f'<text x="{lx + 19}" y="{h - 22}" font-size="10" fill="#495057">'
            f'{html.escape(label.split()[0])}</text>'
        )
        lx += 92
    return f"""
<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet" role="img">
  {''.join(parts)}{''.join(legend)}
</svg>"""


# ============================================================ 面板 C
def panel_pillars_bars(df: pd.DataFrame, w=560, h=300) -> str:
    row = df.iloc[-1]
    pl, pr, pt, pb = 88, 46, 14, 18
    n = len(PILLAR_KEYS)
    bar_h = (h - pt - pb) / n * 0.62
    gap = (h - pt - pb) / n
    parts = []
    for i, (key, col) in enumerate(zip(PILLAR_KEYS, PILLAR_COLS)):
        v = row.get(col, np.nan)
        y = pt + gap * i + (gap - bar_h) / 2
        color = PILLAR_COLOR_MAP[key]
        if pd.isna(v):
            parts.append(f'<text x="{pl}" y="{y + bar_h * 0.7:.1f}" font-size="11" '
                         f'fill="#adb5bd">n/a</text>')
            continue
        v = float(v)
        bw = (w - pl - pr) * v / 100.0
        parts.append(
            f'<rect x="{pl}" y="{y:.1f}" width="{w - pl - pr}" height="{bar_h:.1f}" '
            f'rx="4" fill="#f1f3f5"/>'
            f'<rect x="{pl}" y="{y:.1f}" width="{max(2.0, bw):.1f}" height="{bar_h:.1f}" '
            f'rx="4" fill="{color}"/>'
            f'<text x="{pl - 10}" y="{y + bar_h * 0.72:.1f}" text-anchor="end" '
            f'font-size="11" fill="#343a40">{html.escape(PILLAR_LABELS[key].split()[0])}</text>'
            f'<text x="{pl + max(2.0, bw) + 8:.1f}" y="{y + bar_h * 0.72:.1f}" '
            f'font-size="11.5" font-weight="700" fill="#212529">{v:.0f}</text>'
        )
    # 50 中性刻度
    x50 = pl + (w - pl - pr) * 0.5
    parts.append(f'<line x1="{x50:.1f}" y1="{pt}" x2="{x50:.1f}" y2="{h - pb}" '
                 f'stroke="#868e96" stroke-width="1" stroke-dasharray="3,3" opacity="0.7"/>')
    return f"""
<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet" role="img">
  {''.join(parts)}
</svg>"""


# ============================================================ 面板 B
def panel_regime_kpi(res: dict) -> str:
    regime = res["regime"]
    color = REGIME_COLORS.get(regime, "#adb5bd")
    score = res["regime_score"] or 0.0
    conf = res["confidence"] or 0.0
    # 置信度环
    r_ = 46
    circ = 2 * np.pi * r_
    dash = circ * conf / 100.0
    return f"""
<div class="kpi-wrap">
  <div class="kpi-main" style="border-color:{color}">
    <div class="kpi-cap">CURRENT REGIME</div>
    <div class="kpi-regime" style="color:{color}">{html.escape(regime.upper())}</div>
    <div class="kpi-cn">{REGIME_CN.get(regime, '')}</div>
    <div class="kpi-score"><span>{score:.1f}</span><small>/100</small></div>
  </div>
  <div class="kpi-conf">
    <svg viewBox="0 0 120 120" width="132" height="132">
      <circle cx="60" cy="60" r="{r_}" fill="none" stroke="#f1f3f5" stroke-width="10"/>
      <circle cx="60" cy="60" r="{r_}" fill="none" stroke="{color}" stroke-width="10"
              stroke-linecap="round" stroke-dasharray="{dash:.1f} {circ:.1f}"
              transform="rotate(-90 60 60)"/>
      <text x="60" y="58" text-anchor="middle" font-size="26" font-weight="700"
            fill="#212529">{conf:.0f}%</text>
      <text x="60" y="78" text-anchor="middle" font-size="10" fill="#868e96">CONFIDENCE</text>
    </svg>
    <div class="kpi-note">置信度 = 支柱共识度 × 边界距离 × 数据覆盖<br>衡量模型内部一致性，非预测准确率</div>
  </div>
</div>"""


# ============================================================ 面板 F
def panel_drivers(res: dict, df: pd.DataFrame, md: MarketData) -> str:
    pos = res["drivers_positive"]
    neg = res["drivers_negative"]
    try:
        _spy_close = float(md.series(EQUITY).iloc[-1])
    except Exception:
        _spy_close = float("nan")

    def rows(items, cls, sign):
        if not items:
            return f'<li class="{cls} empty">—</li>'
        return "".join(
            f'<li class="{cls}"><span class="sign">{sign}</span>{html.escape(x)}</li>'
            for x in items
        )

    # 贡献条形
    contribs = res["contributions"]
    max_abs = max([abs(c["contribution"]) for c in contribs] + [1.0])
    bars = []
    for c in contribs:
        v = c["contribution"]
        pctw = abs(v) / max_abs * 100.0
        color = "#12b886" if v >= 0 else "#e03131"
        bars.append(
            f'<div class="crow">'
            f'<span class="cname">{html.escape(c["label"].split()[0])}</span>'
            f'<span class="cbar"><span class="cfill" style="width:{pctw:.1f}%;'
            f'background:{color}"></span></span>'
            f'<span class="cval" style="color:{color}">{v:+.1f}</span>'
            f'</div>'
        )

    d = res["detail"]
    metrics = [
        ("SPY 收盘", _nv(_spy_close, 2)),
        ("EMA200", _nv(df["raw_ema200"].iloc[-1], 2)),
        ("VIX", _nv(d["volatility"].get("vix"), 1)),
        ("VVIX", _nv(d["volatility"].get("vvix"), 1)),
        ("已实现波动率", _nv(d["volatility"].get("realized_vol_pct"), 1, "%")),
        ("ADX(14)", _nv(d["trend"].get("adx"), 1)),
        ("距 EMA200", _nv(d["trend"].get("ema200_dist_pct"), 1, "%", True)),
        ("站上 MA200", d["breadth"].get("above_ma200_count") or "n/a"),
        ("期限结构 近/远", _nv(d["volatility"].get("term_structure"), 3)),
        (_credit_label(d["cross_asset"]), _credit_value(d["cross_asset"])),
        ("美元指数 63日", _nv(d["cross_asset"].get("dxy_chg63_pct"), 1, "%", True)),
        ("收益偏度", _nv(d["risk"].get("skew"), 2)),
        ("当前回撤", _nv(d["risk"].get("drawdown_pct"), 1, "%")),
        ("平均相关性", _nv(d["risk"].get("avg_corr"), 2)),
    ]
    mrows = "".join(
        f'<tr><td>{html.escape(k)}</td><td class="num">{html.escape(str(v))}</td></tr>'
        for k, v in metrics
    )

    return f"""
<div class="drv-grid">
  <div>
    <div class="sub-cap">DRIVERS</div>
    <ul class="drv-list">{rows(pos, "pos", "+")}{rows(neg, "neg", "−")}</ul>
    <div class="sub-cap" style="margin-top:16px">PILLAR CONTRIBUTION (pp)</div>
    <div class="contrib">{''.join(bars)}</div>
  </div>
  <div>
    <div class="sub-cap">KEY METRICS</div>
    <table class="mtable">{mrows}</table>
  </div>
</div>"""


def _credit_label(c: dict) -> str:
    """信用维度的指标名随实际口径变化，避免用统一标签误导读者。"""
    src = c.get("credit_source", "unavailable")
    if src == "CREDIT_SPREAD":
        return "信用价差(Baa-10Y)"
    if src == "HY_OAS":
        return "高收益OAS"
    if src == "HYG/TLT(fallback)":
        return "HYG/TLT 63日"
    return "信用"


def _credit_value(c: dict) -> str:
    """价差类用绝对水平 + 变化(pp)；比值类只有变化(%)。"""
    src = c.get("credit_source", "unavailable")
    if src in ("CREDIT_SPREAD", "HY_OAS"):
        lvl = _nv(c.get("credit_spread_pct"), 2)
        chg = _nv(c.get("credit_chg63"), 2)
        return f"{lvl} ({chg}pp)"
    return _nv(c.get("credit_chg63_pct"), 1, "%", True)


def _nv(v, nd=1, suffix="", signed=False):
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(f):
        return "n/a"
    s = f"{f:+.{nd}f}" if signed else f"{f:.{nd}f}"
    return s + suffix


# ============================================================ HTML 组装
CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f8f9fa;color:#212529;
 font-family:"Source Han Sans SC","Noto Sans CJK SC","PingFang SC",
 -apple-system,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1240px;margin:0 auto;padding:28px 24px 56px}
header{display:flex;align-items:flex-end;justify-content:space-between;
 flex-wrap:wrap;gap:12px;margin-bottom:22px}
h1{font-size:22px;margin:0;letter-spacing:-.3px;font-weight:700}
h1 span{color:#868e96;font-weight:400;font-size:14px;margin-left:10px}
.meta{font-size:12.5px;color:#868e96;text-align:right;line-height:1.7}
.card{background:#fff;border:1px solid #e9ecef;border-radius:12px;padding:16px 18px;
 margin-bottom:16px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.card-cap{font-size:11px;letter-spacing:.10em;color:#868e96;font-weight:600;
 text-transform:uppercase;margin-bottom:10px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.row-main{display:grid;grid-template-columns:1.15fr 1fr;gap:16px}
@media(max-width:940px){.row2,.row-main{grid-template-columns:1fr}}
.kpi-wrap{display:flex;gap:14px;align-items:stretch}
.kpi-main{flex:1;border-left:5px solid #adb5bd;background:#fbfbfc;border-radius:8px;
 padding:14px 16px}
.kpi-cap{font-size:10.5px;letter-spacing:.1em;color:#868e96;font-weight:600}
.kpi-regime{font-size:30px;font-weight:800;letter-spacing:-.8px;line-height:1.15;
 margin:2px 0 0}
.kpi-cn{font-size:12.5px;color:#868e96;margin-bottom:8px}
.kpi-score{font-size:15px;color:#495057}
.kpi-score span{font-size:34px;font-weight:800;color:#212529;letter-spacing:-1px}
.kpi-score small{font-size:12px;color:#adb5bd;margin-left:2px}
.kpi-conf{display:flex;flex-direction:column;align-items:center;justify-content:center;
 min-width:170px}
.kpi-note{font-size:10px;color:#adb5bd;line-height:1.6;text-align:center;margin-top:6px}
.sub-cap{font-size:11px;letter-spacing:.08em;color:#868e96;font-weight:600;
 text-transform:uppercase;margin-bottom:9px}
.drv-list{list-style:none;margin:0;padding:0;font-size:13px}
.drv-list li{padding:7px 10px;border-radius:6px;margin-bottom:5px;line-height:1.45;
 display:flex;gap:8px;align-items:flex-start}
.drv-list li.pos{background:#ebfbee;color:#2b8a3e}
.drv-list li.neg{background:#fff4e6;color:#c92a2a}
.drv-list li.empty{background:#f8f9fa;color:#adb5bd}
.sign{font-weight:800;font-size:14px;line-height:1.2}
.contrib{font-size:12px}
.crow{display:grid;grid-template-columns:64px 1fr 52px;gap:8px;align-items:center;
 margin-bottom:6px}
.cname{color:#495057}
.cbar{background:#f1f3f5;border-radius:3px;height:9px;overflow:hidden;display:block}
.cfill{display:block;height:100%;border-radius:3px}
.cval{text-align:right;font-weight:700;font-variant-numeric:tabular-nums}
.mtable{width:100%;border-collapse:collapse;font-size:12.5px}
.mtable td{padding:5.5px 4px;border-bottom:1px solid #f1f3f5;color:#495057}
.mtable td.num{text-align:right;font-weight:600;color:#212529;
 font-variant-numeric:tabular-nums}
.drv-grid{display:grid;grid-template-columns:1fr 1fr;gap:28px}
@media(max-width:760px){.drv-grid{grid-template-columns:1fr}}
footer{font-size:11px;color:#adb5bd;line-height:1.8;margin-top:8px;
 border-top:1px solid #e9ecef;padding-top:14px}
code{background:#f1f3f5;padding:1px 5px;border-radius:3px;font-size:11px}
"""


def render_html(df: pd.DataFrame, md: MarketData, res: dict,
                out_path: str | None = None) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = out_path or os.path.join(OUT_DIR, "dashboard.html")

    wz = df["regime"].iloc[-1]
    color = REGIME_COLORS.get(wz, "#adb5bd")
    coverage = md.prices.index
    n_univ = res["detail"]["breadth"].get("universe_size", 8)

    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Regime Monitor — {res['asof']}</title>
<style>{CSS}</style></head><body><div class="wrap">
<header>
  <h1>Market Regime Monitor<span>机构版市场状态监测</span></h1>
  <div class="meta">
    as of <b>{res['asof']}</b>（美股最近收盘）<br>
    样本 {coverage[0].strftime('%Y-%m-%d')} → {coverage[-1].strftime('%Y-%m-%d')}
    · {len(coverage)} 交易日 · {md.prices.shape[1]} 标的
  </div>
</header>

<div class="card">
  <div class="card-cap">A · Price + Regime Color</div>
  {panel_price(df, md)}
</div>

<div class="row-main">
  <div class="card">
    <div class="card-cap">B · Regime &amp; Confidence</div>
    {panel_regime_kpi(res)}
  </div>
  <div class="card">
    <div class="card-cap">C · Pillar Scores</div>
    {panel_pillars_bars(df)}
  </div>
</div>

<div class="card">
  <div class="card-cap">D · Final Regime Score History（五支柱加权合成）</div>
  {panel_regime_history(df)}
</div>

<div class="card">
  <div class="card-cap">E · Pillar Time Series（50 为中性）</div>
  {panel_pillar_history(df)}
</div>

<div class="card">
  <div class="card-cap">F · Explanation Layer</div>
  {panel_drivers(res, df, md)}
</div>

<footer>
  <b>方法</b>：regime_score = 0.25·trend + 0.20·breadth + 0.15·risk_on
  + 0.20·(100−vol) + 0.20·(100−crisis)；
  档位 80/60/40/20 对应 Strong Bull / Bull / Neutral / Bear / Panic。<br>
  <b>广度退化说明</b>：未取得标普 500 全成分股日行情，广度宇宙退化为
  {n_univ} 只宽基+行业 ETF 的截面统计（SPY/RSP/QQQ/IWM/XLK/XLF/XLE/XLV），
  方向信号有效但绝对数值精度低于全成分股口径。<br>
  <b>数据来源</b>：同花顺 iFinD 历史日行情（未复权收盘价），
  区间 {coverage[0].strftime('%Y-%m-%d')} 至 {coverage[-1].strftime('%Y-%m-%d')}。<br>
  <b>免责</b>：本文件为量化方法演示与市场状态描述，不构成任何投资建议、
  买卖推荐或收益承诺。历史统计规律不保证未来重现。
</footer>
</div></body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)
    return out_path


# ============================================================ PNG
def render_png(df: pd.DataFrame, md: MarketData, res: dict,
               out_path: str | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fam in ("Source Han Sans SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
                "Noto Sans CJK JP"):
        try:
            font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.family"] = fam
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = out_path or os.path.join(OUT_DIR, "dashboard.png")

    fig = plt.figure(figsize=(15.5, 13.2), facecolor="#f8f9fa")
    gs = fig.add_gridspec(5, 2, height_ratios=[1.16, 0.80, 1.00, 1.00, 0.34],
                          hspace=0.42, wspace=0.20,
                          left=0.058, right=0.972, top=0.925, bottom=0.032)

    fig.suptitle(f"Market Regime Monitor   ·   {res['asof']}   ·   "
                 f"{res['regime'].upper()} ({res['regime_cn']})   ·   "
                 f"Score {res['regime_score']:.1f}   ·   Conf {res['confidence']:.0f}%",
                 fontsize=17, fontweight="bold", x=0.055, ha="left", y=0.982)

    x = df.index
    regimes = df["regime"].values

    # ---- A 价格 + 档位背景
    ax = fig.add_subplot(gs[0, :])
    spy = md.series(EQUITY).reindex(x).ffill()
    ax.plot(x, spy.values, color="#1c1c1e", lw=1.5, zorder=5)
    n = len(regimes)
    start = 0
    for i in range(1, n + 1):
        if i == n or regimes[i] != regimes[start]:
            ax.axvspan(x[start], x[min(i, n - 1)],
                       color=REGIME_COLORS.get(regimes[start], "#adb5bd"),
                       alpha=0.16, lw=0, zorder=0)
            start = i
    ax.set_title("A · Price + Regime Color（SPY 收盘，背景为当日档位）",
                 fontsize=11.5, loc="left", color="#343a40")
    ax.set_ylabel("SPY", fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))

    # ---- B KPI 文本卡
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.set_title("B · Regime & Confidence", fontsize=11.5, loc="left", color="#343a40")
    colr = REGIME_COLORS.get(res["regime"], "#adb5bd")
    ax.add_patch(plt.Rectangle((0.02, 0.16), 0.96, 0.72, transform=ax.transAxes,
                               facecolor="#fbfbfc", edgecolor="#e9ecef", lw=1,
                               zorder=0))
    ax.add_patch(plt.Rectangle((0.02, 0.16), 0.012, 0.72, transform=ax.transAxes,
                               facecolor=colr, edgecolor="none", zorder=1))
    ax.text(0.07, 0.80, "CURRENT REGIME", fontsize=8.5, color="#868e96",
            transform=ax.transAxes)
    ax.text(0.07, 0.635, res["regime"].upper(), fontsize=25, fontweight="bold",
            color=colr, transform=ax.transAxes)
    ax.text(0.07, 0.475, res["regime_cn"], fontsize=11, color="#868e96",
            transform=ax.transAxes)
    ax.text(0.07, 0.235, f"{res['regime_score']:.1f}", fontsize=29,
            fontweight="bold", color="#212529", transform=ax.transAxes)
    ax.text(0.225, 0.235, "/100", fontsize=10.5, color="#adb5bd", transform=ax.transAxes)
    ax.text(0.60, 0.80, "CONFIDENCE", fontsize=8.5, color="#868e96",
            transform=ax.transAxes)
    ax.text(0.60, 0.50, f"{res['confidence']:.0f}%", fontsize=30,
            fontweight="bold", color=colr, transform=ax.transAxes)
    ax.text(0.60, 0.30, "支柱共识 × 边界距离 × 覆盖率", fontsize=7.8,
            color="#adb5bd", transform=ax.transAxes)

    # ---- C 支柱条
    ax = fig.add_subplot(gs[1, 1])
    row = df.iloc[-1]
    vals, colors, labels = [], [], []
    for key, col, lab in zip(PILLAR_KEYS, PILLAR_COLS, PILLAR_LABELS.values()):
        v = row.get(col, np.nan)
        vals.append(0.0 if pd.isna(v) else float(v))
        colors.append(PILLAR_COLOR_MAP[key])
        labels.append(lab.split()[0])
    ypos = np.arange(len(vals))[::-1]
    ax.barh(ypos, [100] * len(vals), color="#f1f3f5", height=0.55, zorder=1)
    ax.barh(ypos, vals, color=colors, height=0.55, zorder=2)
    for y_, v_ in zip(ypos, vals):
        ax.text(min(v_ + 2, 96), y_, f"{v_:.0f}", va="center", fontsize=10,
                fontweight="bold", color="#212529", zorder=3)
    ax.axvline(50, color="#868e96", ls="--", lw=0.9, alpha=0.7, zorder=4)
    ax.set_yticks(ypos, labels, fontsize=9.5)
    ax.tick_params(axis="y", pad=8)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=8.5)
    ax.set_title("C · Pillar Scores", fontsize=11.5, loc="left", color="#343a40")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # ---- D 总分历史
    ax = fig.add_subplot(gs[2, :])
    for lo_b, hi_b, name in REGIME_BANDS:
        ax.axhspan(max(lo_b, 0), min(hi_b, 100),
                   color=REGIME_COLORS[name], alpha=0.09, lw=0)
    ax.plot(x, df["regime_score"], color="#4c6ef5", lw=1.7, zorder=3)
    ax.fill_between(x, 50, df["regime_score"], where=df["regime_score"] >= 50,
                    color="#12b886", alpha=0.18, lw=0)
    ax.fill_between(x, 50, df["regime_score"], where=df["regime_score"] < 50,
                    color="#e03131", alpha=0.18, lw=0)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    for lo_b, hi_b, name in REGIME_BANDS:
        ax.text(0.988, (max(lo_b, 0) + min(hi_b, 100)) / 2, name,
                transform=ax.get_yaxis_transform(), fontsize=7.5,
                color="#868e96", va="center", ha="right")
    ax.set_title("D · Final Regime Score History", fontsize=11.5, loc="left",
                 color="#343a40")
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))

    # ---- E 支柱历史
    ax = fig.add_subplot(gs[3, :])
    for key, col in zip(PILLAR_KEYS, PILLAR_COLS):
        ax.plot(x, df[col], lw=1.35, label=PILLAR_LABELS[key].split()[0],
                color=PILLAR_COLOR_MAP[key], alpha=0.92)
    ax.axhline(50, color="#868e96", ls="--", lw=0.9, alpha=0.65)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("E · Pillar Time Series（50 = 中性）", fontsize=11.5, loc="left",
                 color="#343a40")
    ax.legend(ncol=5, fontsize=8.5, frameon=False, loc="lower left", bbox_to_anchor=(0, -0.02))
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))

    # ---- F 驱动
    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    ax.set_title("F · Explanation Layer — Drivers", fontsize=11.5, loc="left",
                 color="#343a40", pad=12)
    lines = []
    for p in res["drivers_positive"]:
        lines.append(("+", p, "#2b8a3e"))
    for q in res["drivers_negative"]:
        lines.append(("-", q, "#c92a2a"))
    if not lines:
        lines.append(("·", "各维度均处中性区间，无显著驱动", "#868e96"))
    # 用色块作标记，避免依赖字体中的 +/- 字形（个别 CJK 字体缺字形会显示方块）
    from matplotlib.patches import Rectangle
    shown = lines[:8]
    n = max(1, len(shown))
    step = min(0.22, 0.92 / n)
    y = 0.97 - step * 0.5
    for _sign, text, colr2 in shown:
        ax.add_patch(Rectangle((0.006, y - 0.022), 0.010, 0.044,
                               transform=ax.transAxes, facecolor=colr2,
                               edgecolor="none", clip_on=False))
        ax.text(0.026, y, text, fontsize=10, color="#343a40",
                transform=ax.transAxes, va="center")
        y -= step

    with open(os.path.join(OUT_DIR, ".png_meta"), "w") as f:
        f.write(datetime.now().isoformat())
    fig.savefig(out_path, dpi=145, facecolor="#f8f9fa", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    return out_path
