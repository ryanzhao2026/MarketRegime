# Market Regime Monitor

一个机构风格的**市场状态（Regime）监测器**：特征工程 → 五引擎打分 → 解释层 → 可视化 Dashboard。

不是"预测涨跌"，而是回答三个问题：**现在是什么市？有多确定？靠什么在撑、被什么拖？**

---

## 目录结构

```
MarketRegime/
├── data.py                 数据加载、标的角色定义、时点截断（无前视）
├── features.py             指标计算与归一化工具（纯函数，无状态）
├── trend_engine.py         趋势引擎    权重 25%
├── breadth_engine.py       广度引擎    权重 20%
├── volatility_engine.py    波动率引擎  权重 20%（以 100-vol 反向入分）
├── cross_asset_engine.py   跨资产引擎  权重 15%
├── risk_engine.py          风险引擎    权重 20%（以 100-crisis 反向入分）
├── scoring.py              合成打分、档位映射、置信度、解释层
├── dashboard.py            HTML（内联 SVG）+ PNG 两种渲染
├── main.py                 CLI 入口
├── data/
│   ├── market_data.csv     15 个标的日收盘价，2024-01-02 → 2026-11-13
│   └── spy_ohlc.csv        SPY 的 OHLC（ADX / ATR 需要真实高低价）
└── outputs/
    ├── dashboard.html      自包含单文件仪表盘（无外部依赖）
    ├── dashboard.png       静态预览
    ├── report.txt          文本报告
    └── regime_history.csv  完整因子矩阵（可自行做归因分析）
```

## 运行

```bash
python3 main.py                     # 最新数据出报告 + 双形态仪表盘
python3 main.py --asof 2026-10-31   # 指定时点复算（只用该日及以前数据）
python3 main.py --no-png            # 跳过 PNG（无 matplotlib 环境）
python3 main.py --export-csv        # 额外导出因子矩阵
```

依赖：`pandas`、`numpy`、`matplotlib`（仅 PNG 需要）。

---

## 打分模型

```python
regime_score = 0.25 * trend_score
             + 0.20 * breadth_score
             + 0.15 * risk_on_score
             + 0.20 * (100 - vol_score)
             + 0.20 * (100 - crisis_score)
```

档位映射（分数为左闭右开区间）：

| 区间 | 档位 | 中文 |
|---|---|---|
| 80–100 | Strong Bull | 强势牛市 |
| 60–80 | Bull | 牛市 |
| 40–60 | Neutral | 中性震荡 |
| 20–40 | Bear | 熊市 |
| 0–20 | Panic | 恐慌 |

**缺失处理**：任一引擎因数据不足返回 NaN 时，该支柱自动从加权分母中剔除、其余权重重归一化，而不是用 0 或 50 填充。这样"缺数据"不会被误读成"中性"。

### 各引擎因子（全部输出 0~100）

| 引擎 | 因子 | 权重 |
|---|---|---|
| **Trend** | ADX(14)·方向 | 0.25 |
| | Trend Efficiency (Kaufman, 20d) | 0.20 |
| | EMA200 Distance | 0.20 |
| | Multi-Filter Agreement（5 条件投票） | 0.20 |
| | Slope Consistency（60d 内 20d 斜率为正占比） | 0.15 |
| **Breadth** | SPY > MA200 占比 | 0.30 |
| | Advance / Decline（20d 平滑） | 0.25 |
| | RSP / SPY 相对强弱 | 0.25 |
| | New Highs − New Lows | 0.20 |
| **Volatility** | VIX 水平 | 0.35 |
| | VVIX | 0.20 |
| | Realized Vol (21d 年化) | 0.25 |
| | Volatility Expansion（5d/63d） | 0.20 |
| **Cross Asset** | SPY / TLT（股债） | 0.30 |
| | HYG / TLT（信用 vs 久期） | 0.30 |
| | GLD / SPY（避险，**反向**计分） | 0.20 |
| | DXY 趋势（美元，**反向**计分） | 0.20 |
| **Risk** | Skew (63d) | 0.25 |
| | Kurtosis (63d) | 0.15 |
| | Drawdown (vs 252d 高) | 0.25 |
| | Correlation Regime（60d 两两均值） | 0.20 |
| | Sector Dispersion（XLK/XLF/XLE/XLV） | 0.15 |

**打分设计要点**：每个因子先算原始量，再用「绝对刻度分段映射」与「252 日滚动分位」按比例混合（多数子因子取 70/30）。
- 绝对刻度保证分数**可解释**（VIX=19 就是 19 对应的那一档，不因样本期改变）
- 滚动分位保证**自适应性**（牛市里 VIX 15 也该偏高，熊市里 VIX 25 未必极端）

### 置信度（Confidence）

```python
confidence = 0.50 * 共识度 + 0.30 * 边界距离 + 0.20 * 数据覆盖
```
- **共识度**：五支柱彼此有多接近（各自对均值的离散度越小越高）
- **边界距离**：离最近的档位分界（20/40/60/80）有多远；贴边则低
- **数据覆盖**：有效支柱占比

⚠️ 这是**模型内部一致性**的度量，**不是预测准确率**。分数 62、置信度 45% 的正确读法是"略偏多，但这个结论很贴边、支柱之间分歧大"。

---

## 解释层（本项目的重点）

机构真正关心的不是 `Regime Score = 73`，而是"73 靠什么撑起来、哪里在漏水"。所以每个结论都带三条信息：**档位 / 置信度 / 贡献归因**。

贡献度定义为 `(支柱得分 − 50) × 权重`，即相对中性状态的加权偏离（单位 pp），五项之和 = `regime_score − 50`：

```
Pillar Breakdown
支柱                      得分      权重       贡献
广度 Breadth            64.2     20%     +2.8
跨资产 Risk-On           59.4     15%     +1.4
趋势 Trend              42.7     25%     -1.8
安全性 Safety            40.8     20%     -1.8
低波动 Stability         37.5     20%     -2.5
```

正负驱动由两套规则产生：先按绝对阈值（如 trend ≥ 70 / ≤ 30）抓"显著项"，再用贡献排序补齐到至少各 2 条，避免只输出一条驱动。

---

## 当前读数（as of 2026-11-13，美股最近收盘）

```
Current Regime :  NEUTRAL  (中性震荡)
Regime Score   :  48.1 / 100
Confidence     :  88%

Drivers
  + 广度 相对占优（64/100，贡献 +2.8pp）
  + 跨资产 相对占优（59/100，贡献 +1.4pp）
  - 波动率抬升（63/100，VIX 19.3）
  - 尾部风险累积（59/100，明显左偏）

趋势 ADX=14.2  距EMA200=+2.7%  效率=0.06
广度 站上MA200 5/8  上涨占比=55.6%
波动 VIX=19.3  RV=21.8%  扩张=1.37x
跨资产 SPY/TLT=+1.1%  HYG/TLT=-0.5%  DXY=+3.8%
风险 偏度=-1.41  回撤=-3.2%  相关=0.40
```

一个自洽的市场画像：指数仍在 EMA200 上方 2.7%，但 **ADX 14.2 / 效率 0.06 = 明确的震荡无趋势**；广度还行（5/8 站上 MA200）但信用端 HYG/TLT 转弱、美元 63 日走强 3.8%（流动性收紧），波动率在扩张（5d/63d = 1.37x）且收益分布明显左偏。**"指数不难看，但内部正在变脆"** —— 这正是只报一个 48 分会丢掉的信息。

---

## 已知数据限制（重要）

1. **VVIX 不可用**。iFinD 仅返回 `VVIX.TI`，该序列在 21–30 区间，而真实 VVIX（VIX 期权的隐含波动率）长期量级为 70–150，约为 VIX 的 4–6 倍。数值量级严重不符，无法核实其真实身份，**因此不采用**。引擎内置合理性校验（中位数 < 50 即剔除该因子并重归一化），当前 vol_score 实际由 VIX / 已实现波动 / 波动率扩张三项加权得出。修复方式：接入 CBOE 官方 VVIX 历史序列。

2. **VIX 期限结构缺失**。`VIX3M` 代码解析为空，无法计算 VIX/VIX3M 比值（期货升贴水结构是判断恐慌是否见顶的关键信号）。这是你列出的"下一期优先升级项"之一，目前未落地。

3. **广度退化**。未取得标普 500 全成分股日行情，广度宇宙退化为 8 只 ETF 的截面统计（SPY/RSP/QQQ/IWM/XLK/XLF/XLE/XLV）。**方向信号有效，但绝对数值精度低于全成分股口径**——例如「站上 MA200 5/8」不等于「标普 500 有 62.5% 的成分股站上 MA200」。修复方式：接入全成分股或使用指数官方的 % above MA200 数据。

4. **价格未复权**。使用未复权收盘价。ETF 分红对 MA200、63 日比值等中短周期指标影响有限，但长周期趋势判断会略有低估。若用于正式归因，建议改为复权价。

5. **Sector Dispersion 的方向假设**。"行业分化加剧 → 风险上升"是一种建模选择（轮动失序 = 领导力不稳定），并非唯一解读；在强主题行情下分化加剧也可能是健康的。

6. **样本期**。2024-01-02 → 2026-11-13 共 742 个交易日，仅覆盖一轮上行周期 + 若干次回调，未经历完整熊市，因此滚动分位的极值端（如 Score < 20 的 Panic 区间）在样本内从未触发。**档位阈值本身未经跨周期校准**。

---

## 下一期优先升级（按解释力提升排序）

1. **VIX 期限结构**（VIX / VIX3M）—— 恐慌见顶的经典信号，目前最大缺口
2. **HYG/TLT 换成真实信用利差**（如 ICE BofA HY OAS）—— 比 ETF 比值更干净，去掉利率久期的干扰
3. **全成分股广度** —— 替换 ETF 代理，% above MA200 与 NH-NL 才真正可比
4. **Sector Rotation 相对强弱矩阵** —— 从"分化度"升级到"轮动方向"
5. **DXY 之外补流动性指标**（如美联储准备金、TGA）—— 美元只是代理

技术指标侧（EMA/HMA/TwoPole 等滤波器换参）对解释力的边际贡献确实远低于上述跨资产与广度因子，这也是本项目的排序依据。

---

## 免责声明

本项目为**量化方法演示与市场状态描述**，不构成任何投资建议、买卖推荐、目标价或收益承诺。所有分数均为基于历史价格的规则化统计输出，不含对未来走势的预测能力。历史统计规律不保证未来重现。数据可能存在缺失、延迟或口径偏差，使用前请自行核对。

数据来源：同花顺 iFinD 历史日行情（未复权收盘价），区间 2024-01-02 至 2026-11-13。

---

# 本地运行 / GitHub 部署

两条路径都只依赖 Python，不需要数据库、不需要 API Key（行情已内置在 `data/*.csv`）。

## 一、本地运行

### 方式 A：一键脚本（推荐）

```bash
cd MarketRegime
chmod +x run_local.sh
./run_local.sh
```

脚本会创建 `.venv`、装依赖、跑完整流程。产物在 `outputs/`：

```
outputs/dashboard.html     ← 双击用浏览器打开（自包含，可单独发人）
outputs/dashboard.png      ← 静态预览图
outputs/report.txt         ← 文本报告
outputs/regime_history.csv ← 完整因子矩阵（742 行 × 50+ 列）
```

### 方式 B：手动三步

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --export-csv
```

### 方式 C：只想看结果，不装 matplotlib

```bash
python main.py --no-png
```
只生成 HTML（内联 SVG，不依赖 matplotlib），此时 `requirements.txt` 里的 matplotlib 可以不装。

### 常用参数

| 命令 | 作用 |
|---|---|
| `python main.py` | 最新数据出报告 + 双形态仪表盘 |
| `python main.py --asof 2026-10-31` | 指定时点复算，只用该日及以前的数据（无前视） |
| `python main.py --no-png` | 跳过 PNG |
| `python main.py --export-csv` | 额外导出因子矩阵 |

### 本地常见问题

**1. PNG 里中文变方块 / 乱码**
matplotlib 找不到中文字体。Linux：

```bash
sudo apt-get install -y fonts-noto-cjk
```

macOS / Windows 一般自带苹方、微软雅黑，无需处理。
`dashboard.py` 已按 `Source Han Sans SC → Noto Sans CJK SC → WenQuanYi Zen Hei` 顺序自动查找可用字体。

**2. `ModuleNotFoundError: No module named 'pandas'`**
没激活虚拟环境。先 `source .venv/bin/activate` 再运行。

**3. 想换成自己的数据**
把 `data/market_data.csv` 换成同格式即可：第一列 `date`（YYYY-MM-DD），其余每列一个标的收盘价。列名需与 `data.py` 里的常量一致（`SPY`/`RSP`/`QQQ`/`IWM`/`TLT`/`HYG`/`GLD`/`DBC`/`DINI`/`VIX`/`VVIX`/`XLK`/`XLF`/`XLE`/`XLV`）。
缺列不会报错——对应引擎会自动降级并把权重重归一化，这是设计上的容错。

---

## 二、部署到 GitHub Pages

仓库里已带好 `.github/workflows/deploy-pages.yml`，推上去就能自动部署。

### 步骤

**1. 建仓库并推送**

```bash
cd MarketRegime
git init
git add .
git commit -m "Market Regime Monitor"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

**2. 开启 Pages**

仓库 → **Settings** → **Pages** → **Source** 选 **GitHub Actions**（不是 “Deploy from a branch”）。

**3. 等第一次构建**

**Actions** 标签页会看到 `Build & Deploy Dashboard` 在跑。约 1–2 分钟后：

```
https://<你的用户名>.github.io/<仓库名>/
```

就是你的在线仪表盘。

### 工作流做了什么

| 触发条件 | 说明 |
|---|---|
| push 到 `main` | 改代码自动重建 |
| 手动 Run workflow | Actions 页面点按钮随时重建 |
| 定时 `30 22 * * 1-5` | 每交易日 UTC 22:30（美东收盘后）自动刷新 |

流程：装 Python 3.11 → 装依赖 → 装 Noto CJK 字体 → 跑 `main.py` → 把 `dashboard.html` 复制成 `index.html` → 发布到 Pages。
同时会把文本报告打进 Actions 日志，不开网页也能在构建记录里看到当天读数。

### 几个注意点

**① 定时任务的时间是 UTC**
`30 22 * * 1-5` = 北京时间次日 06:30。想改时间就改 `.github/workflows/deploy-pages.yml` 里的 cron。
注意 GitHub 的定时任务是**尽力而为**，高峰期可能延迟几分钟到十几分钟，不适合对时间精度敏感的场景。

**② `.gitignore` 默认不提交 outputs/**
产物由 CI 每次重新生成，不入库可避免仓库被二进制文件撑大。
但如果你想让别人 clone 下来直接看到结果，删掉 `.gitignore` 里 `outputs/*` 那几行，或手动 `git add -f outputs/`。

**③ 数据每日自动更新**
工作流中已内置取数步骤 `python fetch_data.py`，每个交易日 UTC 22:15（美股收盘后）自动抓最新行情再重算指标。
数据源与覆盖见下方「每日自动更新」章节。若数据源临时故障，该步骤标记为 `continue-on-error`，
构建会用仓库中已提交的历史快照继续出图，站点不会挂掉——但当天数字不会变，可在 Actions 日志里看到 `fetch outcome: failure`。

**④ 私有仓库也能用 Pages**
但需要 GitHub Pro/Team；公开仓库免费。

**⑤ 不想用 Pages，只想存文件**
把 workflow 里 `deploy` job 删掉，只留 `build`，结尾加一步 `actions/upload-artifact@v4` 把 `_site` 传成构建产物即可下载。

---

## 三、目录里新增的部署文件

```
MarketRegime/
├── requirements.txt                依赖声明
├── .gitignore                      忽略缓存与运行产物
├── run_local.sh                    本地一键运行
└── .github/workflows/
    └── deploy-pages.yml            构建 + 部署到 GitHub Pages
```


---

# 每日自动更新（数据层）

## 为什么 CI 不能用 iFinD

iFinD 是通过 MCP 连接器接入的**本地工具**，GitHub Actions 的 runner 无法调用它。
CI 必须能在无凭据、无人值守的环境下自取数据，因此 `fetch_data.py` 以 **yfinance 为主源**，并配置三级自动回退，全部无需 API Key。

### 五级取数架构（v2：主源 + 三层官方回退 + 快照）

| 优先级 | 来源 | 覆盖序列 | 说明 |
|---|---|---|---|
| ① 主源 | **yfinance**（Yahoo Finance） | ETF + 5 个波动率指数 + 美元 | 一个库全覆盖，代码最简洁 |
| ② 回退 | Nasdaq 官方行情 `api.nasdaq.com` | 12 只 ETF | yfinance 失败时接管 OHLCV |
| ③ 回退 | CBOE 官方指数历史 `cdn.cboe.com` | VIX / VVIX / **VIX9D / VIX3M / VIX6M** | 指数权威源，含全历史 |
| ④ **新增** | **FRED 官方宏观** `fred.stlouisfed.org` | **BAA10Y / HY OAS / NFCI / WALCL / RRP / DTWEXBGS** | 免 API Key，`fredgraph.csv` 端点 |
| ⑤ 兜底 | 仓库内既有 `data/market_data.csv` | 全部 | 所有网络源均失败时沿用快照 |

回退按**列**粒度逐列判定，某个标的在某一层取到就跳过后续层。
每次运行结束会打印**数据源台账**（每个标的实际来自哪一层），并写入
`data/fetch_report.json`：

```json
{
  "latest_trading_day": "2026-11-13",
  "per_column_source": { "SPY": "yfinance", "CREDIT_SPREAD": "fred:BAA10Y", ... },
  "degraded_columns": ["CREDIT_SPREAD", "NFCI", ...],
  "primary_source_hit": true
}
```

**`degraded_columns` 是关键字段**：任何未由主源提供的列都会列在这里。
这样「降级」永远不会被静默吞掉——悄悄用旧数据是最危险的失败模式。

### 取数起点前推到 2006

`DEFAULT_START = "2006-01-03"`，覆盖 2008 金融危机、2020 熔断、2022 熊市
三个完整压力期。目的不是数据更多，而是让**滚动分位与档位阈值第一次具备
跨周期含义**：在 2024 起点的样本里，2024 年前 252 天的分位几乎无历史可参照。

若长历史下载失败（yfinance 限流），回退层仍会给出近期数据，站点不会空白，
但 `fetch_report.json` 会如实反映最新交易日与降级列。

### 为什么必须有回退：两处实测结论

同一个 `fetch_data.py`，在两个不同网络环境下跑出了**完全相反**的主源表现。这正是保留四级回退的理由。

**场景 A：本地沙箱（开发环境）—— yfinance 被限流，回退层救场**

- 直接 `yfinance.download()`：**100% 返回 `YFRateLimitError`（HTTP 429）**
- 用 `curl_cffi` 伪装 Chrome 浏览器指纹：**同样 429**，无法绕过
- 实际供数：12 只 ETF ← Nasdaq，VIX / VVIX / VIX3M ← CBOE，DXY ← UUP 代理

**场景 B：GitHub Actions runner（CI 环境）—— yfinance 全部成功**

- `[源 1] yfinance（批量）✓ 成功 16 列`：ETF、VIX、VVIX、VIX3M、DXY **全部来自 yfinance**，最新日 2026-11-13
- 同一时刻的连通性预检：`nasdaq HTTP 000`（不可达），`cboe HTTP 200`
- 结论：**CI 环境里 Nasdaq 反而是不通的那一个**，yfinance 正常

**这两次实测说明什么**

- 限流是**按 IP 段与时段变化**的，不能靠单次测试断定某个源"一定可用"或"一定不可用"。
- 单一数据源随时可能整体失效（本地 Yahoo 挂、CI 里 Nasdaq 挂），**回退层因此不是可选优化，而是每日更新能长期跑通的前提**。
- CI 里 2026-11-15 的实际台账为 16/16 全部 ← yfinance，即主源正常工作；一旦将来 yfinance 在 CI 也被限流，代码会自动落到 Nasdaq / CBOE，无需改动。

## 运行

```bash
python fetch_data.py                    # 增量更新到最近交易日
python fetch_data.py --start 2024-01-01 # 指定起始日全量重建
python fetch_data.py --out-dir data     # 自定义输出目录
```

产出与原有格式完全一致（宽表，`date` 为索引，其余每列一个标的），因此 `main.py` 无需改动。
抓取失败的单个标的会沿用已有历史列并打印警告；只有 SPY 失败才整体退出（避免产出空站点）。

## 这一步带来的两个实质升级

**1. VVIX 从「缺失」变成「可用」**

之前 iFinD 只返回 `VVIX.TI`，数值 21–30，与真实 VVIX 量级（70–150）严重不符，被合理性校验剔除。
改用 CBOE 官方源后拿到真实值 **101.53**，量级正确，已通过校验并正式进入 `vol_score`。

**2. VIX 期限结构落地（你列为第一优先的升级项）**

`VIX3M` 已可获取，波动率引擎新增第 5 个因子：

```
term_structure = VIX / VIX3M
```

- 比值 < 1（contango，近月低于远月）= 市场平静，低分
- 比值 > 1（backwardation，近端被抢高）= 恐慌见顶的经典特征，高分

当前读数 19.3 / 20.6 ≈ **0.94**，处于温和 contango。

波动率引擎权重相应调整为：`VIX 0.30 / VVIX 0.15 / 已实现波动 0.20 / 波动扩张 0.15 / 期限结构 0.20`（合计 1.00）。

## v2 模型升级（2026-11）

v2 **不改动五支柱权重**（趋势 25 / 广度 20 / 波动 20 / 跨资产 15 / 风险 20），
新因子一律作为**既有支柱的子因子**注入。这一点是硬约束：

- 权重不变 ⇒ 同一日期的 `regime_score` 在 v1 与 v2 之间保持可比，
  分数变化能归因到「市场变了」而非「模型变了」；
- 若新增第六个支柱并重新分配权重，历史分数序列会断裂，
  解释层的「支柱贡献」也失去跨期可比性。

解释层与每日归档都带 `model_version` 字段，未来真正需要改权重时，
断点在哪天一目了然。

### 三处因子级升级

| 支柱 | v1 | v2 | 变化理由 |
|---|---|---|---|
| 波动 20% | VIX/VIX3M 两点 | **VIX9D+VIX / VIX3M+VIX6M 四端** | 两点比值噪声大，四端平均更稳 |
| 跨资产 15% | HYG/TLT 比值 | **Baa-10Y 信用价差**（FRED） | HYG 与 TLT 都含久期，比值变化混入利率因素；价差已剥离国债基准 |
| 风险 20% | 5 因子 | **+ 流动性（NFCI/WALCL/RRP）** | 「钱在不在系统里」是危机的前置条件 |

风险引擎为容纳新因子，原五因子按比例让渡权重：
`skew 0.20 / kurtosis 0.12 / drawdown 0.22 / correlation 0.18 / dispersion 0.13 / liquidity 0.15`。

### 阈值不是拍脑袋：全部由实测分位标定

**期限结构**（CBOE 四端，2011-01-04 → 2026-11-13，3990 个交易日）：

```
p1=0.641  p5=0.681  p10=0.705  p25=0.752  p50=0.817
p75=0.902 p90=0.994 p95=1.062  p99=1.298
```

**信用价差 Baa-10Y**（FRED，2006-01-03 → 2026-11-12，5215 个交易日）：

```
水平   p1=1.44 p5=1.55 p25=1.83 p50=2.25 p75=2.85 p95=3.46 p99=5.68
       2008-12 峰值 6.16 | 2020-03 峰值 4.31 | 2022-10 峰值 2.37
63日变化 p5=-0.51pp  p50=-0.04pp  p95=+0.58pp  p99=+2.09pp
```

### 回归验证：极端期必须打到高分

用 CBOE 的 VIX 全历史（9316 个交易日）对波动率因子做独立回归：

| 时段 | 平均 VIX | 因子分（均值 / 峰值） |
|---|---|---|
| 2008-10 雷曼后 | 61.18 | **99.2 / 100.0** |
| 2008-11 峰值 | 62.67 | 98.5 / 100.0 |
| 2020-03 熔断 | 57.74 | 97.9 / 100.0 |
| 2022-10 熊市低点 | 30.01 | 80.7 / 89.6 |
| 2017-06 低波平静期 | 10.51 | **4.1 / 11.4** |

全样本分布 p5=7 / p50=39 / p95=87，因子分 >80 的交易日占 8.9%。
这说明该因子在危机期确实饱和于高位、在平静期贴近零，未出现「阈值定得太松
导致 2008 只有 60 分」这类常见的标定失败。

### 整段打分回归：Panic 档位终于能被触发

用 5250 个交易日（2006-01-03 → 2026-11-13）的完整面板逐日打分，
这是 v1 做不到的验证——v1 样本只有 721 天、且全在一轮上行周期内，
**Panic 区间从未触发，等于该档位从未被检验**：

| 时段 | 交易日 | 平均分 | 区间 | 档位分布 |
|---|---|---|---|---|
| 2008-09~12 金融危机 | 85 | 24.8 | 16.3 / 44.7 | **Bear 56 · Panic 25** · Neutral 4 |
| 2020-02~04 熔断 | 62 | 33.3 | 14.6 / 66.1 | **Bear 32 · Panic 14** · Bull 10 · Neutral 6 |
| 2022-01~10 熊市 | 209 | 43.5 | 28.5 / 69.1 | Neutral 106 · Bear 93 · Bull 10 |
| 2015-08~2016-02 调整 | 145 | 41.9 | 23.8 / 60.1 | Neutral 72 · Bear 72 · Bull 1 |
| 2017 低波牛市 | 251 | 68.7 | 52.3 / 79.6 | **Bull 221** · Neutral 30 |
| 2024-01~2026-11 | 721 | 61.6 | 23.7 / 77.0 | Bull 476 · Neutral 211 · Bear 34 |

全样本档位占比：`Bull 53.4% / Neutral 32.8% / Bear 12.5% / Panic 0.9% / Strong Bull 0.4%`
分数分位：`p5=30.6 / p25=50.0 / p50=61.5 / p75=68.6 / p95=74.6`

三点值得注意：

1. **2008 与 2020 确实打进了 Panic**（25 天 / 14 天），不再是"永不触发"的死档位。
2. **2017 低波牛市 84% 的交易日为 Bull**，而 2008 危机 96% 落在 Bear/Panic——
   两端的判别力是对称的，不是"永远偏空"或"永远偏多"。
3. 2020-03 区间内出现 10 天 Bull，原因是熔断前的 2 月仍处高位，
   属区间切分效应而非模型错误；但这确实提示**单日读数比区间均值更能反映当时的处境**，
   也是加入每日归档的原因之一。

需要注意的偏差：全样本 Bull 占 53.4%，明显高于 Bear+Panic 的 13.4%。
这是因为 2006-2026 整体是长期上行市，属样本固有特征，
不应解读为"模型系统性偏乐观"——真正的检验是危机期的绝对分（16~33 分），
而这一项已达标。

### 两个必须说明的取舍

**1. 信用为什么用 Baa-10Y，而不是高收益 OAS**

ICE BofA 高收益 OAS（`BAMLH0A0HYM2`）更贴近高收益债，专业度更高，
但免费端点**只能回溯约 3 年**（受授权限制），无法覆盖 2008/2020 压力期。
v2 的首要目标是把历史拉到 2006 以做跨周期标定，故以 **BAA10Y 为主口径**，
同时把 HY_OAS 一并取回作交叉验证列。代价是 Baa 级偏投资级，
对高收益段的极端恶化不如 OAS 灵敏——这是明确知情的选择，不是疏漏。

**2. 期限结构在 2008 段为空**

VIX9D 自 2011 年才有数据，2008 年段的四端曲线不存在。
引擎按「**缺失即重归一化权重**」处理，**不做前向填充**：
宁愿承认该因子当时不可测，也不造假一条曲线。因此 2008 年段的
`vol_score` 实际由其余四个子因子构成，权重会自动重分配。

## 代理的合理性（重要披露）

美元指数在 v2 中**优先使用 FRED 的官方广义贸易加权美元指数**
（`DTWEXBGS`，美联储口径），已不再是 UUP 代理。

仅当 FRED 不可用时，才依次退回 ICE DXY（yfinance `DX-Y.NYB`）→
旧 DINI 列 → UUP 代理，并在 `credit_source` / `dollar_source` 字段
与仪表盘指标名中标注实际口径。

**UUP 代理为什么仍可接受**（作为最后兜底）：

**为什么这个替代是可接受的**：跨资产引擎只使用美元的两个特征——
1. 63 日变化率（`chg = DXY / DXY.shift(63) - 1`）
2. 与 50 日均线的关系（`DXY > MA50`）

两者都只依赖**相对变化与趋势方向**，对价格绝对水平不敏感。UUP 跟踪 ICE DXY 的日度收益，因此上述两个特征与 DXY 指数高度一致。
但如果将来要让美元因子参与「绝对水平」判断（如判断美元是否处于历史高位），必须换回真实的 DXY 指数序列。

## 局限与后续

- `fetch_data.py` 依赖第三方公开接口的可用性。yfinance / Nasdaq / CBOE / FRED 若变更结构或限流，取数会按五级架构逐级回退，最终回退到历史快照。
- **长历史能否建立取决于 CI 网络**：本地沙箱实测 yfinance 429、Yahoo chart API 429、Stooq 被 JS 挑战拦截，因此 2006 起的长历史只能由 CI 在运行时建立。若 CI 也被限流，数据会停在快照日期，`fetch_report.json` 的 `latest_trading_day` 会如实暴露这一点。
- 各数据源可用性随 IP 段与时段波动：本地沙箱实测 yfinance 429 而 Nasdaq 可用；GitHub Actions 实测 yfinance 全部成功而 Nasdaq 不可达。**回退层的可用性决定了每日更新是否成功**，建议每次运行后查看日志中的数据源台账（`SPY ← yfinance` 这类逐列标注）。
- CBOE 指数 CSV 是**全量历史**（几十年），每次都会完整下载；如需减少流量可在 `fetch_cboe()` 里加区间裁剪。
- GitHub 定时任务是**尽力而为**：高峰期可能延迟数十分钟，不适合对时间精度敏感的场景。
- 仍未解决：全成分股广度（当前仍为 8 只 ETF 代理）、价格复权。
