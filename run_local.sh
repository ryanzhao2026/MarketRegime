#!/usr/bin/env bash
# 一键本地运行：创建虚拟环境 → 安装依赖 → 生成仪表盘
# 用法： ./run_local.sh
set -euo pipefail

cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "==> 创建虚拟环境 .venv"
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> 安装依赖"
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "==> 运行 Market Regime Monitor"
python main.py --export-csv

echo
echo "==> 完成。产物在 outputs/"
echo "    仪表盘： outputs/dashboard.html   （双击用浏览器打开）"
echo "    预览图： outputs/dashboard.png"
echo "    报告：   outputs/report.txt"
