#!/bin/bash
# 全量结构/成本基准。两条腿 × 三系统，但 raw_xml 在大 LaTeXML 论文上批次爆炸（一篇 50-220 批），
# 故 contract+naive 跑全量、raw_xml 只跑代表性子集（基线足矣）。MM = 跨厂商多模型 contract。
# 用法: bash scripts/run_benchmarks.sh [main|rawxml|mm|diff|all]
#   main = contract+naive 全量（两腿）   rawxml = raw_xml 子集   diff = main+rawxml   mm = 跨厂商
set -e
cd "$(dirname "$0")/.."
MODE="${1:-all}"
CN="https://dashscope.aliyuncs.com/compatible-mode/v1"
MM_MODELS="qwen3.7-plus-nothink,qwen3-max-2026-01-23,glm-4.7,kimi-k2.5,minimax-m2.5"
BM() { DASHSCOPE_OPENAI_BASE_URL="$CN" python3 -m xinda.cli.jats_benchmark "$@"; }

run_main() {   # contract+naive 全量，并发6（qwen-plus RPM600 有余量；这俩系统批次正常、快）
  BM --corpus-dir corpus/arxiv_en2zh --lang zh --format ltx \
     --systems contract,naive --models qwen-plus --concurrency 6 --out-dir results/arxiv_en2zh/main
  BM --corpus-dir corpus/jats_zh2en --lang en --format jats \
     --systems contract,naive --models qwen-plus --concurrency 6 --out-dir results/jats_zh2en/main
}
run_rawxml() { # raw_xml 仅子集（arXiv 15 篇代表性 + JATS 全 25 篇），并发4（raw_xml 重）
  BM --corpus-dir corpus/arxiv_en2zh_rawsub --lang zh --format ltx \
     --systems raw_xml --models qwen-plus --concurrency 4 --out-dir results/arxiv_en2zh/rawxml
  BM --corpus-dir corpus/jats_zh2en --lang en --format jats \
     --systems raw_xml --models qwen-plus --concurrency 4 --out-dir results/jats_zh2en/rawxml
}
run_mm() {     # coding 端点跨厂商，并发2（429 靠 SDK 重试吸收）
  python3 -m xinda.cli.jats_benchmark --corpus-dir corpus/jats_zh2en --lang en --format jats \
    --systems contract --models "$MM_MODELS" --concurrency 2 --out-dir results/jats_zh2en/mm
  python3 -m xinda.cli.jats_benchmark --corpus-dir corpus/arxiv_en2zh --lang zh --format ltx \
    --systems contract --models "$MM_MODELS" --concurrency 2 --out-dir results/arxiv_en2zh/mm
}
case "$MODE" in
  main)   run_main ;;
  rawxml) run_rawxml ;;
  diff)   run_main; run_rawxml ;;
  mm)     run_mm ;;
  all)    run_main; run_rawxml; run_mm ;;
esac
echo "DONE ($MODE)"
