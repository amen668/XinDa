# XinDa（信达）— Fidelity-First Scholarly Document Translation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-required-336791.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-supported-2496ED.svg)](docker-compose.yml)

**中文** | [English](README.en.md)

**信达**取自严复"信、达、雅"：**信（保真）为先，达（全文可读）随后**。本框架面向科技论文的
全文机器翻译与质量评测，核心命题是同时拆掉全文多语种出版的三个工程瓶颈：

- **成本**（P1）：国产非前沿模型 + 上下文缓存/批处理，单篇全文翻译约 ¥0.03；
- **结构**（P2）：一套**占位符契约**（placeholder contract）把公式、引文、交叉引用等不可翻译
  的行内元素在送模前令牌化、回填时逐字还原——结构保留是**构造性保证**，不是概率尽力；
- **质检**（P3）：保真核查（事实锚点 + 跨语言比较验证器）+ 门控式选择性人工复核，把逐句
  人工校验压缩到少数被标记单元。

同一契约驱动两种学术 XML 方言：**LaTeXML**（arXiv LaTeX）与 **JATS**（期刊出版标准），
即同时支持 arXiv en→zh 与中文期刊 JATS zh→en 两条翻译腿。

## 架构一览

13 阶段、可断点续跑的流水线（Postgres 记录状态，重跑同一 `job_id` 自动从断点恢复）：

```
Acquire → Convert → Extract → FactExtract → GlossaryBuild → FirstPassTranslate
  → FactVerify → CrossDocFactVerify → Refine → Coherence → ApplyXML → Render → Evaluate
```

评测套件（`xinda/evaluation/`）：结构保真 PPA/MFR（含序敏感变体）、事实保真 FPS/事实陷阱、
跨语言比较验证器、COMET-Kiwi/xCOMET 神经 QE、QA 理解度（RCS）、术语一致性（TCR）、
LLM 判官（RUBRIC-MQM / G-Eval）+ 判官元评测、成本核算、门控人工复核分诊。

## 快速开始

推荐 Docker（宿主机装 LaTeXML/torch 较繁琐）：

```bash
cp .env.example .env        # 填入 DASHSCOPE_API_KEY 等
docker compose up -d db     # Postgres，schema 自动应用
docker compose run --rm app python -m xinda.cli.translate_smoke 2503.15129 zh
# 神经 QE 需要 GPU，在单独的 qe 服务里：
docker compose run --rm qe python -m xinda.cli.neural_qe <job_id>
```

DB-free 的 JATS 单篇翻译与结构基准：

```bash
python -m xinda.cli.jats_translate <jats.xml> en          # 整篇 JATS 翻译
python -m xinda.cli.jats_benchmark --corpus-dir corpus/jats_zh2en --lang en \
    --format jats --systems contract,naive --out-dir results/demo
```

更多 CLI 入口见 `xinda/cli/`（语料构建、许可过滤、基准矩阵、判官元评测等）。

## 配置

`xinda/config.py`（Pydantic BaseSettings，读 `.env`，见 `.env.example`）。无任何硬编码密钥。
关键变量：`DASHSCOPE_API_KEY`、`DATABASE_URL`、`LATEXMLC_PATH`/`LATEXMLPOST_PATH`/`MAGICK_PATH`。

## 数据与许可

实验语料仅收录 **CC0 / CC-BY / CC-BY-SA** 许可的论文（翻译属衍生作品），许可经 arXiv OAI-PMH
与 PMC 元数据逐篇核验，清单见各语料目录的 `manifest.csv`。数据目录结构见 `DATA.md`。

完整语料与全量结果体积大且可复现，不入库；仓库随附 **`examples/`**：双语渲染对照样例
（每条腿 3 篇，同步滚动、含公式/图片）、论文表格对应的聚合指标 CSV、以及两条腿的完整
语料清单（含许可逐篇标注），见 `examples/README.md`。

## 测试

```bash
pytest xinda/tests/ -v   # 纯函数 parity 测试，不依赖 DB
```
