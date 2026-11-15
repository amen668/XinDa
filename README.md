# XinDa（信达）— Fidelity-First Scholarly Document Translation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-required-336791.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-supported-2496ED.svg)](docker-compose.yml)

**中文** | [English](README.en.md)

**信达**取自严复"信、达、雅"：**信（保真）为先，达（全文可读）随后**。本框架面向科技论文的
全文机器翻译与质量评测，组织以下三类工作：

- **用量与费用**：记录模型调用量。双向输入对照中的单篇翻译调用按参考费率折算约
  ¥0.033–0.044，不是实际账单，也不包含附加核查、格式转换和人工审校费用。
- **结构保护**：占位符替换与还原保护公式、引用等元素。效果取决于保护范围、占位符完整性
  与回填规则；中文专项验证发现原规则遗漏上下标，不能承诺全部结构无损。
- **质量检查**：事实核查、质量评分和门控组织自动精修与优先人工复核。
  未被标记不等于译文正确，全文专业审校仍需保留；本研究未证明人工工时下降比例。

框架处理 **JATS XML** 与由 **LaTeXML 工具生成的 XML**，分别支持中文期刊 zh→en
与 arXiv en→zh。LaTeXML 是格式转换工具；仅有 PDF 的论文不能直接进入该流程。

## 实验资料与离线复算

2026年11月整理的[复现资料包](research/revision_20261115/README.md)提供实验分组、
源码、保存的输入输出、提示词、许可与校验和，以及不调用模型的复算步骤。
不同实验的样本与统计口径分别报告：

| 资料 | 样本 | 用途 |
|---|---|---|
| 双向输入方式对照 | 中文25篇、英文92篇；英文原始XML对照使用15篇公共子集 | 结构保留与参考调用费用 |
| 中文结构专项验证 | 50篇，包含原25篇 | 翻译覆盖区域的结构与上下标检查 |
| 中文源句命题核查 | 31篇中的50组命题 | 对照与预设改写的受控核查，尚无独立专家金标准 |
| 英译中改写观察 | 40组，来自34个不同源段 | 描述评测响应；85%是异常标记比例，不是错误检出率 |
| 质量门控记录 | 10篇、2230个单元 | 747个触发、677个实际精修、376个复检后仍被标记 |
| 历史批量运行 | 1656次任务、1560篇有成功输出的论文 | 匿名运行及结构统计；不公开历史全文译文 |

保存数据的离线复算不等于重新调用历史模型。部分旧运行缺少逐次请求参数和不可变模型版本，
模型重跑还需要单独配置服务凭据并产生费用。完整范围及限制见资料包。

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
cp .env.example .env        # 设置 POSTGRES_PASSWORD、DATABASE_URL 和 DASHSCOPE_API_KEY
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
Docker 的数据库服务和应用使用同一个 `POSTGRES_PASSWORD`，本模板要求使用URL安全字符的口令，
在本地 `.env` 中设置。本机直接连接数据库时，`DATABASE_URL` 中的特殊字符需编码。
已有数据库卷不会因修改环境变量自动更换数据库口令。

## 数据与许可

双向输入对照采用 **CC0 / CC-BY / CC-BY-SA** 许可论文，中文50篇专项采用 **CC-BY** 论文；
逐篇来源与许可见相应清单。历史运行快照的版本许可记录不完整，仅公开匿名统计。
代码的 MIT 许可不覆盖第三方原文和译文，数据保留各自许可及署名。目录说明见 [DATA.md](DATA.md)。

`examples/` 保留双语渲染样例、历史结果和语料清单，见 [样例说明](examples/README.md)。
新增实验资料集中在 `research/`，无需把本地 `corpus/`、`results/` 或 `workspace/` 整体上传。
环境文件、日志、论文稿、审稿意见与临时审读资料不随仓库发布。

## 测试

```bash
pytest xinda/tests/ -v   # 纯函数 parity 测试，不依赖 DB
```
