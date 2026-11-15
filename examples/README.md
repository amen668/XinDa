# examples/ — 随仓库发布的样例数据

本目录保留双向对照样例与历史记录。论文实验为117篇原文XML，另有1篇smoke-test文档。
本地全量译文与工作中间产物**不整体入库**
（见根目录 `.gitignore` 对 `corpus/`、`results/`、`workspace/` 的忽略）。这里放
能查看系统输出与历史数值的子集。2026年11月补充的中文结构、受控核查及匿名运行资料见
[`research/revision_20261115/`](../research/revision_20261115/README.md)。

## bilingual_html/ — 双语渲染对照样例

同步滚动的双栏对照页，每条腿各 3 篇（原文均为 CC-BY，译文为衍生作品同样按
CC-BY 附署名发布）：

每页列出具体许可版本、作者、原文链接与改编说明，汇总见 [ATTRIBUTION.md](ATTRIBUTION.md)。

- `jats_zh2en/` — 中文期刊（JATS XML）→ 英文。图片引用 PMC CDN，需联网显示。
- `arxiv_en2zh/` — arXiv（LaTeXML XML）→ 中文。图片在 `assets/` 内，离线可看。

各目录 `index.html` 列出篇目与出处。完整渲染集合用
`xinda/cli/jats_render.py` / `xinda/cli/latexml_render.py` 重新生成。

## results/ — 历史指标记录

目录结构与本地 `results/` 一一对应，只收聚合 CSV/JSON（不含逐单元大文件）：

- `jats_zh2en/`、`arxiv_en2zh/` — 三系统结构保真对比（`main`/`rawxml` 与朴素基线，
  `mm/` 为跨厂商多模型），列含注解/公式保留率、覆盖率、CometKiwi 等。
- `exp2_fidelity/` — 40组模型生成改写的探索性响应观察。样例来自34个不同源段，
  焦点变化未经逐对全部确证；两侧QA独立出题。原文件名、字段名及汇总标签保留供追溯，
  不能据此认定85%为事实差错检出率，或认定专项核查优于通用质量评测。
  `fidelity_vs_qa_zh.csv`的公开副本已去除未逐行关联出处的`source`、`good_tr`、`bad_tr`文本列，
  保留40行原指标、标签及`source_sha256`以核对34个源段分组；本地原始记录未删除。

## corpus_manifests/ — 语料清单

- `arxiv_en2zh_manifest.csv` — 论文实验使用的 92 篇 arXiv 论文：许可（仅 CC0/CC-BY/CC-BY-SA，
  翻译属衍生作品故按许可筛选）、发表场刊、引用数、学科。
- `jats_zh2en_manifest.csv` — 25 篇中文期刊 PMC OA 文章（均 CC-BY）：期刊、
  标题、结构统计。

人工抽检记录、判读标准和门控汇总见 `audit/README.md`。单独用于流水线
冒烟测试的 `2503.15129` 不计入论文的 92 篇实验语料。

按清单中的 ID 即可用 `xinda/cli/build_corpus.py` / `jats_corpus.py` 重建完整语料。
