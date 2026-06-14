# examples/ — 随仓库发布的样例数据

完整语料（118 篇原文 XML）、全部译文与中间产物体积大且可由流水线复现，**不入库**
（见根目录 `.gitignore` 对 `corpus/`、`results/`、`workspace/` 的忽略）。这里放
能直观看到系统输出与论文核心数字的最小子集。

## bilingual_html/ — 双语渲染对照样例

同步滚动的双栏对照页，每条腿各 3 篇（原文均为 CC-BY，译文为衍生作品同样按
CC-BY 附署名发布）：

- `jats_zh2en/` — 中文期刊（JATS XML）→ 英文。图片引用 PMC CDN，需联网显示。
- `arxiv_en2zh/` — arXiv（LaTeXML XML）→ 中文。图片在 `assets/` 内，离线可看。

各目录 `index.html` 列出篇目与出处。完整渲染集合用
`xinda/cli/jats_render.py` / `xinda/cli/latexml_render.py` 重新生成。

## results/ — 聚合指标（论文表格的原始数据）

目录结构与本地 `results/` 一一对应，只收聚合 CSV/JSON（不含逐单元大文件）：

- `jats_zh2en/`、`arxiv_en2zh/` — 三系统结构保真对比（`main`/`rawxml` 与朴素基线，
  `mm/` 为跨厂商多模型），列含注解/公式保留率、覆盖率、CometKiwi 等。
- `exp2_fidelity/` — 保真-质量解耦实验：QA 理解度、声明校验、CometKiwi 盲度
  （`summary_zh.json` 即论文引用的汇总数字）。

## corpus_manifests/ — 语料清单

- `arxiv_en2zh_manifest.csv` — 93 篇 arXiv 论文：许可（仅 CC0/CC-BY/CC-BY-SA，
  翻译属衍生作品故按许可筛选）、发表场刊、引用数、学科。
- `jats_zh2en_manifest.csv` — 25 篇中文期刊 PMC OA 文章（均 CC-BY）：期刊、
  标题、结构统计。

按清单中的 ID 即可用 `xinda/cli/build_corpus.py` / `jats_corpus.py` 重建完整语料。
