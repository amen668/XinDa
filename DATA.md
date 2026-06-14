# 数据地图（语料 / 译文 / 结果）

两条翻译腿，`corpus/` 与 `results/` **同名一一对应**：

| 腿 | 方向 | 格式 | 源文（原文） | 输出 |
|---|---|---|---|---|
| **中文期刊** | zh→en | JATS XML | `corpus/jats_zh2en/*.xml`（25 篇，CC-BY，色谱/血液/肺癌） | `results/jats_zh2en/` |
| **arXiv** | en→zh | LaTeXML XML | `corpus/arxiv_en2zh/*.xml`（93 篇，多学科） | `results/arxiv_en2zh/` |

## 原文在哪

- **JATS 原文（中文期刊）**：`corpus/jats_zh2en/*.xml` —— 直接就是 PMC 抓来的 JATS 全文 XML。
  清单/许可：`corpus/jats_zh2en/manifest.csv`。
- **LaTeX 原文**：
  - 我们实际翻译的**结构源**（LaTeXML 转换后的 XML）：`corpus/arxiv_en2zh/*.xml`
  - **原始 `.tex`**（latexmlc 的输入）：`workspace/<arxiv_id>/<时间戳>/*.tex`

## 输出目录里有什么（每条腿）

```
results/<leg>/
  diff/jats_structure.csv      结构差异化：contract vs raw_xml vs naive（qwen-plus）
  mm/jats_structure.csv        跨厂商多模型：contract × {qwen3.7-plus/qwen3-max/glm-4.7/kimi-k2.5/minimax-m2.5}
  comet.csv                    CometKiwi 无参考质量（每篇 median/p10/mean）
  comet_pairs.json             (源, 译) 散文对，逐单元
  xml/*_en.xml                 整篇译文 XML（仅 JATS 腿已生成）
  bilingual_render/index.html  双语渲染对照（含公式/图片/同步滚动）
                               JATS 腿走 pandoc（cli/jats_render）；arXiv 腿走
                               latexmlpost 三阶段（cli/latexml_render，见其 docstring）
```

> 本文件描述的是**本地数据布局**。`corpus/`、`results/`、`workspace/` 均被 gitignore，
> 不随仓库分发；随库样例（双语对照页 ×6、聚合指标 CSV、语料清单）在 `examples/`。

## 其他目录

- `corpus/_archive/` —— 探路语料（jats=5 篇、jats_formula=9 篇公式压力子集），已被 25/93 篇正式语料取代。
- `corpus/_scratch/` —— 语料筛选过程留痕（筛选日志、候选清单），保留作复现/limitations 证据。
- `workspace/` —— 流水线工作目录（arXiv 下载的 .tex、LaTeXML 中间产物）；DB 存的是这里的路径。
- `data/glossaries/` —— 外部术语表（术语接地功能用，当前结构实验未启用）。
