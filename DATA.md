# 数据地图（语料 / 译文 / 结果）

## 公开复现资料

框架源码保留在 `xinda/`。2026年11月补充的实验源码、保存输出、提示词、署名、许可及离线复算步骤集中在
[`research/revision_20261115/`](research/revision_20261115/README.md)。`examples/` 保留已有样例和历史记录。

| 资料 | 样本与统计单位 | 适用范围 |
|---|---|---|
| 双向输入对照 | 中文25篇、英文92篇；英文原始XML对照为15篇公共子集 | 整篇文档结构与参考调用费用 |
| 中文结构专项 | 50篇，含原25篇 | 去重后的翻译覆盖区域，采用严格子树匹配 |
| 中文源句命题核查 | 31篇中的50组 | 与构造标签的一致性，尚无独立专家金标准 |
| 数值扰动核查 | 45篇中的200个数值单元 | 指定扰动和匹配规则的测试 |
| 英译中改写观察 | 40组、34个不同源段 | 探索性评测响应；异常标记不等于已确证差错 |
| 门控记录 | 10篇、2230个翻译单元 | 747触发、677实际精修、376复检后仍标记 |
| 历史运行快照 | 1656次任务、1560篇有成功输出的论文 | 匿名运行及结构统计，不公开历史全文译文 |

不同资料的样本可能重合，不能相加为“全文验证论文总量”。旧25篇的260个公式与新50篇的143个公式，
以及相应单元数，采用不同范围与匹配口径，不能直接比较。未标记比例83.1%不表示正确率或人工工时节省。
`examples/audit/`保留抽样记录；原2.3%残余错误率估计已撤下，历史标签仅供追溯。

公开包是从本地资料中筛选的独立副本。离线复算只读取保存结果，不调用模型，也不能恢复缺失的历史参数；
模型重跑需要另行配置凭据并产生费用。软件MIT许可不覆盖论文原文、摘录、图片和译文，第三方内容保留逐篇CC许可及署名。

## 本地工作目录

两条翻译腿，`corpus/` 与 `results/` **同名一一对应**：

| 腿 | 方向 | 格式 | 源文（原文） | 输出 |
|---|---|---|---|---|
| **中文期刊** | zh→en | JATS XML | `corpus/jats_zh2en/*.xml`（25 篇，CC-BY，色谱/血液/肺癌） | `results/jats_zh2en/` |
| **arXiv** | en→zh | LaTeXML XML | `corpus/arxiv_en2zh/*.xml`（论文实验 92 篇，另含 1 篇 smoke test，多学科） | `results/arxiv_en2zh/` |

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
> 不整体随仓库分发；随库样例（双语对照页 ×6、聚合指标 CSV、语料清单）在 `examples/`，新增筛选资料在 `research/`。

## 其他目录

- `corpus/_archive/` —— 探路语料（jats=5 篇、jats_formula=9 篇公式压力子集），已被论文实验的 25/92 篇正式语料取代；另保留 1 篇 arXiv smoke test。
- `corpus/_scratch/` —— 语料筛选过程留痕（筛选日志、候选清单），保留作复现/limitations 证据。
- `workspace/` —— 流水线工作目录（arXiv 下载的 .tex、LaTeXML 中间产物）；DB 存的是这里的路径。
- `data/glossaries/` —— 外部术语表（术语接地功能用，当前结构实验未启用）。
- `paper/` —— 私有论文稿、审稿意见与投稿材料，保持本地。
- `logs/`、`backups/`、`tmp/` —— 日志、备份及临时审读记录，不公开；整理不删除本地原始数据。
