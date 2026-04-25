# PHYRE 周报存档 / Weekly Reports

每周一篇 markdown，记录该周完成的 track / commits / metrics / 风险点 / 下周计划。

## 命名约定
- `W{n}_summary.md` — 当周总结，状态在文件首部 frontmatter 中（`status: in-progress | frozen`）
- 每个 track 单独章节：A (L1 ontology) / B (L2 reasoning) / C (L3 PCE) / D (L4 BOED)
- 每章节末尾：`commit:` + `tag:` + 量化指标（pass/fail）

## 索引
| Week | 主线 | Tracks | 状态 |
|---|---|---|---|
| W1 | benchmark schema + vocab seed + PCE estimator | A1, B1, C1, D1 | frozen |
| W2 | NOTEARS prior + FunSearch + PRM + analytic EIS + oracle CSV | A2, A3, B?, C2, D2 | frozen (D2 with sanity_note) |
| W3 | MCTS runtime + L3 e2e + L4 BOED + NER retrain + real DFN | B2, C3, D3, A4, C4 | in-progress |
| W4 | PRM 9A.3 holdout re-eval + policy learning | TBD | planned |

## 模板
见 `_TEMPLATE.md`。
