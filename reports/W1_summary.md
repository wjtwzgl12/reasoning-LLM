---
week: W1
status: frozen
theme: benchmark schema + vocab seed + PCE estimator baselines
---

# W1 Summary

## 主线
搭起 §9E.1 benchmark 与四层各自的 baseline notebook，让 W2+ 有可加载的数据 / 评测口径。

## Track A — A1 seed vocab (L1)
- **notebook**: `nb/A1_seed_vocab.ipynb`
- **commit**: `3b04c8a` (real fix — grow corpus 65 → ~250 via L1 inclusion + LLM sub-rule expansion)
- **tag**: `w1-vocab-done`
- **指标**: corpus 65 → ~250 sub-rules (LLM 扩展 + L1 包含)
- **PASS**: vocab size + coverage 满足 A2 NOTEARS 输入下限
- **note**: 早期 corpus 65 条不足以支撑 NOTEARS，扩展后 R/N 比满足。

## Track B — B1 PRM retrain (L2 prep)
- **notebook**: `nb/B1_prm_retrain.ipynb`
- **commit**: ~ `cf9db19` 之前
- **tag**: 无独立 W1 tag（在 W2 中 frozen 为 `w2-prm-done-pending-9A3-recheck`）
- **指标**: traj-level AUC 0.829 (在 §9A.3 子集上)
- **note**: 真正的 §9A.3 holdout re-eval 推迟到 W4。

## Track C — C1 PCE estimator (L3)
- **notebook**: `nb/C1_pce_estimator.ipynb`
- **commit**: 早期 (W1 主线)
- **tag**: `w1-pce-done`
- **指标**: PCE MI lower bound ≥ 1.0 nats (h1_baseline)
- **PASS**: True

## Track D — D1 benchmark schema (L4 prep)
- **notebook**: `nb/D1_benchmark_schema.ipynb`
- **commits**:
  - `7d380a0` D1 cell-4: extract key_elements from question_text + structured GT
  - `3742f0c` D1 cell-5: escape literal braces in SUBGRAPH_PROMPT
  - `904c2fb` D1 cell-5: bump max_tokens 600→1500 + retry/repair on truncation
  - `f187dda` D1 cell-5: relation normalizer — map illegal rels to 6 legal labels
- **tag**: `w1-bench-schema-frozen`
- **指标**: §9E.1 benchmark 128 entries (L1:31 / L2:60 / L3:20 / L4:17)
- **PASS**: schema 通过

## 风险 / 偏差
- B1 AUC 0.829 是在 §9A.3 训练子集上测的，holdout re-eval 留待 W4。
- A1 corpus 来源依赖 LLM 扩展，可能引入 hallucinated sub-rules（A3 FunSearch 在 W2 中过滤）。

## 输出物
- `data/benchmark/echem_reason_benchmark.jsonl` (128 条 §9E.1)
- `ckpt/prm_v1.pt` (DeBERTa-v3-large + LoRA, AUC 0.829)
- vocab/corpus 扩展产物
