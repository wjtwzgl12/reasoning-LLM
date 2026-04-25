---
week: W2
status: frozen
theme: NOTEARS prior + FunSearch + analytic EIS + oracle CSV
---

# W2 Summary

## 主线
把 4 条 track 的 W2 layer 各推一格：A→A2/A3，C→C2 (real EIS 探索)，D→D2 (oracle CSV)。

## Track A2 — NOTEARS prior (L1)
- **notebook**: `nb/A2_notears_prior.ipynb`
- **commits**:
  - `cf9db19` relax record threshold 200→30 hard / 100 soft + auto-weak-prior
  - `abb3d6b` cell-6: PMI fallback when NOTEARS underdetermined (R<N)
  - `99d47d0` lower lambda1 to 0.005, drop centering, fix probe vocab
  - `fea2306` fix NameError EXPECTED → CANDIDATE_PROBES in review.md writer
- **tag**: `w2-track-a2-frozen` → `fea2306`
- **指标**: NOTEARS / PMI 双轨产出有向因果先验，probe vocab 校核通过
- **PASS**: True (R/N 不够时自动降级 PMI)

## Track A3 — FunSearch growth filter (L1)
- **notebook**: `nb/A3_funsearch_growth.ipynb`
- **commits**:
  - `54d210c` load expanded corpus for delta-scoring; C2: auto-restart on numpy install
  - `1d2f934` fix expand-log schema + add §9E.1 corpus + per-candidate exclusion
- **tag**: `w2-track-a3-frozen` → `1d2f934`
- **指标**: 候选 sub-rule 在 §9E.1 corpus 上的 delta 评分通过
- **PASS**: True

## Track B — PRM (carry-over from W1)
- **commit / tag**: `w2-prm-done-pending-9A3-recheck`
- **指标**: AUC 0.829（仍待 §9A.3 holdout 重测，安排在 W4）

## Track C2 — real / surrogate EIS (L3)
- **notebook**: `nb/C2_pybamm_real.ipynb`
- **commits**:
  - `6bd49e7` C2 single-pass setup, no kernel restart
  - `9ba7176` list pybamm runtime deps + auto-retry on missing modules
  - `101fb86` pin pybamm 25.x; abort cleanly on all-NaN
  - `26d4c43` drop pybamm pin to 24.x (25.12.2 缺 parameter_sets)
  - `5651636` hypothesis-aware analytic EIS fallback (no pybamm dependency)
  - `e64e7e1` rescale analytic EIS surrogate so RC arcs are visible
- **tag**: `w2-track-c-frozen` → `e64e7e1`
- **指标**: PCE MI lower bound 1.386 nats，h1_baseline 解析 EIS 可解
- **PASS**: True (analytic surrogate)
- **note**: pybamm 真实 DFN EIS 在 24.11.2 上 `EISSimulation` 不在顶层，只能走解析 surrogate；真实 DFN 推迟到 W3 C4。

## Track D2 — oracle CSV fill (L4)
- **notebook**: `nb/D2_oracle_csv_fill.ipynb`
- **commits**:
  - `26d4c43` (与 C2 同) D2 auto-purges stale fallback checkpoints
  - `a38c0c1` fix eis_from_params call signature mismatch with C2 surrogate
  - `c703c6b` fuzzy-match mechanism nodes to C2 param overrides
  - `3627af3` D2 freeze with W2 acceptance note; defer real-DFN EIS to W3
- **tag**: `w2-track-d-frozen` → `3627af3`
- **指标 (硬指标)**:
  - schema valid: 50/50 ✓
  - 每 qid ≥3 CSV: 10/10 ✓
- **PASS**: True (硬指标)
- **sanity_note (写入 D2_report.json)**:
  - `eis_backend = analytic_surrogate_via_C2_pce_simulator_pybamm`
  - `real_dfn_status = deferred_to_W3`
  - 视觉上不同假设的 Nyquist 区分度有限（解析 surrogate 限制），W3 C4 用真实 DFN 修复。

## 风险 / 偏差
- **关键反思（用户 push back）**: W2 后期出现"针对 Nyquist 视觉效果反复调 surrogate 参数"的趋向，已停手并以 sanity_note 记录限制，硬指标过线即接受。
- pybamm 版本漂移消耗较多时间；W3 C4 用 24.x 探针穿透到底（含 SPM 兜底）。

## 输出物
- `logs/A2_review.md` `logs/A3_expand_log.json`
- `pvgap_experiment/pce_simulator_pybamm.py` (analytic surrogate 模块化)
- `logs/D2_report.json` (含 sanity_note)
- 5 个 W2 tags 已 push 到 GitHub

## 下周 (W3) 计划
- B2: L2 MCTS+PUCT runtime on §9E.1 L2 (60 题)，目标 EM≥0.55 / F1≥0.70
- C3: L3 PCE 端到端 on §9E.1 L3 (20 题)
- D3: L4 BOED interactive on §9E.1 L4 (17 题)
- A4: NER retrain w/ A2 prior + A3 vocab on §9A.3 holdout (CPU 可跑)
- C4: 真实 PyBaMM DFN EIS（pin 24.x，找到 `EISSimulation` 真路径或退到 SPM）
