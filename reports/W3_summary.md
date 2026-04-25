---
week: W3
status: in-progress
theme: each track climbs one layer (B2 / C3 / D3) + A4 NER retrain + C4 real DFN
---

# W3 Summary (in-progress)

## 主线
- B2 + C3 + D3：每条 track 各上一个 new layer（runtime / e2e / interactive）
- C4 + D4 共享 pybamm 24.x 修复成果
- A4 CPU 跑作并行

## Track B2 — L2 MCTS+PUCT runtime
- **notebook**: `nb/B2_mcts_runtime.ipynb` ✅ 已写
- **算法**: PUCT `Q + c_puct·P·√ΣN/(1+N)`；K_BRANCH=6；N_SIMS=24；C_PUCT=1.5；PRM_HIGH=0.9
- **prior**: DeepSeek-V3 提名 top-k 均匀概率（W4 替换为学得 policy）
- **value**: PRM (`prm_v1.pt`) trajectory-level Sigmoid score
- **目标**: §9E.1 L2 60 题，EM≥0.55 ∧ F1≥0.70
- **commit**: pending
- **status**: 待 Colab Pro+ A100 运行

## Track C3 — L3 PCE end-to-end (20 题)
- **notebook**: pending (`nb/C3_l3_runtime.ipynb`)
- **目标**: 接 C2 analytic surrogate + PRM；§9E.1 L3 20 题端到端 MI ≥ 1.0 nats
- **status**: planned

## Track D3 — L4 BOED interactive (17 题)
- **notebook**: pending (`nb/D3_l4_boed.ipynb`)
- **目标**: BOED 选实验 → 调 oracle (D2) → 更新后验；EIG 改善 ≥ baseline +0.5 nats
- **status**: planned

## Track A4 — NER retrain on §9A.3 holdout
- **notebook**: pending (`nb/A4_ner_retrain.ipynb`)
- **目标**: 用 A2 prior + A3 vocab 重训 NER，§9A.3 holdout F1 ≥ baseline +0.05
- **status**: planned (CPU)

## Track C4 — real PyBaMM DFN EIS
- **notebook**: pending (`nb/C4_real_dfn.ipynb`)
- **目标**: pin pybamm 24.x，多路径探测 `EISSimulation`；不行则落到 SPM 时域阶跃→FFT
- **status**: planned

## 风险 / 偏差
- B2 EM 目标 0.55 偏激进；recovery ladder 已写入 notebook (bump N_SIMS / K_BRANCH 等)。
- C4 仍可能找不到真实 DFN EIS — 退回 SPM + 时域阶跃 FFT。

## 输出物 (滚动更新)
- ✅ `nb/B2_mcts_runtime.ipynb`
- ⬜ `nb/C3_l3_runtime.ipynb`
- ⬜ `nb/D3_l4_boed.ipynb`
- ⬜ `nb/A4_ner_retrain.ipynb`
- ⬜ `nb/C4_real_dfn.ipynb`

## 下周 (W4) 计划
- B1 在真实 §9A.3 holdout 上重新评估 PRM
- B2 prior 升级为学得 policy（如果 W3 EM 接近但未达标）
- C4 / D4 联调
