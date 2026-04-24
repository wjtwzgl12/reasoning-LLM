# PHYRE — W1 Notebook 骨架

4 份 notebook 对应 `Paper1_实验路线_v1.md` 的 Week 1 四轨并发任务:

| Notebook | 轨 | Runtime | 目标 | 产出 |
|---|---|---|---|---|
| `A1_seed_vocab.ipynb` | A | Colab CPU(免费)| 9 份规则 → V/S/M/C seed vocab | `ontology_v1_seed.json` |
| `B1_prm_retrain.ipynb` | B | **Colab Pro+ A100**(主号)| PRM 重训(§9A.3 复现)| `prm_v2.pt` |
| `C1_pce_estimator.ipynb` | C | **Colab Pro L4**(副号)| PCE MI 下界估计器 | `src/pce_estimator.py` |
| `D1_benchmark_schema.ipynb` | D | 本地 Python | Oracle benchmark JSON schema | `phyre_oracle_schema.json` |

## 启动顺序

1. **D1 先冻结 schema**(A/C/其他轨都要用到其 mechanism_subgraph 定义)
2. **A1 / B1 / C1 同日开跑**:B 吃主号 A100,C 吃副号 L4,A 走 CPU notebook,三者 GPU 不冲突
3. 每日收工前:`git add -A && git commit -m "W1-D<n>-<轨>: ..."` + Drive 同步

## 共享约定

- Drive 根目录:`/content/drive/MyDrive/phyre/`(已在各 notebook 首 cell 挂载)
- DeepSeek API key:Colab Secrets 配 `DEEPSEEK_API_KEY`
- 每个 long job 每 500 step / 2h checkpoint 到 Drive(Colab 12h 断线保护)

## Go/No-Go(Week 1 末周五)

- A1: |V|≥15, |S|≥20, |M|≥12, |C|≥10,人工 spot-check 20 条通过
- B1: 第一个 PRM checkpoint 出,loss 下降无 NaN
- C1: toy MI 误差 <10%
- D1: schema validator 通过,首 10 题 draft 手工过检

不过 → 本文档 §6 预案,但 **不许拖到 W2**(关键路径不能滑)。
