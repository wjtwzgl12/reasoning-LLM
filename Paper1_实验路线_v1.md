# PHYRE 主实验路线 v1(方案 A · 严格对齐架构)

**配套文档**:`Paper1_架构设计_L1-L4.md` v1
**执行环境**:Colab Pro+(A100 / L4 GPU)+ 本地 Obsidian 仓库作版本控制
**时间**:8–12 周(起点 2026-04-26,预计完稿 2026-06-21 ~ 2026-07-19)
**产出目标**:§9F 主实验,可直接进 paper §3–§6 的全量数据

---

## 0. 收紧清单(相对上一版路线的 6 处修正)

| # | 文档承诺 | v0 路线 | v1 路线(本文) |
|---|---|---|---|
| 1 | NOTEARS/DAGMA 先验 | Week 4 才训 | **Week 2 完成**,Week 3 并入 L2 |
| 2 | DAD/iDAD amortized BOED | "有空再做" | **Week 6-7 必做**,作 L4 主实验对比项 |
| 3 | Trust-region refine | 埋在 MCTS | **B3 独立里程碑**,Week 4 |
| 4 | Library-growth active 协议 | 事后统计 | **D 轨 benchmark 植入 ≥10 OOV 题**,主实验触发 |
| 5 | IdentGap/EIG 三联指标 | 隐含 | **Week 5 主表头即切换**,aggregate 降级附录 |
| 6 | 6 类 relation 封闭集 | 未约束 | **D1 标注 schema 写死**,边 relation 枚举 |

---

## 1. Colab 并行资源规划

### 1.1 账号与运行槽

* **主号(Pro+)**:GPU A100 80GB · 长时训练(PRM, SBI, NOTEARS, DAD 策略网)
* **副号(Pro)** :GPU L4 22GB · 长时推理(MCTS 主实验、benchmark 批处理)
* **本地 CPU**  :PyBaMM 正演(oracle 响应预算、trust-region refine 的残差计算)

### 1.2 并行轨道(Week 1 起同时开工)

```
轨 A  本体 + 词表生长      Colab CPU notebook     (LLM API 密集,不要 GPU)
轨 B  PRM/SBI/MCTS/Refine  Colab Pro+ A100        (GPU 密集)
轨 C  PCE/BOED/DAD 估计器  Colab Pro L4           (GPU 中等)
轨 D  Oracle benchmark    本地 CPU + Colab CPU   (PyBaMM 正演 + LLM 标注)
```

**4 轨同时跑是 Colab 并行的关键**:A/D 用 CPU runtime(免费额度都够),B 吃主号 GPU,C 吃副号 GPU。B 与 C 的 GPU 互不冲突。

### 1.3 每日 Colab 工作流

* **开机**:`git pull` → 挂载 Drive → 激活对应轨的 notebook
* **关机前**:`git add -A && git commit -m "W<週>-D<日>-<轨>: ..."` → `git push`
* **长任务**:用 `nohup` + `tmux`-style cell(`%%script bash --bg`),12h 自动断线前 checkpoint 到 Drive

---

## 2. 预置资产(Week 0,启动前必须完成)

### W0 checklist(约 3–5 天)

1. **`git init`** + push 到 private GitHub repo(`phyre-paper1`)
2. **API 余额**:DeepSeek-V3 充值 ¥5000(估 §9F 全量 ~4000 万 tokens + Week 1 试跑)
3. **Colab Pro+ 订阅**(主号)· Pro(副号)
4. **Google Drive 目录**:`/MyDrive/phyre/{ckpt,data,results,logs}` 150 GB
5. **本地环境**:Windows + PyBaMM 24.x + Python 3.11 conda env(已有)
6. **数据归档**:`data/echem_rules/staging/*.jsonl`(9 份)+ `data/benchmark/echem_reason_benchmark.jsonl`(128 题)挂 Drive
7. **恢复缺失 ckpt 的 rerun 计划**:PRM(§9A.3)+ SBI w₄(§9C.2)重训入口 notebook 就位

---

## 3. 任务矩阵(W×任务 × 轨)

### 约定

* 每个任务有 **前置依赖** `← [task_id]`、**产出** `→ <artifact>`、**执行位置** `@ <Colab主号|Colab副号|本地|CPU-notebook>`。
* 🔴 = 关键路径;🟡 = 关键路径上的并行支路;🟢 = 可延后。

---

### Week 1(启动 · 四轨并发)

| ID | 任务 | 轨 | 依赖 | 产出 | 位置 | 优先级 |
|---|---|---|---|---|---|---|
| A1 | 从 9 份规则抽 V/S/M/C seed 词表 | A | W0 | `ontology_v1_seed.json` (|V|=15, |S|=20, |M|=12, |C|=10) | CPU-nb | 🔴 |
| A2 | 冻结 Grammar(6 种 relation,节点 schema) | A | A1 | `grammar_spec.md` | 本地 | 🔴 |
| B1 | PRM 重训启动(§9A.3 复现) | B | W0 | `ckpt/prm_v2.pt` training job | Colab 主号 | 🔴 |
| B2 | SBI w₄ 重训启动(§9C.2) | B | W0 | `ckpt/sbi_w4_v2.pt` training job | Colab 主号(B1 后) | 🟡 |
| C1 | PCE 估计器实现 + 单测(合成 h) | C | — | `src/pce_estimator.py` + 合成 benchmark 通过 | Colab 副号 | 🟡 |
| C2 | BOED 枚举版实现 + 单测 | C | C1(接口) | `src/boed_enum.py` | Colab 副号 | 🟡 |
| D1 | Oracle benchmark 标注 schema(JSON contract + 6 relation 枚举 + OOV 题指标) | D | — | `data/benchmark/phyre_oracle_schema.json` | 本地 | 🔴 |
| D2 | 从 §9E.1 128 题筛 L3/L4 的 37 题 + 扩写为多轮 | D | D1 | 10 题 draft | 本地 | 🟡 |

**W1 验收**:A1/A2 冻结;B1 出第一个 checkpoint;C1 PCE 在合成 toy 问题上 MI 估计误差 <10%;D1 schema 通过自查,D2 首 10 题成型。

---

### Week 2(先验 + benchmark 加速 + 工件集成)

| ID | 任务 | 轨 | 依赖 | 产出 | 位置 | 优先级 |
|---|---|---|---|---|---|---|
| A3 | NOTEARS/DAGMA 从规则语料学结构先验 | A | A1/A2 | `ckpt/notears_prior_v1.pt` + `p(h)` sampler | Colab 副号(GPU) | 🔴 |
| A4 | OOV resolver 实现(LLM propose + grammar-check + physics-check + PRM conf) | A | A2, B1 初版 | `src/oov_resolver.py` | CPU-nb | 🟡 |
| B1-cont | PRM 重训完成 + holdout 验证 | B | B1 | `ckpt/prm_v2.pt` 收敛,AUROC ≥ §9A.3 | Colab 主号 | 🔴 |
| B2-cont | SBI w₄ 重训完成 | B | B2 | `ckpt/sbi_w4_v2.pt` | Colab 主号 | 🟡 |
| C3 | PCE 在真实 PyBaMM 正演上验证(K=4,|H| 小) | C | C1, B2 | log MI 与 枚举真值的偏差报告 | Colab 副号 | 🟡 |
| D2-cont | 40 题扩写(累计 50),其中 ≥ 8 题标为 OOV | D | D1 | benchmark 50 题 | 本地 + 标注搭档 | 🔴 |

**W2 验收**:A3 NOTEARS 先验采样合法子图比例 >90%;B1 PRM holdout AUROC 达标;D2 50 题合法(schema 校验通过);C3 PCE 真实正演偏差 <15%。

---

### Week 3(L2 骨架:MCTS + policy + physics veto)

| ID | 任务 | 轨 | 依赖 | 产出 | 位置 | 优先级 |
|---|---|---|---|---|---|---|
| B3 | MCTS + PUCT 主循环(LLM policy + NOTEARS prior 融合) | B | A3, B1-cont | `src/mcts_search.py` | Colab 主号 | 🔴 |
| B4 | physics veto 模块(PyBaMM 残差 → 硬门) | B | W0 | `src/physics_veto.py` | 本地 CPU | 🟡 |
| B5 | PRM trajectory scorer 接入 MCTS leaf reward | B | B1-cont, B3 | MCTS reward pipeline 打通 | Colab 主号 | 🔴 |
| C4 | BOED 枚举版在 toy + §9E.1 对接 PCE | C | C1, C2 | `boed_enum` 在 toy 上 EIG 与枚举真值一致 | Colab 副号 | 🟡 |
| D3 | benchmark 100 题目标推进(累计 80),每题 ≥ 5 种 e 预跑 PyBaMM 响应(oracle) | D | D2-cont | benchmark 80 题 + 400 条 oracle 响应 | 本地(长跑) | 🔴 |

**W3 验收**:B3+B5 在 D2 的前 20 题上能端到端跑出 top-K hypothesis(不保证质量);B4 在 20 条合成谱上 veto 正负例分类精度 >85%;D3 oracle 响应覆盖率 100%。

---

### Week 4(Trust-region refine + L3 上线 + 端到端首跑)

| ID | 任务 | 轨 | 依赖 | 产出 | 位置 | 优先级 |
|---|---|---|---|---|---|---|
| B6 | **Trust-region refine**(SBI Fisher + quasi-Newton 连续参数优化) | B | B2-cont, B3 | `src/refine_trust_region.py` + 并入 MCTS reward | Colab 主号 | 🔴 |
| C5 | L3 selection 管线(MI + KL merge + IdentGap) | C | C3, B3 | `src/l3_selection.py` | Colab 副号 | 🔴 |
| C6 | IdentGap 阈值 τ_ret/τ_def 从 D 的 30 题标定 | C | C5, D3 | `config/thresholds_v1.json` | Colab 副号 | 🟡 |
| D4 | benchmark 完成 100 题(含 ≥ 10 OOV 题 + ≥ 20 需多轮题) | D | D3 | `phyre_oracle_v1.jsonl` 冻结 | 本地 | 🔴 |
| E1 | **端到端 pilot**:L1+L2+L3 在 benchmark 30 题上 1 seed 跑通 | B+C | B6, C5, D4 | `results/phyre_pilot_W4.jsonl` | Colab 主号 | 🔴 |

**W4 验收**:E1 pilot 完成,Top-1 mechanism accuracy ≥ 0.5,IdentGap 分布合理(不是全贴在 0 或 τ_def)。benchmark v1 100 题冻结上传 Drive。

---

### Week 5(L4 上线 + 指标切换 + 中期 pilot)

| ID | 任务 | 轨 | 依赖 | 产出 | 位置 | 优先级 |
|---|---|---|---|---|---|---|
| C7 | L4 BOED 枚举版接入闭环(IdentGap ≥ τ_def 触发选 e*) | C | C4, C5 | `src/l4_boed_loop.py` | Colab 副号 | 🔴 |
| C8 | **指标主表头切换**:IdentGap/rounds-to-resolve/EIG 为主,aggregate 降附录 | C | — | `src/report_metrics.py` | CPU-nb | 🔴 |
| A5 | **Library-growth 协议**激活(运行中触发,写 `ontology_v2.json`) | A | A4, E1 | `src/library_growth.py` + 生长事件日志 | CPU-nb | 🔴 |
| E2 | 端到端 pilot 60 题 × 1 seed(L1+L2+L3+L4) | all | C7, A5 | `results/phyre_pilot_W5.jsonl` + IdentGap 曲线 | Colab 主号 | 🔴 |

**W5 验收**:E2 完成,rounds-to-resolve 中位数 ≤ 3;library-growth 在 OOV 题上触发 ≥ 5 次;指标 dashboard(IdentGap/EIG/rounds)出第一版。

---

### Week 6(DAD amortization · 关键差异点)

| ID | 任务 | 轨 | 依赖 | 产出 | 位置 | 优先级 |
|---|---|---|---|---|---|---|
| C9 | **DAD 策略网络 π_φ(e\|D)** 实现 + 训练 | C | C7, D4 | `ckpt/dad_policy_v1.pt` | Colab 主号(A100) | 🔴 |
| C10 | DAD vs 枚举 BOED 对比(EIG per wall-second) | C | C9 | `results/dad_vs_enum.json` | Colab 副号 | 🟡 |
| B7 | MCTS 搜索效率调优(c_puct, N_sim, top-B) | B | E2 | sweep 报告 | Colab 主号 | 🟡 |
| E3 | 端到端 pilot 60 题 × 3 seeds(换 DAD 版) | all | C9, C10 | `results/phyre_pilot_W6.jsonl` | Colab 主号 | 🟡 |

**W6 验收**:DAD 在 EIG/wall-second 上 ≥ 枚举版 3×;E3 跨 seed 方差可控(IdentGap std/mean < 0.3)。

---

### Week 7(主实验正式跑 + 消融实验)

| ID | 任务 | 轨 | 依赖 | 产出 | 位置 | 优先级 |
|---|---|---|---|---|---|---|
| F1 | **主实验 full run**:PHYRE(all 4 layers)× benchmark 100 题 × 3 seeds | all | E3 | `results/phyre_main_v1.jsonl` | Colab 主号 + 副号 并跑 | 🔴 |
| F2 | baseline 复现:C0, C1, C3full(§9E.1) × 100 题 × 3 seeds | — | D4 | `results/baseline_v1.jsonl` | Colab 副号 | 🔴 |
| F3 | baseline 新增:AutoEIS, Coscientist-lite(接口适配) | — | D4 | `results/baseline_ext.jsonl` | Colab 副号 | 🟡 |
| F4 | **消融**:去 L1-grow / 去 MCTS / 去 L3 merge / 去 L4 / 去 refine | all | F1 | 5 份 ablation jsonl | Colab 主号 | 🔴 |

**W7 验收**:F1 完成率 ≥ 95%;F2 baseline 与 §9E.1 对齐度 >0.95(aggregate 相关);消融 5 条全跑完。

---

### Week 8(数据读出 · 论文稿 · 收尾)

| ID | 任务 | 依赖 | 产出 |
|---|---|---|---|
| F5 | 三联指标主表 + 消融表 + IdentGap/rounds 学习曲线 | F1-F4 | 论文 §4 全部图表 |
| F6 | Library-growth 追踪图(|V| 随 episode 增长,新条目案例) | A5, F1 | 论文 §3.1 图 |
| F7 | DAD vs 枚举 scaling 曲线 | C10, F1 | 论文 §4.4 图 |
| F8 | 反思错误案例 20 条(失败模式分类) | F1 | 论文 §5 讨论 |
| P1 | Paper §3-§6 定稿初版 | F5-F8 | `Paper1_v0.8.md` |
| P2 | Repro bundle(`colab/phyre_main.zip` + README + Jupyter reproduction) | 全部 | `colab/phyre_v1.zip` |

**W8 验收**:paper §3-§6 草稿出齐,结果图自洽,可进入 §7 相关工作 + §8 结论撰写。

---

### Week 9-10(缓冲 · 若 W7 有滑期)

* W7 主实验若因 API 限流/Colab 断线延误,W9 补跑。
* library-growth 若触发 <5 次,W9 加样(再写 20 道 OOV 题)。
* 同行内部评审修改。

### Week 11-12(缓冲 2 · 若 DAD 训练不收敛)

* DAD 替代方案:iDAD(Ivanova 2021)或退回枚举 + 报告为 negative result。
* 整体时间兜底上限 12 周。

---

## 4. Colab 并行作战图(关键周逐日细化)

### Week 1

```
D1 周一 ┃ A1 读 9 规则 起 · B1 PRM 训练 cell 起(主号 GPU)· C1 PCE cell 起(副号 GPU)· D1 起草 schema
D2 周二 ┃ A1 词表 v0 · B1 监控 loss · C1 合成 toy · D1 schema review
D3 周三 ┃ A2 grammar_spec · B1 @checkpoint1 · C1 单测通过 · D2 前 5 题
D4 周四 ┃ A1/A2 冻结评审(你 + 我) · B2 SBI 训练起(主号 B1 完后) · C2 BOED 枚举 · D2 10 题
D5 周五 ┃ W1 验收 · git tag w1-done · Drive 同步 · 同步 notebook 列表
D6-D7 ┃ 缓冲 / 追进度
```

### Week 4(端到端首跑,最关键)

```
D1 周一 ┃ B6 trust-region refine 写 + 单测
D2 周二 ┃ B6 并入 MCTS,小 benchmark 验证
D3 周三 ┃ C5 L3 selection 合并通路
D4 周四 ┃ C6 τ_ret/τ_def 标定 30 题
D5 周五 ┃ D4 benchmark 100 题冻结 · Drive 上传
D6 周六 ┃ **E1 pilot 启动**(主号 A100)— 30 题 × 1 seed,预计 6-10 h,夜间跑
D7 周日 ┃ E1 读数据 · 问题分诊 · 决定是否滑期
```

### Week 7(主实验)

```
D1-D2 ┃ F1 PHYRE main 启动(主号 A100 · 100 题 × 3 seeds · 并发=8)
       ┃ 并行 F2 baseline(副号 L4 · 100 题 × 3 seeds × 3 config)
D3-D4 ┃ F1/F2 继续 · 每 6h checkpoint 到 Drive
D5    ┃ F3 扩展 baseline · F4 ablation 启动(主号,F1 完后)
D6-D7 ┃ F4 ablation 跑完 5 条 · 初步读数据
```

---

## 5. Drive 目录结构(固定不变)

```
/MyDrive/phyre/
├── ckpt/
│   ├── prm_v2.pt
│   ├── sbi_w4_v2.pt
│   ├── notears_prior_v1.pt
│   └── dad_policy_v1.pt
├── data/
│   ├── ontology_v1_seed.json
│   ├── ontology_v2.json          # library-growth 后
│   ├── phyre_oracle_v1.jsonl     # benchmark 100 题
│   ├── phyre_oracle_schema.json
│   └── echem_rules/staging/*.jsonl
├── results/
│   ├── phyre_pilot_W4.jsonl
│   ├── phyre_pilot_W5.jsonl
│   ├── phyre_pilot_W6.jsonl
│   ├── phyre_main_v1.jsonl
│   ├── baseline_v1.jsonl
│   ├── baseline_ext.jsonl
│   ├── ablation_{noL1grow,noMCTS,noL3merge,noL4,noRefine}.jsonl
│   └── dad_vs_enum.json
└── logs/
    └── <task_id>_<date>.log
```

---

## 6. 每周产出收敛指标(Go/No-Go 判据)

| Week | 必须过的指标 | 不过则... |
|---|---|---|
| W1 | PRM 训练 loss 下降 · PCE toy 误差 <10% | 暂停 B,查 §9A.3 数据是否完整 |
| W2 | PRM AUROC ≥ 基线;NOTEARS 合法子图率 >90% | PRM 不达标 → 延后 1 周;NOTEARS 不达标 → 退回 uniform prior 并标"v1 降级" |
| W3 | MCTS 端到端打通(可跑 20 题)· benchmark 80/100 | 任一不过,延后 W4 pilot 到 W5 |
| W4 | pilot 30 题 top-1 acc ≥ 0.5 | <0.4 说明 L2 有 bug,停 L4 开发,查根因 |
| W5 | rounds-to-resolve 中位数 ≤ 3 · library-growth 触发 ≥5 | <5 次 → D 补 OOV 题 |
| W6 | DAD EIG/wall-second ≥ 3× 枚举 | 不过 → 降级为 iDAD 或 negative result |
| W7 | F1+F2+F4 全量跑完 · 数据完整率 ≥95% | <90% → W9 补跑 |
| W8 | 论文 §3-§6 草稿出齐 | 缺图 → W9 补 |

---

## 7. 风险登记册

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| PRM 重训不达 §9A.3 原水平 | 中 | 🔴 L2 reward 失准 | W2 就能看到 AUROC,不达 → 加训 2 天 + 换 DeBERTa backbone |
| NOTEARS 先验学不出有意义结构 | 中 | 🟡 L1 退化为 uniform | 预案:uniform prior + 在论文中标注 "结构先验 as future work"(但需文档同步降级) |
| DAD 策略网训练不稳 | 高 | 🟡 L4 amortization 失败 | 预案:iDAD 或 negative result + 枚举兜底 |
| benchmark 100 题工作量爆炸 | 高 | 🔴 D 轨滑期 | 分两批,W4 冻结 80 题先开跑;其余 20 在 W6 补 |
| Colab GPU 排队 / 断线 | 高 | 🟡 长任务中断 | 每 2h checkpoint Drive;主号 +副号双保险;夜间跑 |
| DeepSeek API 限流 | 高 | 🟡 主实验拖慢 | 充值分两次,W4 和 W7 前各一次;concurrency=8 留余量 |
| library-growth 触发不足 | 中 | 🟡 paper §3.1 图空 | D 轨主动植入 10-15 OOV 题(已在 D4 要求) |
| judge 仍偏短文本 | 低 | 🟢 附录问题 | 三联指标已不以 judge 为主 |

---

## 8. 版本与交付

* `Paper1_架构设计_L1-L4.md` v1(设计)
* `Paper1_实验路线_v1.md` v1(本文档)
* 每周 Friday `git tag w<N>-done`,关键产出与路线交叉对账(见 §6 Go/No-Go)
* W8 交付 `Paper1_v0.8.md`,W10/W12 视情况 v0.9/v1.0

---

## 9. 下一步(本周立刻执行)

**W0 → W1 切换动作(本周内 3-5 天完成)**:

1. 今天:`git init` + GitHub repo + 第一次 push(含本文档 + 架构文档)
2. 明天:Colab Pro+ 订阅 + Drive 目录创建 + DeepSeek 充值
3. 后天:我起草以下 4 份 notebook 骨架(你审过后开跑):
   - `nb/A1_seed_vocab.ipynb`(CPU-nb)
   - `nb/B1_prm_retrain.ipynb`(Colab 主号)
   - `nb/C1_pce_estimator.ipynb`(Colab 副号)
   - `nb/D1_benchmark_schema.ipynb`(本地)
4. D4-D5:4 轨并发启动,进入 Week 1 正式执行。

---

**路线严格对齐 `Paper1_架构设计_L1-L4.md` v1 全部 6 项承诺。若架构文档后续修改,本路线同步改版(vN)并 git diff 记录。**
