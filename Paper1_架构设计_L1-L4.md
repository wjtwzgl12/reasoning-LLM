# PHYRE: Physics-grounded Hypothesis Reasoning for Electrochemistry

**Paper1 架构设计 v1 · L1–L4 核心规格**
草案日期 2026-04-24 · 作者 WJT
定位:§9E.1 C3full 部分收敛的根因诊断 + §9F 主实验的统一架构基线

---

## 0. 动机与核心主张

§9E.1 实测:C3full vs C0 judge.aggregate Δ = +0.030(95% CI 跨零),C3full
边际收益被 LLM-judge 的 length/fluency/plausibility bias 吞没。深层病因有三:

1. **候选坍缩(candidate collapse)**:4 candidates 语义重合率高,critic+refine
   在狭窄空间里"打磨同一个答案"。
2. **等效性(equifinality, Beven 2006)**:EIS 反问题下多组机理拟合同一谱,
   PRM / physics veto 无法在观测不足时做出唯一裁决。
3. **评估失真**:ORM 式 judge.aggregate 不惩罚不可判别的过度自信,也不奖励
   "识别出歧义并提出区分性实验"的行为。

**主张**:把"机理可解释性"重新定义为 **机理识别 + 可区分性 + 实验经济性**
三联指标,并用四层架构解决:

> L1(假设空间)→ L2(在空间内搜索+验证)→ L3(选最有区分力的解释)
> → L4(观测不足时主动取数)→ 回到 L2

---

## 1. 顶层架构图

```
            ┌──────────────────────────────────────────────────┐
            │ L1  Parameterized Mechanism Ontology              │
            │  Grammar (closed) × Vocabulary (growable)         │
            │  primitives = V × S × M × C                       │
            │  prior over subgraphs: NOTEARS/DAGMA              │
            │  OOV resolver + FunSearch library growth          │
            └──────────────────────────────────────────────────┘
                            │ legal hypothesis subgraphs h ∈ H
                            ▼
            ┌──────────────────────────────────────────────────┐
            │ L2  Hypothesis Search & Verification              │
            │  MCTS with LLM policy π_θ (node = partial graph)  │
            │  expansion from Grammar rules                     │
            │  leaf reward = PRM(step) × physics_veto × refine  │
            │  trust-region refinement for continuous params    │
            └──────────────────────────────────────────────────┘
                            │ top-K hypotheses {h_k, p_k}
                            ▼
            ┌──────────────────────────────────────────────────┐
            │ L3  Information-gain Selection                    │
            │  MI(H; Y|D) via PCE / InfoNCE lower bound         │
            │  pairwise KL spread → diversity filter            │
            │  IdentGap = H(p_k) − margin → trigger defer?      │
            └──────────────────────────────────────────────────┘
                            │ if IdentGap < τ: return answer
                            │ else ↓
            ┌──────────────────────────────────────────────────┐
            │ L4  Bayesian Optimal Experimental Design          │
            │  argmax_e  EIG(e) − λ·cost(e)                     │
            │  over discrete E = {f-range, T, SoC, DRT, CV ...} │
            │  amortized via DAD / iDAD when budget allows      │
            └──────────────────────────────────────────────────┘
                            │ recommended next probe e*
                            ▼
                      (oracle / sim) returns y_e  → back to L2
```

---

## 2. L1 — 参数化机理本体(Parameterized Mechanism Ontology)

### 2.1 设计哲学:Grammar 固化 + Vocabulary 生长

对"L1 固化会不会太死板"这一质疑的正式回应:**不固化清单,固化构造规则**。
灵感来自 DreamCoder (Ellis 2021), FunSearch (Romera-Paredes 2024),
AlphaGeometry (Trinh 2024) 的库生长范式,以及 PDDL 的 action schema 分解。

* **Grammar(封闭,v1 冻结)** — 5 类原语位槽,每个节点形如:

  ```
  node ::= ⟨Verb, Substrate, Modifier*, Condition*⟩
  edge ::= ⟨src, dst, relation⟩   relation ∈ {triggers, rate-limits,
                                              competes-with, co-occurs,
                                              amplifies, suppresses}
  ```

  位槽的类型是封闭的(V/S/M/C 四类 + 6 种 relation),但**每个位槽内的取值
  开放**。

* **Vocabulary(开放,可生长)** — V、S、M、C 的取值来自两个源头:

  1. **seed 词表**:从 `data/echem_rules/staging/*.jsonl` 的 9 份顶刊规则中
     抽取(autoeis_joss, allagui2025, hallemans2024, kiefl2025, limon2025,
     perovich2025, singh2024, vasconcelos2024 + 自建),预估 |V|≈15,
     |S|≈20, |M|≈12, |C|≈10。
  2. **OOV resolver**:见 §2.3,遇新术语时先尝试"投影到已有原语的组合"
     (sub-mechanism 分解);投影失败且 confidence > τ 时走 **library-growth
     协议**(§2.4)加入 vocabulary。

* **组合机制**天然由图表达 — 不需要为"SEI-growth + Li-plating 共存"单独建
  条目,它就是两节点 + co-occurs/competes-with 边。

### 2.2 假设空间 H 与先验

一个"假设" h = 子图 ⟨nodes ⊂ Ontology, edges⟩。H 是 Grammar 允许的所有
合法子图,规模 |H| 指数级,需要结构化先验压缩:

* **结构先验 p(h)** 由 **NOTEARS/DAGMA**(Zheng 2018, Bello 2022)从规则
  语料中学:边权先验 W ∈ ℝ^{|nodes|×|nodes|},penalty h(W)=tr(e^{W∘W})−d
  约束无环(或 GFlowNet, Bengio 2021,若允许非 DAG)。先验只负责"哪些边
  合理",不负责识别。
* **条件化**:给定题面 q,p(h|q) ∝ p(h)·π_θ(h|q),其中 π_θ 是 LLM policy
  (§3)。

### 2.3 OOV 解析器(Sub-mechanism Decomposition)

遇到 seed vocabulary 未覆盖的术语 t 时:

1. **LLM propose decomposition**:prompt "把 t 拆成 ⟨V,S,M,C⟩ 组合或
   子图",输出 k=4 个候选分解。
2. **Grammar-check**:每个候选必须能在 Grammar 下重构。
3. **Physics-consistency check**:分解后的子图跑一次轻量 PyBaMM/EIS 残差,
   拒绝与 q 观测矛盾的分解。
4. **Confidence**:基于候选一致性(4 proposal 中出现频次)+ PRM 步评分。

### 2.4 Library-growth 协议(FunSearch-style)

当 OOV 无法被现有 V/S/M/C 的任意组合解释,且跨 ≥3 个题面重复出现时,触发
增长:

* 提议新的 V/S/M/C 值(不是新类别),由 LLM 生成名称 + 定义 + 2 条等价
  chain-of-reasoning。
* **守门**:新条目必须(i)在 ≥3 题面上给出 PRM 步评分 > θ_lib 的
  reasoning;(ii)与现有条目的 cosine 相似度 < 0.85(去冗余);
  (iii)引入后,holdout set 上 judge.mechanism 不退化。
* 通过才写入 `ontology_vx.json`,版本化(git diff 可见每次增长)。

> **Top-conf 灵感**:DreamCoder wake-sleep(神经+符号 library grow)、
> FunSearch("program database + island evolution")、AlphaGeometry
> ("synthetic theorems")。

---

## 3. L2 — 假设搜索与验证(MCTS + PRM/physics + Refine)

### 3.1 状态/动作/奖励

* **state** s = 当前部分子图 h_partial(根节点 = ∅)
* **action** a ∈ A(s):Grammar 允许的一次扩张 —
  ADD_NODE(type, value) / ADD_EDGE(src, dst, rel) / REFINE_PARAM(θ) / STOP
* **policy π_θ(a|s,q)**:LLM-as-policy(DeepSeek-V3 prompt),输出 softmax
  over A(s) 的 top-B
* **rollout value V(s)**:MCTS 模拟到 STOP 后取 R
* **reward** R(h) = α·PRM_trajectory(h) + β·physics_residual(h) +
  γ·refine_gain(h) − δ·|h|(节点数作复杂度惩罚,Occam)

  - PRM_trajectory:§9A.3 PRM ckpt 对每步打分再取几何平均(Lightman 2024)
  - physics_residual:PyBaMM 正演 + EIS 残差 → 0/1/分档 veto(硬门)
  - refine_gain:Self-Refine(Madaan 2023) / SCoRe(Kumar 2024)在
    continuous param 上做 trust-region 更新后残差降低量

### 3.2 搜索算法:MCTS + LLM prior

采用 **AlphaZero-style PUCT**(Silver 2017):

```
UCB(s,a) = Q(s,a) + c_puct · π_θ(a|s,q) · √N(s) / (1 + N(s,a))
```

* LLM policy 充当 prior,MCTS 负责 credit assignment;收缩到 ToT (Yao
  2023) 之上,但加了 physics veto 作为硬约束节点 pruning。
* 终止:N_sim 达到预算,或 top-K 候选 KL 分化已 > τ_div。
* 输出:top-K hypotheses 及其归一化权重 {(h_k, p_k)}。

### 3.3 连续参数 refinement(trust-region)

对 h 中的连续参数 θ(D_Li, R_ct, k_0, …),在 PyBaMM 正演上做带信赖域
约束的 quasi-Newton:

```
θ* = argmin_θ ||ẑ(θ) − z_obs||²   s.t.  ||θ − θ_prior||_M ≤ Δ
```

M 取 SBI 后验的 Fisher 估计。这一步在 §9C.2 SBI w₄ 基础上加了 trust-region
(LaMM 风格),避免等效谷里漂移。

> **Top-conf 灵感**:AlphaGeometry(neural policy + symbolic engine
> verify)、AlphaZero(PUCT)、ToT(deliberative search)、Lightman
> PRM(ICLR 2024)、Self-Refine / SCoRe(iterative refinement)、
> PICARD(Scholak 2021,硬 grammar 约束式解码)。

---

## 4. L3 — 基于互信息的选择(MI-based Selection)

### 4.1 目标

在 top-K {h_k, p_k} 上回答:*哪些假设是"真实可区分"的,哪些应被合并或
放弃?* 用互信息下界衡量:

```
I(H; Y | D) ≥ E_{h~p, y~p(y|h,D)} [log p(y|h,D) − log (1/K) Σ_k p(y|h_k,D)]
                                    ↑ Prior-Contrastive Estimation (Foster 2020)
```

* 实现:用现有 PyBaMM 正演生成 y|h 样本(每 h 取 32 noisy 正演),
  PCE 估计器(Foster NeurIPS 2020)给出 MI 下界。
* 若语料/算力紧,退化版用 InfoNCE(Oord 2018)或 MINE(Belghazi 2018)。

### 4.2 候选合并(diversity filter)

对 top-K 做两两 **pairwise KL spread** 估计:

```
KL_ij = E_{y ~ p(y|h_i)} [log p(y|h_i) − log p(y|h_j)]
```

KL_ij < τ_merge 的两者聚合权重,留下 K'<K 个"真正不同"的假设。

### 4.3 IdentGap 与 defer 触发器

```
IdentGap = H(p_k) − log K'           (归一化的 posterior 熵 — log(effective K))
```

* IdentGap < τ_ret:直接返回 top-1 + 机理图 + 置信度。
* τ_ret ≤ IdentGap < τ_def:返回合并解释 + "仍不可区分的候选集"。
* IdentGap ≥ τ_def:进入 L4,让系统自己选下一次实验。

> **Top-conf 灵感**:Foster et al.(NeurIPS 2020, 2021)的 PCE/DAD 系列、
> MINE(Belghazi 2018)、InfoNCE(Oord 2018)、BALD(Houlsby 2011)、
> MacKay 1992 信息论实验设计。

---

## 5. L4 — 贝叶斯最优实验设计(BOED)

### 5.1 形式化

离散实验空间 E(例如 {EIS f-range, T-scan, SoC-scan, DRT, CV rate,
Galvanostatic pulse, …}),cost(e) 预估(时间、设备占用、样品损耗)。

```
e* = argmax_{e ∈ E}  EIG(e) − λ · cost(e)

EIG(e) = I(H; Y_e | D)
       ≈ PCE lower bound at (e, top-K hypotheses)
```

### 5.2 实现路径

* **MVP**:枚举 |E| ~ 20 个预设实验原型,对每个用 L3 的 PCE 估计器算
  EIG,取 argmax。每步 O(|E|·K·32) 次正演 — 在 CPU 可接受。
* **放大**:amortized BOED,DAD(Foster ICML 2021)/ iDAD(Ivanova 2021)
  训练一个策略网络 π_φ(e|D) 直接预测下一步实验,避免每步枚举。
* **代理环境**:建"experiment oracle"—给定 e,用高保真 PyBaMM(MPM / DFN)
  模拟 y_e;供 benchmark 评测,真实实验留到最终案例验证。

### 5.3 关闭反馈

y_e 观测后更新 D ← D ∪ {(e, y_e)},回 L2 以新 D 为条件再搜一轮。序列长度
T ≤ 5(经验设定,避免无限推迟答案)。

> **Top-conf 灵感**:Foster DAD(ICML 2021)、Ivanova iDAD、MacKay 1992、
> BALD(Houlsby 2011)、Coscientist(Boiko Nature 2023)、ChemCrow(Bran
> 2024)— 后两者是 LLM 驱动的闭环化学实验代理。

---

## 6. 评估指标(重新定义)

抛弃"judge.aggregate 单一标量",改为三联表:

| 维度                | 指标                               | 衡量什么                         |
|---------------------|-----------------------------------|----------------------------------|
| **识别力**          | Top-1 mechanism accuracy (oracle) | 有标注题上的正确率               |
| **可区分性**        | IdentGap, K'/K, MÎ(H;Y)          | 系统是否知道自己知道什么         |
| **实验经济性**      | EIG/cost, rounds-to-resolve       | 做到 IdentGap<τ 需要几次实验     |
| (辅)文本质量       | judge.aggregate(保留作 sanity)    | 可读性,不作为主指标             |

这样 C3full 那种"答得流利但不可区分"的情形会在 IdentGap 上暴露。

---

## 7. 与 §9E.1 证据的关系

| 观察                                   | 本架构在哪一层解决                     |
|----------------------------------------|---------------------------------------|
| candidate collapse (4 候选重合)         | L2 MCTS 的 KL-spread pruning + L3 merge |
| equifinality (谱拟合多解)               | L3 IdentGap → L4 主动取数              |
| judge bias (length/fluency)            | §6 指标改用 IdentGap + EIG,不以 agg 为主 |
| C3full Δagg 小                         | 因为指标本身失真,架构证明路径正确     |

---

## 8. 论文定位与新意

据我检索,**以上四层在电化学诊断领域从未被统一**:

* Coscientist / ChemCrow 覆盖 L4 但缺 L1 本体与 L3 MI 选择。
* AutoEIS / hallemans2024 覆盖 L2 参数搜索但无假设空间搜索。
* AlphaGeometry / FunSearch 提供 L1/L2 范式但不是电化学。
* Foster DAD 系列做 L4 但与机理识别解耦。

**新意声明**:首个把"参数化机理本体 + MCTS 假设搜索 + MI-based 选择 +
BOED 闭环"四件事耦合在一起,用于电化学原位诊断,并自带 library-growth
协议应对 OOV 机理的框架。

---

## 9. 参考文献(用于 paper §2 related work)

* Ellis et al. DreamCoder. PLDI 2021.
* Romera-Paredes et al. FunSearch. Nature 2024.
* Trinh et al. AlphaGeometry. Nature 2024.
* Silver et al. AlphaZero. Science 2017.
* Yao et al. Tree of Thoughts. NeurIPS 2023.
* Lightman et al. Let's Verify Step by Step (PRM). ICLR 2024.
* Madaan et al. Self-Refine. NeurIPS 2023.
* Kumar et al. SCoRe. 2024.
* Scholak et al. PICARD. EMNLP 2021.
* Zheng et al. NOTEARS. NeurIPS 2018.
* Bello et al. DAGMA. NeurIPS 2022.
* Bengio et al. GFlowNet. NeurIPS 2021.
* Foster et al. Variational Bayesian Optimal Experimental Design. NeurIPS 2020.
* Foster et al. DAD. ICML 2021.
* Ivanova et al. iDAD. NeurIPS 2021.
* Belghazi et al. MINE. ICML 2018.
* Oord et al. InfoNCE. 2018.
* Houlsby et al. BALD. 2011.
* MacKay. Information-based objective functions for active data selection. Neural Computation 1992.
* Beven. A manifesto for the equifinality thesis. J. Hydrol. 2006.
* Boiko et al. Coscientist. Nature 2023.
* Bran et al. ChemCrow. 2024.
* Hallemans et al. EIS ambiguity quantification. 2024.

---

## 10. 版本

v1 · 2026-04-24 · 初稿,对应 §9E.1 结果复盘后的架构重构。
下一版 v2 在 §9F 主实验 pilot 数据出来后修订 τ_ret/τ_def/λ 等阈值。
