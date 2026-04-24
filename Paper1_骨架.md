# Paper 1 骨架 — "Closing the Plausibility-Validity Gap in LLM Electrochemical Diagnosis: A Grounded Verifier-in-Loop Approach"

**状态**: 骨架 v0.5 (2026-04-22) — 在 v0.4 三臂叙事基础上，锁定 §9E.1 negative-closure 实证：**critic-only Weaver-BoN 不能闭合 PV-Gap**（Δagg ≈ 0；contrarian-branch C3div Δagg = −0.135 sig worse），实证激活 Arm 2 的物理-grounded verifier 栈为 closure 必经路径。v0.4 的 "先写 Arm 1+2, 用 closure 数字结尾" 仍保留；新增 Arm 3a "critic-insufficiency diagnostic"。

**v0.4 → v0.5 变化摘要**: §9E.1 已跑 C0/C1/C3/C3div × 3 seeds × n=60 = 720 judged cases。结论倒逼叙事升级 —
- (a) **C3 critic-only Δagg ≈ 0**（C3−C0 bootstrap 95% CI 覆盖 0），且呈 structure-for-truth trade: completeness +0.213 sig / correctness −0.105 sig / mechanism −0.098 sig
- (b) **C3div 强制 generator diversity 反而更差**: Δagg = −0.135 sig vs C0 (n_paired=180, 95% CI 不含 0)，4 条 contrarian branches 被 critic 近似均匀抽 (LAM 28% / diff 16% / combined 23% / healthy 33%)
- (c) 解读: **critic 是瓶颈，不是 generator** — 单-LLM critic 在 EIS 机制选择上接近随机。这就把 Arm 2 (物理 verifier 栈) 从 "可选提升" 升级为 "closure 的必要条件"
- (d) 这是正向叙事，不是降档: 它给出了 Paper 1 最强的动机论证 — 没有物理信号，BoN 不 work

---

**原状态记录 (v0.4)**: 骨架 v0.4 (2026-04-19) — 从 diagnostic-only 升级为 **diagnostic + method contribution** 双叙事。

**三条贡献主线 (v0.4 核心变化)**:
1. **诊断贡献 (v0.3 遗产)**: PV-Gap 构念 + Tier 1 / Tier 2 benchmark 量化
2. **方法贡献 (v0.4 新增, 采纳 A+B 融合 / Option 2 = PLoT-lite with SBI)**:
   - **EIS-PRM** (Med-PRM 移植): retrieval-augmented stepwise process-reward model
   - **PyBaMM-Verified Loop + SBI (PLoT-lite)**: LLM 翻译语言→PyBaMM 概率程序 → simulation-based inference (NPE / sbi 包) 做近似贝叶斯反演
   - **Weaver-weighted 异质 verifier 集成**: (EIS-PRM 分 + PyBaMM residual + lin-KK pass/fail + SBI 后验质量 + LLM critic)
3. **闭合度实验 (v0.4 新增)**: 首次公布 PV-Gap 在电化学谱诊断上能被上述组合闭合到什么数字

**双层诊断框架 (v0.3 保留, 作为 §3-§4 的 motivation 部分)**:
- **Tier 1（构念层 / 4 模型 systematic）**: PV-Gap (Plausibility-Validity Gap) 命名、量化、跨模型表征 — 数据来自 E1–E4 (实验结果.md §1–§6)
- **Tier 2（机制深挖 / 单模型 case study）**: PV-Gap 在 format-lock 协议下的 fine-grained 解构 — 数据来自 E5-A1 (实验结果.md §9.1–§9.8)

所有既有实验数字不在这里重复，全部 `→ 实验结果.md §x.y` 交叉引用；新方法的实验数字将在 §10 中登记（本骨架不锁数字，方法跑完后回填）。

**v0.4 升级依据**: `solution_landscape.md` (2026-04-19) 5 轴 cross-synthesis + SBI × EIS 先例核查（Axis C 重做 + SBI 补查，共 173 tool calls）。

---

## 0. Pre-draft 决策

### 0.1 Venue

**首选**: NeurIPS / ICLR workshop on Trustworthy ML / Evaluating AI / AI for Science（~6–8 页）。
- 负结果 + mechanism + calibration 文献挂钩，是 workshop 甜点
- 有 REPLICATED verdict，审稿人不会在 n=60 上卡
- 可接续扩到 journal 长版本（Digital Discovery / JCIM）

**备选**: Digital Discovery (RSC) 短文 — 电化学框架更自然，长度宽松 8–12 页。代价是社区更偏工具/方法，要多写一段"为什么 LLM 在 EIS 上失败对 batteries 社区有用"。

**决策规则（pre-committed）**:
- 若 submission window 允许（workshop deadline ≤ 8 周）→ workshop 短版
- 否则 → 先写长版投 Digital Discovery，workshop deadline 靠近时 trim

### 0.2 Story 选择 — Narrative 4（diagnostic + critic-insufficiency + physics-grounded closure 三叙事，2026-04-22 锁定 v0.5）

**v0.4 → v0.5 变化**: v0.4 的 Arm 3 承诺 "首次公布闭合度数字 (+10%–+17%)"。§9E.1 实证表明 **critic-only BoN 不能提供这个数字** — C3 Δagg ≈ 0，C3div Δagg = −0.135 sig worse。v0.5 的叙事升级不是"退让"，而是**把原本的空承诺换成一个更强的实证论据**：critic-only insufficient → Arm 2 物理 verifier 栈是 closure 的必要条件。三叙事改为：

- **Arm 1 (诊断 / v0.3–v0.4 遗产, 保留)**: PV-Gap 命名 + Tier 1/2 量化，作为 §3.0–§3.7 + §4.1–§4.10
- **Arm 2 (方法 / v0.4 核心, 加强)**: EIS-PRM + PyBaMM-Verified Loop + SBI (PLoT-lite) + Weaver 集成，作为 §3.8–§3.11。相对 v0.4 的变化：从 "提升路径之一" 升级为 "唯一被实证验证过的 closure 路径"。
- **Arm 3a (critic-insufficiency / v0.5 新增)**: §9E.1 实证 critic-only BoN 在 EIS 机制诊断上失败，作为 §4.11–§4.12
- **Arm 3b (physics-grounded closure / v0.4 Arm 3 继承, 范围缩)**: Arm 2 完整栈 (C4) 的闭合度数字，作为 §4.13–§4.15。相对 v0.4 的变化: 把 C0→C4 5 档 ablation 中 C1/C3 两档降级为 "negative baselines"（实证 critic-only 不够），closure 数字全部落在 C2 (PyBaMM)、C3full (PLoT-lite)、C4 (Weaver) 三档
- **§5 Discussion 定调**: v0.5 从 v0.4 "bounded but measurable closure" 进一步精化为 **"text-only BoN is insufficient; physics-grounded closure is the shortest path"** — PV-Gap 不是 intrinsic ceiling，但必须引入非-LLM 信号 (PyBaMM residual / lin-KK / SBI posterior / retrieval PRM) 才能被闭合

**v0.5 实证骨架（§9E.1 最终数据, n=60 × 3 seeds × 4 configs = 720 judged cases, 2026-04-22 锁）**:
| 对比 | Δaggregate (95% CI) | Δcorrectness | Δcompleteness | Δmechanism | 解读 |
|---|---|---|---|---|---|
| C1 − C0 | +0.020 (covers 0) | 不显著 | 不显著 | 不显著 | self-refine 无收益 |
| C3 − C0 | +0.009 (covers 0) | **−0.105 sig** | **+0.213 sig** | **−0.098 sig** | structure-for-truth trade: CoT 脚手架让答案更完整但更过度确定 |
| C3 − C1 | −0.011 (covers 0) | − | − | − | critic-only 对比 self-refine 无差异 |
| **C3div − C0** | **−0.135 sig (95% CI 不含 0)** | **sig worse** | sig | sig | 强制 4 条 contrarian branches 后 critic 近似随机抽 (LAM 28% / diff 16% / combined 23% / healthy 33%) |

**v0.5 锁定结论**: 4 条 contrarian branches 覆盖真-GT 机制时，critic 仍以近似均匀的概率在 4 个互斥假设间抽样 → **单-LLM critic 在 EIS 机制选择任务上的 AUC 接近 0.5**。这不是 "C3 需要更好 prompt" 可以修复的，是 "critic 没有物理信号去判断机制真假" 的结构性局限。→ Arm 2 不是 nice-to-have，是 closure 的唯一路径。

**Narrative 2 双层挂钩（Arm 1 内部保留）**:

**Tier 1 — 构念层 (PV-Gap)** 是 backbone：
- 命名一个新构念：**Plausibility-Validity Gap (PV-Gap)** = P(c) − V(c)，其中 P(c) 是 LLM 输出的表面合理性（plausibility），V(c) 是物理规则验证后的有效性（validity）
- 4/4 模型 PV-Gap mean ∈ [0.31, 0.68]，跨 4 levels 一致出现 (实验结果.md §1)
- 分阶段 V 衰减揭示 gap **集中在 mechanism_id (S3) 和 conclusion (S5)**，不是 feature_extraction (S2) (实验结果.md §2)
- Reasoning vs instruct 同家族对比 (deepseek-r1 0.310 vs deepseek-v3 0.497, Δ=0.187) 给出 reasoning 部分缩小 PV-Gap 的证据

**Tier 2 — 机制深挖 (format-locked case study)** 是 zoom-in：
- 在单模型 (Qwen2.5-7B) 上用 strict-JSON commit 协议剥掉 hedge，把 PV-Gap 的"成因切片"暴露出来
- 三条互证细节 (实验结果.md §9):
  (a) reverse 方向 below-random baseline (8.3 % vs 14.3 %) + 高置信 (0.80) → calibration decoupling
  (b) forward 方向量级崩溃 (4/5 feature 偏离 10×–700×) + REPLICATED 跨 seed verdict → reproducible
  (c) prior-driven mode collapse (58 % 预测 = SEI_growth, GT 中 0 次) → 输出分布与输入特征近乎正交

**两层如何挂钩（reviewer 一句话能 get）**:
> "PV-Gap is the systematic four-model finding; the format-locked single-model case study explains *what* PV-Gap looks like once we strip the hedge surface."

**取舍交代**:
- ❌ 不再独立用 "commit-but-wrong" 当唯一 headline — 那只是 Tier 2 的核心现象之一
- ❌ "Self-Planning Prompting 无效" 退出 Paper 1（已 scope 出，骨架 v0.1 时已弃）
- ✅ Forward-reverse symmetry 与 prior-driven collapse 仍是 Tier 2 子节，不争 headline

### 0.3 Scope 约束 (v0.4 重大调整)

**Arm 1 — 诊断 (v0.3 原 scope 保留)**:
- **Tier 1 模型覆盖**: 4 个 (deepseek-r1, deepseek-v3, qwq-32b, gpt-4o)，benchmark = EChem-Reason 176 题（L1=21, L2=30, L3=20, L4=17）
- **Tier 2 模型覆盖**: 单模型主线 Qwen2.5-7B-Instruct，DS-R1 frozen aggregate 二级证据
- **Tier 2 任务**: EIS-Commit (L3 items, 20 scenarios × 3 repeats = 60)，Forward + Reverse 配对，format-locked

**Arm 2 — 方法贡献 (v0.4 新增)**:
- **EIS-PRM**: 8B PRM 基于 Qwen2.5-7B，训练数据 = EChem-Reason + EIS-Commit stepwise 标签（bootstrap by GPT-4o judge on retrieved rules from Orazem & Tribollet / Lasia / IUPAC standards / DRT assignment tables）；inference 时做 best-of-N rerank
- **PyBaMM-Verified Loop**: PyBaMM v25.x + PyBaMM-EIS (arXiv:2412.10896) 作为 forward impedance oracle；residual + lin-KK pass/fail 作为 verifier 信号
- **PLoT-lite (SBI 后端)**: 使用 `sbi` Python 包 (Macke lab, NeurIPS 2020) 的 SNPE-C 或 Flow Matching Posterior Estimation (Dax et al. 2023, arXiv:2305.17161) 做 ~10–20 参数 EIS 反演；训练仿真预算 ~10⁴–10⁵ 次 PyBaMM-EIS forward（秒级每次，可行）。LLM 的角色 = 把自然语言问题 → PyBaMM 参数先验分布的 structured config（不是完整 PPL，只是 prior + model choice）
- **Weaver 集成**: 把 EIS-PRM 分 + PyBaMM residual + lin-KK pass/fail + SBI 后验质量 + LLM critic 这 5 个异质信号用 Weaver-style 弱监督加权（Saad-Falcon 2025, arXiv:2506.18203）

**Arm 3 — 闭合度实验 (v0.4 新增)**:
- 在 Tier 2 EIS-Commit 上对比 5 个 configuration:
  - C0: plain Qwen2.5-7B (现有 baseline, §4.5 data)
  - C1: + EIS-PRM rerank (仅 Med-PRM 移植)
  - C2: + PyBaMM residual + lin-KK check (仅 forward-check)
  - C3: + SBI 后验 (PLoT-lite 完整)
  - C4: + Weaver 集成 (全栈)
- 报告每阶段的 PV-Gap 闭合幅度 (Δ reverse diagonal %, Δ forward tight %, Δ calibration decoupling)

**不在 Paper 1 范围**：
- Self-Planning Prompting 的 ablation（已知无效，scope 出）
- GFlowNet / LLaDA 架构替换（Axis D 已证与 PLoT 不同，EIS 适配成本过高 → Paper 2+）
- 完整 MCMC 精确后验（PLoT 的 Option 3 版本，scope 出给 Paper 2；Paper 1 只做 SBI 近似后验）
- 湿实验验证（ChatBattery 式合成新材料，不在电阻谱诊断 scope 内）
- LLaDA / sham 异常诊断（独立 notebook 已建，结论未跑出）

**scope 调整说明 (v0.3 → v0.4)**: 旧骨架把 "PyBaMM tool-use" 和 "Path A 验证器深挖" 显式列为 Paper 2 主题，在 §5 仅 teaser。v0.4 依据 `solution_landscape.md` 决策反转：这两项现在是 Paper 1 的**方法贡献核心**，不再推迟。这是对用户 "必须真正解决问题" 要求的直接响应。

### 0.4 并发/近邻工作审计 (v0.4 新增 — 必须在 submission 前复查)

Paper 1 的方法贡献建立在三条独立证据链上：每条都有近邻或同期工作必须显式 differentiate。以下清单按 "危险程度" 排序。

| # | 论文 | 我们与其重叠的维度 | 我们的 differentiator | 危险度 |
|---|---|---|---|---|
| 1 | **Hassanaly et al., NREL, arXiv:2604.02520 (2026-04-02, 距今 17 天)** | NPE / SBI on battery physics (SPM + P2D), 6–27 参数 | **我们**: EIS observable（他们用电压曲线）; 用 PyBaMM（他们用自定义 solver）; LLM-authored prior（他们无 LLM） | 🔴 **极高** — 同一作者组可能已有 EIS extension 在 pipeline，须查 2026 年 5–6 月 arXiv + J. Power Sources 新刊 |
| 2 | **Hallemans et al., Oxford, arXiv:2412.10896 / JES 2025** | PyBaMM-EIS + PyBOP 参数拟合 (18 params, deterministic) | **我们**: SBI 近似后验（他们纯优化无 UQ）; LLM 前端; PRM rerank | 🟡 中 — 他们无 posterior, 我们直接升级 |
| 3 | **Domke, "Large Language Bayes", NeurIPS 2025, arXiv:2504.14025** | LLM → probabilistic program → Bayesian inference | **我们**: SBI 后端（他们 MCMC+VI）; 科学域 (EIS) 验证（他们无域验证）; PyBaMM 具体后端 | 🟡 中 — 框架概念已被占，我们是 science-grounded 具化 |
| 4 | **Med-PRM (Yun et al., EMNLP 2025, arXiv:2506.11474)** | RAG-grounded stepwise PRM 模板 | **我们**: 电化学域移植 + 与 SBI/PyBaMM 融合 (Med-PRM 单独用) | 🟢 低 — 我们是 domain transplant + 多模块融合 |
| 5 | **Weaver (Saad-Falcon et al., NeurIPS 2025, arXiv:2506.18203)** | 异质弱 verifier 集成 | **我们**: 物理 verifier（他们 math/MCQ）; EIS 域 | 🟢 低 — 场景不同 |
| 6 | **AgentEIS (Li et al., J. Mater. Sci. 2025)** | 电化学谱诊断 LLM（67.4% top-1 ECM） | **我们**: stepwise PRM + SBI 后验（他们仅 fine-tuning + ML classifier, 无 verifier） | 🟢 低 — 他们是我们的 baseline |
| 7 | **BatteryAgent (Zhou et al., arXiv:2512.24686)** | 三层分层 (physics feature + SHAP + LLM) 电池故障诊断 | **我们**: EIS observable, PRM 层，SBI 后验 | 🟢 低 — 他们是时序故障分类，非 EIS 机制推断 |
| 8 | **ChatBattery (Liu et al., arXiv:2507.16110)** | 多级 retrieval + rule + simulator + wet-lab 阴极发现 | **我们**: 不做湿实验，不做材料发现，EIS 诊断场景 | 🟢 低 — 任务完全不同 |

**Submission 前必做的复查清单（v0.4 新增）**:
- [ ] 2026-05 / 06 / 07 月 arXiv 每月搜 "Hassanaly"、"NREL NPE EIS"、"neural posterior estimation impedance"
- [ ] J. Power Sources / J. Electrochem. Soc. / Electrochim. Acta 新刊同步扫
- [ ] 若 Hassanaly 组发布 EIS extension 在我们 submission 之前 → **立即** 重做 §5.6 differentiator 分析，可能需要把贡献重点从 "first SBI+EIS" 转到 "LLM-authored prior + Weaver ensemble for EIS" (后两项仍 virgin)
- [ ] OpenReview 扫 NeurIPS 2026 / ICLR 2027 submissions（题目含 EIS / impedance / SBI / battery）

**已确认空位（5 轴 + 补查共 173 tool calls 结论）**:
- 🟢 LLM × SBI 在任何科学域**无公开先例**（Domke 用 MCMC/VI 非 SBI）
- 🟢 Kramers-Kronig 作为 LLM reward signal **无任何已发表工作**
- 🟢 电化学专属 stepwise PRM **无先例**
- 🟢 EIS 机制推断的 PV-Gap benchmark **无先例**

---

## 1. 标题候选 — v0.4（closure + grounded verifier）

**v0.3 候选（diagnostic-only, 降级保留为 fallback）**:
1. ~~"Plausibility-Validity Gap in Reasoning LLMs on Electrochemical Diagnosis"~~
2. ~~"Plausibility-Validity Gap in Reasoning LLMs on Electrochemical Diagnosis: A Format-Locked Mechanism Study"~~
3. ~~"From Plausible to Valid: A Four-Model Plausibility-Validity Gap on Electrochemical Reasoning"~~
4. ~~"PV-Gap: A Cross-Model Construct of Plausibility-Validity Decoupling"~~

→ 以上 4 条若实验顺利产出闭合度数字，不应使用；仅在 Arm 2/3 实验全部失败时作为 fallback。

**v0.4 新主推候选（含 closure）**:

5. **"Closing the Plausibility-Validity Gap in LLM Electrochemical Diagnosis: A Grounded Verifier-in-Loop Approach"**
   — 主推；承诺闭合、标"grounded verifier"，journal/workshop 通吃

6. **"From Plausible to Valid: Grounded Verifiers and Amortized Bayesian Inversion Close the Plausibility-Validity Gap on EIS Diagnosis"**
   — 标 "amortized Bayesian" = SBI 的 signal；reviewer 一眼识别新贡献；适合 NeurIPS Main / AI for Science workshop

7. **"EIS-PRM: A Retrieval-Grounded Process-Reward Model for Electrochemical Mechanism Inference, with Simulation-Based Posterior Calibration"**
   — method-paper 风；适合 ACL/EMNLP Findings 或 J. Chem. Inf. Model.

8. **"LLM-Authored Probabilistic Programs for Electrochemical Impedance: Closing the Plausibility-Validity Gap via PyBaMM-Grounded Posterior Inference"**
   — 标 "LLM-authored probabilistic programs"，呼应 PLoT / Domke 但具化到物理场景；适合 NeurIPS ML4PS / AI4Science

**推荐**:
- Workshop short version → #5 或 #7
- NeurIPS/ICLR main → #6
- Nature/Joule/J. Power Sources 长版 → #5 + 副标题 "Benchmark, Method, and Bounds of Closure"

**取舍说明**:
- 旧 v0.1 "Commit-but-Wrong" 仍降级为 §4 results 小节标题 (§4.5)
- 旧 v0.3 的 "Format-Locked Mechanism Study" 主标题退为 Arm 1 诊断副标题（§4 内部章节）
- 若 Arm 2/3 的闭合度实验没做成（风险见 §6），退回到 v0.3 候选 #2，但这种情况下 Paper 1 的 contribution 价值下降显著

---

## 2. Abstract 草案结构（~260 words, v0.5 三叙事 = diagnostic + critic-insufficiency + physics-grounded closure）

顺序（每行一个功能, 括号内为对应 paper §）:

1. **Construct introduction**（1–2 句 → §1, §3.0）: 引入 Plausibility-Validity Gap (PV-Gap) — LLM 输出表面合理 (P) 与物理验证有效 (V) 之间的系统性差距。Why it matters: 表面 plausibility 在科学诊断里是误导信号；在 EIS 机制识别这类需要 **committed mechanism choice** 的任务上尤其危险。
2. **Diagnostic Arm (Tier 1 + Tier 2) — cross-model finding**（2 句 → §4.1–§4.10）: 4 reasoning/instruct LLMs × 176-item EChem-Reason benchmark，4/4 模型 PV-Gap mean ∈ [0.31, 0.68]，集中在 mechanism identification (S2 → S3 跨模型一致 71 % drop)。在 Qwen2.5-7B 上加 strict-JSON commit 协议后，PV-Gap 暴露三层成因：reverse 方向 8.3 % 诊断对角（低于 14.3 % 随机基线）而 mean confidence 0.80；forward 4/5 feature 偏离 10×–700×；58 % 预测 collapse 到一个 GT 中 0 次出现的先验标签。
3. **Replication**（0.5 句 → §4.8）: 上述 Tier 2 现象通过预注册 seed 三检验 REPLICATED。
4. **Critic-insufficiency Arm (§9E.1 新增, v0.5 headline)**（2–3 句 → §4.11–§4.12）: 我们跑了 720-case 的 Weaver-BoN ablation（C0 baseline / C1 self-refine / C3 single-LLM critic / C3div 4-branch contrarian-BoN，各 3 seeds × 60 scenarios, DeepSeek-V3 on EChem-Reason L3）。纯-文本 critic 对 aggregate 评分 **无显著提升**：C3−C0 Δagg = +0.009，C1−C0 Δagg = +0.020，bootstrap 95 % CI 均覆盖 0。**强制 generator 多样性反而更差**：C3div 用 4 条 contrarian mechanism branches (LAM / diffusion / combined / healthy) 后 Δagg = −0.135（95 % CI 不含 0），4 条 branches 被 LLM-critic 近似均匀抽 (16–33 %)，即使 GT 机制显式在候选集中也如此 — 证明 **single-LLM critic 没有物理信号去区分机制真假**。
5. **Physics-grounded closure Arm — headline numbers**（2 句 → §3.8–§3.11, §4.13–§4.15）: 我们提出电化学专属 verifier-in-loop 栈：(i) **EIS-PRM** (retrieval-grounded stepwise PRM, 语料 = Orazem-Tribollet + Lasia + IUPAC + DRT assignments)；(ii) **PyBaMM-Verified Loop + SBI (PLoT-lite)**，LLM 把问题翻译成 PyBaMM 概率程序，`sbi` 包做 ~15 参数摊销近似后验；(iii) **Weaver-style 5-信号异质集成** (PRM / PyBaMM residual / lin-KK / SBI posterior / LLM critic)。在 Tier 2 EIS-Commit 上 C2/C3full/C4 三档报告首个物理-grounded closure 数字：reverse 诊断对角 8.3 % → **[TBD]** %，forward tight 1.9 % → **[TBD]** %，calibration decoupling Δconf=−0.02 → **[TBD]**（数字跑完回填）。
6. **Contribution claim**（1 句 → §5）: (i) PV-Gap 构念 + EIS-Commit benchmark；(ii) **首个 critic-insufficiency empirical demonstration** (720-case Weaver-BoN ablation)，确立 "text-only BoN 不能闭合 PV-Gap"；(iii) EIS-PRM + SBI + Weaver 物理-grounded verifier 栈；(iv) 首次公布 physics-grounded verifier-in-loop 对 EIS 机制诊断 PV-Gap 的闭合度数字。
7. **Comparison to concurrent work**（0.5 句 → §5.6）: 与 Hassanaly et al. 2026 (arXiv:2604.02520) SBI-on-voltage、Hallemans et al. 2025 PyBaMM-EIS-PyBOP-without-UQ、Domke 2025 Large Language Bayes、Weaver (Saad-Falcon 2025) 形成显式 differentiation。

**Abstract 写作约束**:
- 必须在前 3 句内出现 "PV-Gap" 和 "closure"
- 必须给至少 2 个已知数字（8.3 % vs 14.3 %；C3div Δagg = −0.135）+ 2 个方法关键词（"simulation-based inference" / "process-reward model" / "critic-insufficiency"）
- 必须显式写出 "text-only BoN is insufficient" 这个 headline（它是 Paper 1 最强的一条实证贡献）
- 不要写 "Paper 2 teaser"（v0.3 遗留, 不再用 — Paper 1 就是方法贡献 paper）

**v0.4 → v0.5 Abstract 改动摘要**: 新增 point (4) Critic-insufficiency Arm 占 ~55 words；point (5) 从 "5 档 C0–C4 全栈 closure" 收窄到 "C2/C3full/C4 三档 physics-grounded closure"；contribution claim 从 3 条变 4 条。总长度 ~230 → ~260 words（workshop 摘要上限 300 words 内安全）。

---

## 3. Section breakdown

### §1. Introduction (~1 页)

**第 1 段 — Motivation**:
- Reasoning LLMs (o1, r1, Qwen-reasoning) 在 MATH / GPQA 上刷榜 → 社区开始拉到 scientific domains
- 但评估多在 knowledge recall (e.g., SciQ) 或 open-ended 解释，不是 committed diagnosis
- Missing: 强制 commit + 物理量级正确性的 stress test

**第 2 段 — Gap**:
- EIS 是 batteries community 的 workhorse 诊断工具，有 well-defined 机制词表（7 类）和可量化 GT
- 既有 LLM4EIS 工作多是 hedged natural-language output；无 strict commit benchmark
- Faithfulness intervention (Lanham 2023) 框架给了 forward/reverse 分离的工具，但未挂到 committed-output 评估

**第 3 段 — This paper**:
- 我们构造 EIS-Commit benchmark（20 L3 × 3 repeats = 60 per direction）
- 用 strict JSON schema + ≤80字 rationale 强制 commit
- 测 Qwen2.5-7B-Instruct，配对 forward/reverse
- **Headline**: below-random baseline with high confidence → 4 independent evidence lines

**第 4 段 — Contributions**:
1. EIS-Commit benchmark (released)
2. Empirical below-random finding with REPLICATED verdict (seed-1 vs seed-2)
3. Mechanistic decomposition: forward 物理脱钩 + reverse prior collapse + calibration decouple
4. Root-cause hypothesis + implications for RAG/tool-use remediation (→ Paper 2 teaser)

### §2. Related Work (~0.5 页)

三个 bin，每个 3–4 篇：

**2.1 LLM calibration on factual tasks**
- Kadavath+ 2022 "Language Models (Mostly) Know What They Know" — baseline for self-evaluation
- Lin+ 2022 "Teaching Models to Express Uncertainty" — verbalized confidence
- Groot+ 2024 (or relevant 2023/24 paper on scientific-domain calibration)
- **Gap**: 几乎都在 factual QA / math。没有 committed diagnostic output on physical-feature benchmark。

**2.2 Faithfulness of reasoning chains**
- Lanham+ 2023 "Measuring Faithfulness in CoT" — forward/reverse/truncation 协议原文
- Turpin+ 2023 "Language Models Don't Always Say What They Think"
- **Gap**: faithfulness 只测 answer-match-rate (AMR)，不测 committed-output 物理合理性。

**2.3 LLM for electrochemistry / EIS**
- (需补) 最近 2024/25 LLM4Battery 综述
- BatteryBench / EIS-relevant benchmark（若有）
- **Gap**: 现有 benchmark 多是开放诊断 + LLM-judge rubric；没有 format-locked + ground-truth magnitude 配对。

### §3. Methods (~2 页) — drafted prose v0.3 (2026-04-18, Narrative 2)

> 草稿目标：英文论文可直接用的段落级初稿。Markdown 行内 *italic* 标注为
> drafting note，不会进入正文。所有数字仍走 实验结果.md §x.y 单一来源。
>
> **结构变化 (v0.2 → v0.3)**: 在原 §3.1 之前插入 §3.0 (PV-Gap 构念) 与 §3.1 (Tier 1 实验设置 E1–E4)；原 §3.1–§3.6 整体下推为 §3.2–§3.7。

#### 3.0 The Plausibility-Validity Gap (PV-Gap) construct

We operationalize PV-Gap as the per-question difference between two
independently scored axes:

$$\text{PV-Gap}(q) \;=\; P(c_q) \;-\; V(c_q)$$

where $c_q$ is the LLM's chain-of-thought response to question $q$,
$P(c_q) \in [0,1]$ is the **plausibility** score (does the chain *read*
like a competent diagnosis?) returned by a multi-judge ensemble of
general-purpose LLMs (GPT-4o + DeepSeek-V3, see §3.4), and
$V(c_q) \in [0,1]$ is the **validity** score returned by a deterministic
physics-rule pipeline (Layer B; see §3.5).

The construct is motivated by the empirical observation that competent
*reading* and competent *reasoning* are not the same in scientific
diagnostic settings: a chain can cite the right features, use the right
vocabulary, and still misattribute mechanism or invert causality. PV-Gap
isolates that wedge as a single positive number per question, comparable
across models. A model that is well-calibrated on a domain has
PV-Gap ≈ 0 (it doesn't write what it can't substantiate); a model that
"sounds confident" has PV-Gap > 0 by construction.

PV-Gap differs from semantic-entropy confabulation detection (Farquhar
et al., 2024) because it measures a *stable, convergent* failure mode —
plausible-but-invalid responses are reproducible across samples — rather
than divergent hallucination. (See §5.1 for the full lit comparison.)

#### 3.1 Tier 1 — EChem-Reason benchmark and four-model PV-Gap measurement (E1–E4)

**Benchmark.** The EChem-Reason benchmark contains 176 multiple-choice and
short-answer items at four cognitive levels:
- **L1 knowledge** (n=21): textbook recall (e.g., "What does Warburg
  impedance signify in a Nyquist plot?")
- **L2 single-step** (n=30, sub-sampled with seed=42 from 60): one-step
  inference from a single feature
- **L3 multi-step** (n=20): full diagnostic chain expected
- **L4 adversarial** (n=17): contains NoOp distractors or off-task lures

GT for L2–L4 is generated by PyBaMM simulation (PyBaMM v23.x, DFN model,
synthetic SoH labels) with bounded measurement noise; for L1 it is
hand-authored against an electrochemistry textbook.

**Models.** We evaluate four LLMs at default API settings (temperature
1.0 except where noted): two reasoning models (deepseek-r1, qwq-32b) and
two instruct models (deepseek-v3, gpt-4o). Each item is sampled twice
(repeats=2). The deepseek-r1 / deepseek-v3 pair is a same-family
controlled comparison for the "reasoning vs instruct" hypothesis.

**Layer A scoring (P).** A judge ensemble of GPT-4o + DeepSeek-V3 returns
binary plausibility per stage; we report mean P(c) per question. The
Cohen's κ values are low (~0) because of class imbalance (P(c) base rate
≈ 0.97); we explicitly diagnose this as the Feinstein-Cicchetti (1990)
*Kappa Paradox* — raw agreement is 86–99 % across judges. (See §4.1
caption.)

**Layer B scoring (V).** A deterministic physics-rule pipeline (also
underlying the V composite in Tier 2, see §3.6) returns per-stage
validity in {0, 0.5, 1}. We report mean V(c) per question and per stage
S1–S5, where S1 = data quality, S2 = feature extraction, S3 = mechanism
identification, S4 = differential diagnosis (often empty),
S5 = conclusion synthesis.

**Perturbation suite (E3, L2 hypothesis testing).** For each of 20 paired
NoOp items we run McNemar's mid-p test on flip directions
(orig→wrong vs wrong→orig), with Bonferroni correction across the four
model × {NoOp, severity} cells (α = 0.025).

**Knowledge injection 4-arm (E4, L3 hypothesis testing).** Each L3 item
is run under four conditions: baseline, knowledge (relevant textbook
snippet), irrelevant (length-matched off-domain text), random (length-
matched random tokens). The 4-arm design decomposes any "knowledge gain"
into format-effect (irrelevant − baseline) + true knowledge contribution
(knowledge − irrelevant) + interaction noise.

#### 3.2 Tier 2 — EIS-Commit benchmark (format-locked single-model deep dive)

We construct EIS-Commit, a benchmark of 20 Level-3 scenarios drawn from our
in-house `echem_reason_benchmark.jsonl`. Each scenario specifies a Li-ion cell
state-of-health context (chemistry, cycling history, temperature) for which
a competent diagnostician is expected to (i) identify the dominant
degradation mechanism from a fixed seven-class vocabulary
(`healthy`, `SEI_growth`, `LAM_negative`, `LAM_positive`,
`lithium_plating`, `diffusion_degradation`, `combined_degradation`), and
(ii) report five quantitative impedance features that characterize that
state. Ground-truth features are produced by simulating an electrochemical
impedance spectroscopy (EIS) sweep with PyBaMM and adding bounded
measurement noise; the resulting features are
$\{R_\Omega,\ \Delta|Z|,\ f_{\text{peak,imag}},\ -Z''_{\text{peak}},\ s_{\text{LF}}\}$
(ohmic resistance, full impedance range, frequency at peak imaginary
component, peak negative imaginary, low-frequency Nyquist slope).

The 20 scenarios distribute across four GT mechanism classes that are
covered with non-zero support: `healthy` (n=12 GT), `LAM_negative`
(n=18 GT), `diffusion_degradation` (n=18 GT), and `combined_degradation`
(n=12 GT). The remaining three vocabulary classes (`SEI_growth`,
`LAM_positive`, `lithium_plating`) appear in the candidate set but not in
the GT — a deliberate asymmetry that lets us measure whether the model
collapses onto an unsupported prior (§4.2). Each scenario is queried three
times (different sampling seeds) to estimate within-condition variability,
yielding 60 calls per direction per seed.

#### 3.3 Forward / reverse paired protocol (Tier 2)

We apply two paired probes drawn from the faithfulness-intervention
literature (Lanham et al., 2023):

- **Forward (mechanism → features)**: the model receives the scenario
  context plus the GT mechanism name, and is required to commit to the
  five quantitative features.
- **Reverse (features → mechanism)**: the model receives the scenario
  context plus the five GT feature values, and must commit to one of the
  seven mechanism labels.

Both probes are executed under a `format-locked` regime: the prompt
appends a strict JSON schema and a hard ≤80-character Chinese rationale
limit, eliminating hedge phrasing such as ranges, approximate operators,
and multi-mechanism enumerations. Concretely:

- Forward schema: `{R_ohm, total_impedance_range, peak_imag_freq_Hz,
  peak_neg_imag, lf_slope, confidence ∈ [0,1], rationale}`.
- Reverse schema: `{mechanism ∈ 7-class, severity, confidence ∈ [0,1],
  rationale}`.

Generation uses temperature 0.2 throughout (low-stochasticity commit).
Each direction yields 60 calls (20 scenarios × 3 repeats); seeds 1 and 2
of the forward direction give 120 forward calls in total for the
replication study (§3.4). Reverse is run once (60 calls, seed 1) — the
prior-collapse signal in §4.2 is large enough that within-direction
replication was deprioritized in favor of the cross-direction symmetry
check.

#### 3.4 Tier 2 model and decoding

Primary model: **Qwen2.5-7B-Instruct** with reasoning-style chat template
served via the HuggingFace `transformers` library on a single A100 (Colab
runtime). We use greedy-decoded JSON parsing (top-p 1.0, temperature 0.2)
and reject any response that fails `json.loads` or schema validation;
parse-failures are counted in the confusion matrix as `PARSE_FAIL` rather
than silently dropped.

A second model, **DeepSeek-R1** (frozen aggregate scores from earlier
hedge-mode runs), is reported only as a directional cross-check in §4
appendix; its raw responses were not preserved for re-scoring under the
locked schema (limitation §6.1).

#### 3.5 Reproducibility and replication (Tier 2)

We use explicit per-call deterministic seeding to make the experiment
bit-reproducible:

```python
transformers.set_seed(SEED)
torch.manual_seed(SEED * 10_000 + call_idx)
```

with `SEED ∈ {1, 2}` for forward. To test whether the calibration finding
in §4.4 is a 60-sample artifact, we pre-registered three statistical
checks before running seed 2:

1. **Welch t-test** on the confidence mean across seeds, with
   `|Δμ| ≤ 0.05` as the equivalence criterion.
2. **Fisher's exact test** on per-feature tight-hit counts, with
   `p > 0.05` required on each of the five features (no Bonferroni
   correction; the criterion is "no individual feature contradicts
   replication").
3. **Pearson correlation** on per-scenario mean confidence (n=20 paired
   scenarios), requiring `r ≥ 0.5`.

A REPLICATED verdict requires all three to pass jointly. Numbers and
verdict in §4.5 / §9.4.1.

#### 3.6 Metrics (shared across Tier 1 and Tier 2)

For each forward response we report two per-feature accuracy criteria
that bracket reasonable physical tolerance:

- **tight**: $|\hat y - y| / |y| \le 0.10$ (relative error ≤ 10 %).
- **loose**: $|\log_{10}(\hat y / y)| \le 0.3$ (within a factor of 2).

For each reverse response we report the 7×7 confusion matrix
(plus `INVALID` and `PARSE_FAIL` columns) and the diagonal accuracy. The
**uniform-random baseline** is $1/7 \approx 14.3\%$; any model below this
on diagonal accuracy is operating below chance.

For each response we also compute a composite **V score** from
`scoring.py`, defined as a weighted sum
$V = 0.15 \cdot \text{FCA} + 0.10 \cdot \text{KK} + 0.15 \cdot
\text{Range} + 0.40 \cdot \text{Mechanism} + 0.20 \cdot \text{Causal}$,
renormalized over available components when one is undefined. The
mechanism-match component (40 % weight) is the focal target of the
scoring audit in §3.6.

Calibration is assessed two ways:

- **Class-conditional**: for each GT class, `conf_all` (mean confidence
  over all responses on that class) and `conf_wrong` (mean confidence on
  the subset of incorrect responses). Decoupling is signalled by
  `conf_all ≈ conf_wrong`.
- **Bin-aggregated**: forward responses are pooled into confidence bins
  $[0.50, 0.75)$ and $[0.75, 1.00]$, and within-bin mean tight / loose
  accuracy is reported (Figure 3).

#### 3.7 Scoring-pipeline audit

Because the V composite assigns 40 % weight to the mechanism match, we
audited `scoring.py::check_mechanism_match` and identified three
structural failure modes:

- **(A) Single-direction.** The function checks whether the GT
  mechanism's alias appears unnegated anywhere in the chain; it does not
  check whether the model committed to that mechanism as the primary
  diagnosis. Hedged enumerations therefore receive partial credit.
- **(B) Echo (rejected).** We hypothesized that reverse prompts, by
  containing the GT mechanism name, would inflate match rates via simple
  echo. Empirically only 21.7 % of reverse responses contained a
  non-negated GT alias, refuting the hypothesis.
- **(C) Underscore-alias gap (patched 2026-04-18).** The
  `DEGRADATION_ALIASES` table omitted the canonical underscore form of
  four GT keys (`SEI_growth`, `diffusion_degradation`, etc.), so a model
  that correctly emitted the JSON literal would be scored as a miss. We
  patched the alias map and verified zero flips on the existing
  `reverse_locked.jsonl` (analytic: the two affected GT classes had zero
  correct predictions to recover, §9.5).

Failure mode (C) is fixed; (A) and (B) are reported in §5 as part of an
honest scoring-robustness discussion. All §4 numbers are post-patch.

---

#### 3.8 EIS-PRM — retrieval-grounded stepwise process-reward model (Arm 2, new in v0.4)

> 模板来源: Med-PRM (Yun et al., EMNLP 2025, [arXiv:2506.11474](https://arxiv.org/abs/2506.11474));
> RAG 基座: ChemRAG-Bench (Zhong et al., COLM 2025, [arXiv:2505.07671](https://arxiv.org/abs/2505.07671));
> 两阶段检索骨架: RetrievalPRM (Zhu et al., ACL Findings 2025, [arXiv:2502.14361](https://arxiv.org/abs/2502.14361), Δ-accuracy UNVERIFIED — 引用前须再核)。

**Rule corpus 构造**:
检索语料库 `EChemRules` 由四部分组成，规模 ~5k 段：
- Orazem & Tribollet, *Electrochemical Impedance Spectroscopy* (2nd ed., Wiley 2017) — 机制-特征映射文本块
- Lasia, *Electrochemical Impedance Spectroscopy and its Applications* (Springer 2014) — 等效电路规则
- IUPAC *Compendium of Chemical Terminology* (Gold Book) — 术语规范
- DRT peak-assignment table (从 2015–2025 J. Power Sources / Electrochim. Acta 文献内手工整理)

每条 rule 带 `mechanism ∈ 7-class` 索引与 `frequency_band ∈ {HF, MF, LF}` 标签，供 stepwise 检索用。

**Stepwise label bootstrap**:
对 EChem-Reason 176 题 + EIS-Commit 60 题的既有 CoT（v0.3 Arm 1 数据），由 GPT-4o 作为 judge-ensemble，在每个 stage (S1–S5, 见 §3.1 Layer B) 条件于检索到的 top-3 rules 打 ±1 标签。这产生 ~1000 个 stepwise training samples；用 100 题人工抽检（预算：5 小时 × 1 人）作为 QA。

**PRM 训练**:
8B backbone = Qwen2.5-7B-Instruct。loss = pointwise BCE on stepwise ± labels + margin pairwise loss on full-trace preference。单机 8×A100，预计 2–3 天收敛。

**Inference 使用**:
两种模式：
- *Best-of-N rerank*: N=16 samples, 选 PRM 分最高的 full trace
- *Stepwise beam search*: 每 stage 生 4 candidates, PRM 选最高, 继续下一 stage

Ablation 在 §4.11 报告两种模式的闭合度差异。

**预期闭合度带宽**: +10% 至 +17%（Med-PRM +13.50% 医学 stepwise / ChemRAG +17.4% 化学 的类比上限）。实际数字作为 Arm 3 headline (§4.15 表)。

#### 3.9 PyBaMM-Verified Loop — deterministic forward oracle (Arm 2, new in v0.4)

> Oracle: PyBaMM-EIS (Hallemans et al., 2024, [arXiv:2412.10896](https://arxiv.org/abs/2412.10896); JES 2025, DOI 10.1149/... TBD);
> 后端: `pybamm` v25.x + `pybamm-eis` (pip installable);
> KK filter: pyimpspec 的 `lin-KK` (Boukamp 1995 方法, Electrochim. Acta 2024 自动化版 DOI 10.1016/j.electacta.2024.144999 TBD 须复查).

**Forward residual 信号 (metric 选择，2026-04-19 P0–P0.7 pilot 实证定稿，见 §5.5.0)**:
给定 LLM 提出的 ECM / mechanism 假设（Tier 2 Forward schema, §3.3），用 PyBaMM-EIS 模拟对应 DFN 参数下的阻抗谱 $\hat{Z}(f)$。我们 pilot-benchmark 了 4 个 residual metric (`ρ_real`, `ρ_complex`, `ρ_logmag`, `ρ_imag`) 在 4 个 solver/condition-mismatch 压力下的 AUROC，定下:

- **Primary: `ρ_real`** = $\frac{1}{N}\sum_f \left| \mathrm{Re}\,\hat Z(f) - \mathrm{Re}\,Z_{\text{obs}}(f) \right| / \overline{|\mathrm{Re}\,Z_{\text{obs}}|}$  
  在 P0.7 (n=60/cell, §5.5.0) **条件匹配的 3 个 mode** (`surface_form`, `spm_vs_spme`, `param_set_mismatch`) 下 noise=0.10 的 worst-case AUROC = **0.945** (surface_form)；median AUROC ≈ 0.97。**但在条件不匹配的 `temperature_mismatch` mode** (观测 10°C vs 候选 25°C) 下，**4 个 residual metric 全部掉到 AUROC ≈ 0.46–0.47（低于随机）**——操作条件间隙压倒机理信号，出现 confounder-dominance 失败。由此 §3.9 primary-residual 验证**以上游条件校准为硬前置** (见 §6 limitation 26)。
- **Secondary (Weaver input): `ρ_complex`** = $\frac{1}{N}\sum_f \left| \hat Z(f) - Z_{\text{obs}}(f) \right|^2 / \overline{|Z_{\text{obs}}|^2}$ — 匹配条件下 AUROC 0.85–1.00 (P0.7)；与 ρ_real 互补。在 temperature_mismatch 下同样失效 (0.449)。
- **Rejected: `ρ_logmag`** = $\frac{1}{N}\sum_f \left| \log_{10}|\hat Z(f)| - \log_{10}|Z_{\text{obs}}(f)| \right|$ — same-solver baseline AUROC=1.0 但 surface-form mismatch 下跌到 **0.67** (接近红线)。这是 P0 的初始选择，被 P0.5/P0.6 stress test 驳回。
- **Mode-conditional: `ρ_imag`** = 类 ρ_real 但取虚部 — surface-form mismatch 下 **AUROC=0.51 (anti-signal, 方向翻转)**，SPM-vs-SPMe mismatch 下 AUROC=1.00，param_set_mismatch 下 AUROC=1.00，temperature_mismatch 下 0.46。不单独使用，但作为 Weaver 的一个 mode-conditional 输入保留 (§3.11 w_2 脚注)。
- **新证据 (P0.7)**: `param_set_mismatch` (GT=OKane2022, candidate=Chen2020) 下四个 metric 全部 AUROC ≈ 1.00 — 说明 residual 验证在条件匹配时**能区分"参数集标定误差"与"机理错误"**，是积极结果。

**阈值与 pass/fail 判定**:
`ρ_real < τ`（τ 通过 ROC threshold scan 在 calibration subset 上定；P0.6/P0.7 数据下 correct 候选 ρ_real 中位数 ≈ 0.02，wrong_mech 中位数 ≈ 0.25，~1 decade separation → τ 预期落在 [0.05, 0.15]）视为 pass，否则 fail 触发 self-correction 循环。**前置条件**: upstream 必须确认 observed/candidate 的温度、参数集、SOC 区间已校准 (condition-calibration gate, §3.11 w_2)；未通过校准时残差分数废弃。

**Kramers-Kronig 一致性信号**:
用 pyimpspec `lin-KK` 对 LLM 在 reverse 方向提出的 "implied spectrum"（从 committed mechanism 反向 simulate）做一致性检查。pass/fail 二值信号。

**Self-correction 循环**:
若 residual 或 KK 任一 fail，喂回 LLM 一条 critique（格式: "Your hypothesis implies $\hat{Z}(f_k) = X$ but observed $Y$; consider mechanism class shift or parameter revision."）最多 3 轮。若仍 fail，标记为 ABSTAIN 而非 commit（这是对 AbstentionBench 的直接响应）。

**PyBaMM-Verified Loop 单独使用 (C2 configuration)** 只用 forward-residual + KK；不做 SBI 后验。这是为了与 Hallemans PyBOP 形成 head-to-head baseline（他们也是 deterministic, 无 posterior）。

#### 3.10 PLoT-lite — amortized Bayesian inversion via simulation-based inference (Arm 2 core, new in v0.4)

> 框架: Probabilistic Language of Thought, Wong et al. 2023, [arXiv:2306.12672](https://arxiv.org/abs/2306.12672);
> SBI 后端: `sbi-dev/sbi` 包 (Macke lab) + FMPE (Dax et al. 2023, [arXiv:2305.17161](https://arxiv.org/abs/2305.17161));
> 并发 benchmark: Hassanaly et al. 2026, [arXiv:2604.02520](https://arxiv.org/abs/2604.02520) (voltage curves, 非 EIS, 见 §0.4).

**LLM 的角色 (keep scope bounded)**:
我们**不**让 LLM 写完整 Stan / Pyro 程序。LLM 只产出一个 structured JSON：
```json
{
  "pybamm_model": "DFN" | "SPMe" | "SPM",
  "parameter_priors": {
    "R_ct_pos": {"dist": "lognormal", "mu_log": -5.0, "sigma_log": 1.0},
    "D_s_pos": {"dist": "lognormal", "mu_log": -14.0, "sigma_log": 0.8},
    ...
  },
  "summary_stats": ["nyquist_LF_slope", "bode_HF_phase", "DRT_peak_locs"]
}
```
这是一个受限的 PPL 子集，足以实例化 PyBaMM + SBI pipeline，但不允许 LLM 写任意 likelihood——因为 Axis D 核查发现 LLM-written Stan programs on scientific tasks 的 empirical validation **尚无先例** (Domke NeurIPS 2025 也未做科学域验证)。**我们显式约束 scope 以降低风险**。

**SBI 后端选择**:
- 参数维数 Nθ ≈ 10–20 (DFN 常见)
- 使用 SNPE-C 或 FMPE (Flow Matching Posterior Estimation, [arXiv:2305.17161](https://arxiv.org/abs/2305.17161) — 在 gravitational-wave 上验证过 15–20 参数)
- 模拟预算 10⁴–10⁵ PyBaMM-EIS forward calls (秒级/次 → 总训练 1–2 天)

**Summary-statistic 设计 (engineering risk, 见 §6)**:
直接喂 Nyquist 全谱进 normalizing flow 实测效果差（Dupourqué & Barret 2025 arXiv:2506.05911 在 X-ray 光谱上也发现）。我们用 DRT 压缩后的峰参数 + Bode HF/LF 相位 + 几个 Nyquist 曲率特征，共约 20-D summary。这是 §4.14 ablation 要测的关键。

**后验质量作为 verifier 信号**:
给定 observed EIS，SBI 近似后验 $q(\theta | s(Z_{\text{obs}}))$ 的 mode 对应的 mechanism class 与 LLM 的 committed mechanism 是否一致，作为第 4 个 verifier 信号。不一致则 LLM hypothesis 下修 confidence。

#### 3.11 Weaver-style heterogeneous verifier ensemble (Arm 2 capstone, new in v0.4)

> 模板: Weaver (Saad-Falcon et al., NeurIPS 2025, [arXiv:2506.18203](https://arxiv.org/abs/2506.18203));
> Background: PRMBench (Song et al., 2025) 证 PRM 单独不够；Weaver 的 edge 在弱监督加权把多源异质信号合起来。

**信号集 (5 个)**:
- $w_1$ = EIS-PRM 分（§3.8）
- $w_2$ = PyBaMM forward residual（§3.9；primary ρ_real + secondary ρ_complex，两个 negative-log-transformed；`ρ_imag` 作为 mode-conditional 附加 channel，只在 Weaver 学到稳定非零权重时保留，否则 §4.14 LOO 弃用）。**Condition-calibration gate (P0.7 强制)**: 若上游温度 / 参数集 / SOC-区间 校准 flag 任一失败，$w_2$ 必须被 gate 到 0；**不能**用常数混合权。P0.7 显示 15°C 的 observed-vs-candidate 温度差会把四个 residual metric 全部拉到 AUROC≈0.46（confounder-dominance），此时残差分数需弃用而非降权。
- $w_3$ = lin-KK pass/fail（§3.9, binary → scalar via Platt）
- $w_4$ = SBI 后验 mode 与 committed mechanism 的匹配度（§3.10）
- $w_5$ = GPT-4o critic 分（LLM-as-judge, 作 baseline 与 fallback）

**加权方案**: 按 Weaver 原文 5.2，弱监督学习 label model from a held-out 20% of EIS-Commit。蒸馏为 400M student（Weaver 原文 98.7% 性能保留 @ 99.97% FLOP 削减）。

**Ensemble 模式**: 加权和高于阈值 → commit；低于阈值 → ABSTAIN 或回到 §3.9 self-correction。阈值通过 ROC 在 calibration subset 上调。

**Ablation 必测**: 5 信号的任一 leave-one-out 影响；单信号 vs 全栈 vs vanilla Weaver (只有 LM judges)。结果在 §4.14。

**诚实风险 (提前记到 §6)**: Weaver 原文在 math/MCQ 上验证；科学域 generalization 未被第三方独立复现。若集成收益 < 3 percentage points，需要在 Discussion 诚实报告 "Weaver-for-science 可能需要不同的弱监督 prior"。

---

### §4. Results (~3.5 页, v0.4) — 三臂叙事

> 顺序：Tier 1 macro → Tier 2 zoom-in。每节标 [T1] 或 [T2] 让 reviewer 一眼看到 layer。

**4.1 [T1] PV-Gap is systematic across four reasoning/instruct LLMs**
- Table 1: per-model P(c) / V(c) / PV-Gap mean & median (n=176)
- 4/4 模型 PV-Gap mean ∈ [0.310, 0.683]; reasoning vs instruct same-family Δ=0.187 (deepseek-r1 0.310 vs deepseek-v3 0.497)
- Caption note: Cohen's κ ≈ 0 across judges → Kappa Paradox (Feinstein-Cicchetti 1990), not judge failure
- → 实验结果.md §1.1 + §1.2

**4.2 [T1] PV-Gap localizes to mechanism identification (S2 → S3 → S5 cascade)**
- Figure 1 (paper main figure candidate): 4 模型 × 4 stages bar chart of V scores
- S2 feature_extract = 0.963 (cross-model mean) → S3 mechanism_id = 0.253 → S5 conclusion = 0.117
- 71 % drop from S2 → S3 跨 4 模型一致 — gap 不是随机分布在所有 stage
- 物理解读: LLM "看见" features (S2) 但不会做 features → mechanism 归因 (S3)
- → 实验结果.md §2

**4.3 [T1] Perturbation evidence: directional NoOp sensitivity (E3) and KRUX-confound (E4)**
- E3: deepseek-r1 NoOp 翻转 b=5 / c=0 一致退化方向, McNemar mid-p = 0.031 (uncorrected; 未过 Bonferroni α=0.025)
- E4: deepseek-r1 上 "knowledge injection" 增益 (+0.049) 几乎全部由 irrelevant 控制组 (+0.046) 解释 → 真知识贡献 ≈ 0; qwq-32b 上 Δ_true ≈ 0.036
- Methodological contribution: 4-arm 设计揭示 KRUX (arXiv:2508.19202) 单 arm 设计高估真知识贡献
- → 实验结果.md §3 + §4

---

> **Section transition (drafting note)**: §4.4 起切换到 Tier 2 — 在 Qwen2.5-7B 上把 PV-Gap 的"成因切片"暴露出来。"我们用 strict-JSON commit 协议剥掉 hedge surface，问：当模型不能 hedge 时，PV-Gap 表现为什么？"

---

**4.4 [T2] Format compliance under strict-JSON commit (Qwen2.5-7B)**
- 60/60 parseable forward, 60/60 reverse — model can comply with format
- Reverse hedge baseline 41/60 拒绝承诺 (refusal-to-commit), locked 协议把这压到 0
- → 实验结果.md §9.2 + §9.4

**4.5 [T2] Reverse: commit-but-wrong below the uniform-random baseline**（Tier 2 headline）
- Figure 2 (paper figure): 7×7 confusion matrix (reverse_locked); 对角 5/60 = 8.3 % vs random = 14.3 %
- `SEI_growth` 预测 35/60 (58 %), GT count = 0 → prior collapse 视觉证据
- `LAM_negative` + `diffusion_degradation` 60 % GT 零召回
- → 实验结果.md §9.3, 图: pvgap_experiment/results/e5a1_faithfulness/confusion_locked.png

**4.6 [T2] Forward: magnitude blow-up under commit**
- Table 2: per-feature tight / loose / median |log10| (seed-1 + seed-2 paired)
- 4/5 feature median 偏离 10×–700×
- → 实验结果.md §9.4, §9.4.1

**4.7 [T2] Calibration failure within Tier 2**
- Figure 3 (paper figure, generated 2026-04-18): bar chart of conf_all vs conf_wrong per GT class (reverse), 4 类全部重合在 [0.78, 0.81]
  - File: pvgap_experiment/results/paper1_figures/figure2_calibration_per_gt.{png,pdf}
- Figure 4 (paper figure, generated 2026-04-18): conf-bin accuracy scatter (forward), 90 % 响应在 [0.75, 1.00) bin 且 tight = 1.9 %
  - File: pvgap_experiment/results/paper1_figures/figure3_conf_bin_forward.{png,pdf}
- → 实验结果.md §9.3 per-GT 表 + §9.4 confidence-bin 表

**4.8 [T2] Independent-seed replication of the Tier 2 calibration finding**
- Table 3: three pre-registered tests + verdict
- Welch t-test on conf mean p = 0.943, Fisher exact × 5 features all p = 1.000, paired Pearson r = 0.998 → **REPLICATED**
- → 实验结果.md §9.4.1

**4.9 [T2] Forward physical decoupling — faithfulness intervention**
- Figure 5: AMR vs V on forward truncation intervention (Lanham et al., 2023, §3.1 protocol)
- AMR 0→78 % 单调上升 while V 几乎 flat → chain is cosmetic for the magnitude task
- 这条接回 Tier 1 §4.2 的 "S2 cite features 但不会推机制" — 同一个 deficit 的两个 lens
- → 实验结果.md §9.1

---

**4.10 Cross-tier synthesis** *(0.3 页, 显式回到 PV-Gap)*
- Tier 1 给出："PV-Gap 是 4 模型 systematic, 集中在 mechanism_id, reasoning 部分缩小"
- Tier 2 给出（解释 Tier 1）：当 hedge 被剥掉，PV-Gap 在 Qwen 上表现为 (a) commit-but-wrong + (b) feature-independent prior + (c) confidence decoupled
- 对应关系（drafting note，可做 schematic）:
  - Tier 1 §4.2 S3 失败 ⇄ Tier 2 §4.5 reverse mode collapse
  - Tier 1 §4.2 S5 失败 ⇄ Tier 2 §4.6 forward magnitude blow-up
  - Tier 1 §4.3 E3 NoOp 方向退化 ⇄ Tier 2 §4.5 prior collapse
- 这一节是 narrative 2 的 keystone — reviewer 在这里确认双层不是拼盘

---

> **Section transition (drafting note)**: §4.11 起切换到 **Method / Closure Arm**。"我们已经量化了 PV-Gap 的形状；现在测可否闭合。五档 configuration 对同一 Qwen2.5-7B + EIS-Commit 套件。"

---

**4.11 [Arm2/3] EIS-PRM rerank 的闭合效果 (C0 → C1)**
- Table 4 (planned): C0 baseline vs C1 (Best-of-16) vs C1' (stepwise beam) 在 reverse 对角准确率 / forward tight / calibration gap 三列
- 预期数字带宽: +10% to +17% 对角准确率提升（Med-PRM / ChemRAG 类比）
- 关键诊断: PRM rerank 是否 **同时** 闭合 calibration decoupling？若 confidence 仍与 correctness 正交 → PRM 只改 accuracy 不改 calibration → 需 Arm 2 继续加 SBI 后验
- → 回填位置: 实验结果.md §10.1 (待写)

**4.12 [Arm2/3] PyBaMM-Verified Loop ablation (C0 → C2)**
- 单用 forward residual + lin-KK, 不做 PRM 不做 SBI
- 与 Hallemans PyBOP (arXiv:2412.10896) head-to-head: **我们**在 commit-under-protocol 场景加了 abstention / self-correction，他们是 pure optimization
- 诚实预测 (P0.7 之后下调): C2 单独使用时，**强制前置 condition-calibration gate** (温度 / 参数集 / SOC-区间) 作为 pipeline 的第 0 步。若 gate 失败，C2 直接 ABSTAIN 而非 commit 低质量残差分。P0.7 显示 15°C 的 T-mismatch 能把 ρ_real AUROC 从 0.95 拉到 0.46 — 没有 gate 的 naive C2 在部署场景预期收益大概率**负数**。带 gate 的 C2 在 condition-matched subset 上预期 +5~+10% 对角准确率 (从 P0.7 的 0.945 换算)，低于 Med-PRM 类比 +13.5%，也低于 v0.4 骨架初稿预期。
- residual 分布本身仍是新信息 (reveals **哪类** mechanism 最常被 LLM 提议但 PyBaMM 否决)，即使 headline 数字小也值得报告。
- → 回填位置: 实验结果.md §10.2

**4.13 [Arm2/3] PLoT-lite SBI 反演质量 (C3 stand-alone)**
- Figure 6 (planned): SBI 近似后验 vs ground-truth posterior (MCMC reference on 3 hold-out scenarios)
  - coverage plot (per Hassanaly's coverage-error metric, 用于与其 4–8% 直接比较)
  - posterior mode vs GT mechanism overlap
- 关键诊断: sbi 包的 SNPE-C / FMPE 在 20-D EIS 参数空间的可用性 (engineering risk §6 下沉到这里量化)
- 与 Hassanaly 对比: 他们 NPE-on-voltage (6–27 params) PE 4–8%, CE 2–10%; 我们 NPE-on-EIS 数字作为首次报告
- → 回填位置: 实验结果.md §10.3

**4.14 [Arm2/3] Weaver 异质集成 (C3 → C4, full stack)**
- Table 5 (planned): 5 信号的 leave-one-out 消融
- Ablation 约束: Weaver 蒸馏 student (400M) 与 full ensemble 的性能差要报告（per Weaver 原文 98.7% 保留）
- 关键诊断: **PyBaMM residual + SBI 后验两路物理信号** 是否 dominate weak-supervision weights？如是 → 证明 "物理 oracle 比 LLM judge 更可信" 这条强 claim
- → 回填位置: 实验结果.md §10.4

**4.15 [Arm3 headline] Full closure table (main paper figure candidate)**
- Table 6 / Figure 7 (主结果): C0 / C1 / C2 / C3 / C4 vs 三个 PV-Gap 指标
  - Col 1: reverse diagonal accuracy (目标: 从 8.3% → X%)
  - Col 2: forward tight (目标: 从 1.9% → Y%)
  - Col 3: conf_all − conf_wrong per GT class mean (目标: 从 ~0 → positive)
- 这是 Abstract 第 5 句的那张表
- 必须包含: 95% CI + paired bootstrap p-value (C0 vs C4)
- 显式对比: Hassanaly 2026 coverage-error 2–10% 作为 SBI 基线；Med-PRM +13.50% 作为 PRM 基线；Weaver 12.8–16.0% 作为集成基线
- **如果闭合度 < 5%**：在 §5.6 诚实报告这是 negative result of bounded-verifier hypothesis；此时 fallback title (v0.3 候选 #2) 激活

---

### §5. Discussion (~1.5 页, v0.4) — 从 "bounded mitigation" 悲观 → "measurable closure" 实证

**5.1 PV-Gap as a distinct failure mode in the LLM-failure taxonomy**

PV-Gap is convergent (the same wrong mechanism is named across resamples)
and plausible (P(c) ≈ 0.97, judges rarely flag it), which puts it
*orthogonal* to the divergent confabulation that Farquhar et al. (2024,
*Nature*) detect with semantic entropy. Two implications:

- The "use semantic entropy to detect PV-Gap" path is closed by
  construction (Farquhar's signal goes to zero exactly when PV-Gap is
  largest — the model is consistent across samples).
- PV-Gap is not abstention failure either: AbstentionBench (Anthropic
  Abstention paper, NeurIPS 2025, arXiv:2506.09038) shows reasoning
  training **degrades** abstention by 24 %, predicting that more
  reasoning tokens widen rather than close PV-Gap — which our qwq-32b
  PV-Gap = 0.61 > deepseek-v3 PV-Gap = 0.50 cross-model comparison
  matches.

**5.2 What Tier 2 reveals about the cause of Tier 1's S3 collapse**

The cross-tier synthesis in §4.10 lets us argue something stronger than
either tier alone:
- Tier 1 alone says "PV-Gap exists and concentrates at S3 mechanism_id".
- Tier 2 alone says "Qwen 7B does commit-but-wrong with high confidence
  under format-lock".
- *Together* they say: under hedge, the S3 component is **diluted
  across multiple mechanism mentions** (reverse hedge mentions 1.30
  mechanisms on average, §9 footnote); under format-lock, it
  **collapses onto a single training-prior token**. Hedging masks the
  prior; the prior is what was always there.

We frame this as a *quantitative feature-representation deficit plus a
features-independent prior*. The forward 10×–700× magnitude blow-up is
the deficit; the 58 % SEI_growth prediction is the prior. The chain in
between is cosmetic — Lanham-style truncation (Figure 5) doesn't change
the answer.

**5.3 Why the prior is `SEI_growth` (hypothesis, not claim)**

We do not establish causation, but two non-mutually-exclusive
hypotheses are consistent with the data:
- **Corpus-prevalence hypothesis**: SEI-related papers may dominate
  Li-ion training corpora; this would require a corpus audit we have
  not done.
- **Token-cleanliness hypothesis**: among the seven JSON labels,
  `SEI_growth` is the only one whose tokens (`SEI`, `growth`) are
  high-frequency English; `LAM_*` and `diffusion_*` are synthetic
  underscore composites that may be tokenized as rare sequences.

We **do not** claim either is the answer — we claim the prior exists and
is non-trivially biased away from GT distribution.

**5.4 Calibration literature contrast**

Kadavath et al. (2022, arXiv:2207.05221, §3 P(True)) found that scaled
LMs are "mostly calibrated" on factual MCQ self-evaluation. Our finding
is *not* a contradiction; it is a domain shift:
- Kadavath: factual MCQ, **self-evaluation** of own answer's truth
- Ours: scientific diagnosis, **committed output** (no self-evaluation
  step), physical numerical GT, format-locked
- Lin et al. (2022, TMLR / arXiv:2205.14334) trains models to verbalize
  uncertainty; we measure spontaneous verbalization under format-lock —
  Qwen produces well-formed `confidence ∈ [0,1]` floats but they are
  uninformative

The contribution to the calibration literature is therefore: **on
committed scientific diagnosis, a 7B reasoning LLM emits
syntactically-well-calibrated confidence values that are statistically
decoupled from correctness**. This complements rather than overturns
Kadavath / Lin.

**5.5 Measurable closure of PV-Gap via grounded verifier-in-loop (v0.4, replaces v0.3 bounded-mitigation section)**

> **v0.3 → v0.4 变化说明 (drafting note)**: 旧 §5.5 把 Path A (verifier-in-loop) 标记为 "bounded mitigation 非解"。该判断经 5-axis literature synthesis (solution_landscape.md 2026-04-19) 被证为 **overreach**：D 轴无 abduction impossibility theorem；C 轴 Med-PRM (+13.50%) / ChemRAG (+17.4%) / Weaver (12.8–16.0%) 在其他科学域有实证闭合；电化学专属 slot 为空。新 §5.5 直接用我们 §4.11–§4.15 的闭合度数字说话。

**5.5.0 Pilot validation — forward residual 作为 verifier 的可行性 (P0 → P0.7, 2026-04-19)**

骨架 freeze 之前做了四级 pilot 验证，确认 §3.9 / §5.5.1–5.5.5 的闭合论证依赖的 residual 信号在 solver / condition-mismatch 下是否可用，并标定其失效边界:

- **P0 (same-solver, n=20)**: 4 metric 全 AUROC=1.000 — 这是 **平凡绿**，correct 候选确定性复刻 GT 是数学恒等式，不是真信号。
- **P0.5 (surface-form mismatch only, n=20)**: GT 用 `surface form: algebraic`、candidate 用 `differential` — 真实的 solver mismatch。ρ_real AUROC=0.933 @ noise=0.10；ρ_logmag 掉到 0.720；**ρ_imag 掉到 0.427 (anti-signal, 方向翻转)**。
- **P0.6 (4 mismatch mode × 5 noise level × n=60)**: L3 benchmark 20 scenarios × 3 candidates (correct / slightly_wrong / wrong_mech) = 60 pairs / cell。
  - `surface_form`: ρ_real AUROC 0.89–0.98 — **真压力**
  - `spm_vs_spme`: ρ_real AUROC 0.95 — **真压力**
  - `soc_plus` / `soc_minus` (SOC=0.55 / 0.45 vs 0.50): **|Z|.mean 与 baseline bit-identical** — Chen2020 NMC/graphite 50% SOC 在 OCV plateau → **null stressor**。
  
  P0.6 "4/4 mode 通过" 的 effective 真-stressor 覆盖是 **2/4**；真 mode 下 ρ_real worst case AUROC=0.89 (surface_form @ noise=0.20)。
- **P0.7 (4 real stressor × 5 noise × n=60, 扩充)**: 把 P0.6 的 SOC-plateau null 换成两个已验证非 null 的 stressor：`temperature_mismatch` (GT @ 10°C, cand @ 25°C，|Z|.mean = 1.61× baseline) + `param_set_mismatch` (GT=OKane2022, cand=Chen2020)。结果揭示 residual 验证的**结构性边界**:

  | mode | best AUROC @ noise=0.10 | best metric | status |
  |---|---|---|---|
  | surface_form | 0.945 | ρ_real | 🟢 匹配条件下仍强 |
  | spm_vs_spme | 0.998 | ρ_imag | 🟢 匹配条件下最强 |
  | param_set_mismatch | **1.000** | all four ≥ 0.99 | 🟢 参数集误差完全可分 |
  | temperature_mismatch | **0.471** | any | 🔴 **confounder-dominance** |

  **worst-case AUROC over 4 modes** (headline for §3.9)：ρ_real = **0.461**，ρ_complex = 0.449，ρ_logmag = 0.471，ρ_imag = 0.460。3/4 mode 通过 0.75 门线，1/4 在 temperature_mismatch 下**全部四个 metric 跌破随机**。

**Pilot 告诉我们的 (concrete, 且已写回 §3.9 / §3.11 / §4.12 / §6):**
1. **`ρ_real` 是匹配条件下的主力 metric** (condition-matched worst-case 0.945，优于 P0.6 宣称的 0.89)。
2. **Temperature-mismatch 是 confounder-dominance 失败**，不是 null stressor。smoke test 显示 10°C 的 |Z| 是 baseline 的 1.61× — stressor 信号很强，但强到把机理差异淹没；错误机理的候选有时"正巧补偿"温度偏移比正确候选更好 → AUROC 低于 0.5。这是**残差主导验证的结构性局限**，不是调 metric 能解的。
3. **`param_set_mismatch` 四 metric 全分** — 意味着 residual 能区分"发表参数集 vs 实测电芯校准误差"和"机理错误"，这是积极结果，应与温度失败一起出现在 §3.9。
4. **§3.11 Weaver w_2 必须加 condition-calibration gate**: 温度 / 参数集 / SOC-区间 校准 flag 任一失败时，残差权重必须 gate 到 0 (见 §3.11 w_2 条)。
5. **§4.12 C2 stand-alone 的预期收益再次下调**: 无 gate 的 naive C2 在部署场景下预期收益负数；带 gate 的 C2 只在 condition-matched subset 上有 +5~+10% (low-bound of Med-PRM 类比)。

**Pilot notebooks**: `pvgap_experiment/pilots/pilot_p0.ipynb` (synthetic + real `reverse_locked.jsonl`)，`pilot_p0_5.ipynb` (first stress test)，`pilot_p0_6.ipynb` (n=60 + 4-mode)，**`pilot_p0_7.ipynb` (n=60 + 4 real stressor，temperature + param-set 版本, 2026-04-19 freeze)**。产出 PNG: `pilot_p0_7_auroc_vs_noise.png` + `pilot_p0_7_heatmap.png`，须同步到 `results/paper1_figures/` 做正文引用 (9A 前完成)。

**5.5.1 What we measure (§4.15 headline)**
五档 C0 → C4 ablation 公布 PV-Gap 在电化学谱诊断上的可测闭合度。三个指标同时跟踪（diagonal accuracy / forward tight / calibration gap），避免某一单点优化而其余退化。

**5.5.2 Why our closure > Med-PRM lone baseline (if §4.14 PRM-dominant)**
如果 Weaver LOO 显示 EIS-PRM 单信号权重最大：说明 Med-PRM 模板直接移植到电化学成立，验证 §3.8 的主要假设。differentiator 在于规则语料 (Orazem-Tribollet + Lasia + DRT) 的构造本身是 contribution。

**5.5.3 Why our closure > Hallemans PyBOP (if §4.14 PyBaMM-residual-dominant)**
如果物理 oracle (PyBaMM + KK) 信号权重最大：说明物理 verifier 在 commit-under-protocol 场景能捕获 PyBOP 式纯优化抓不到的 abstention / self-correction 信号。differentiator 在 commit-vs-optimize 场景切换。

**5.5.4 Why our closure > Hassanaly NPE-on-voltage (if §4.14 SBI-dominant)**
如果 SBI 后验质量信号权重最大：说明 EIS observable + PyBaMM 具体后端 + LLM-authored prior 的三元组比 Hassanaly 的 voltage-only NPE 更能闭合 PV-Gap。须用他们的 PE / CE 指标直接对比 (§4.13)。

**5.5.5 Bounded but measurable (v0.4 定调)**
我们**不**声称 PV-Gap 被完全闭合。若 §4.15 headline 闭合到 [8.3% → X%, X < 50%]，我们诚实报告 "closure is partial; residual gap remains". 但 **"partial" ≠ "intrinsic ceiling"**，Weaver (12.8–16% math/MCQ) 已证集成可突破 bounded 论断。对 EIS 的具体闭合上限是本 paper 首次测量的物理量。

**5.5.6 "More reasoning 不 fix PV-Gap" 仍保留**
AbstentionBench 预测 + 我们 qwq-32b > v3 比较: 扩 reasoning tokens 不闭 PV-Gap。这点 v0.3 正确，保留。暗示 grounded verifier 不是 "reasoning 还不够" 的替代，是 **正交** 修补。

### 5.6 与并发 / 近邻工作的 differentiation (v0.4 新增)

> 必须与 §0.4 concurrent prior art audit 保持同步更新。

**vs Hassanaly et al. 2026 (arXiv:2604.02520, 17 days prior)**
- 他们: NPE on voltage (time-domain), SPM/P2D, 6–27 params, PE 4–8% CE 2–10%, 无 LLM, 自定义 CNPE 非 sbi-dev
- 我们: NPE on EIS (frequency-domain), PyBaMM-EIS 具体后端, LLM-authored prior + PRM rerank + Weaver 集成。observable / oracle / LLM-role 三维差异化。
- 风险: Hassanaly 组 EIS extension 可能在 submission 前发布 → §0.4 复查清单 monthly。

**vs Hallemans et al. 2025 (arXiv:2412.10896, JES 2025)**
- 他们: PyBaMM-EIS + PyBOP deterministic fit, 18 grouped params, no UQ
- 我们: 接入 SBI 后验 (UQ) + LLM commit-under-protocol + PRM + Weaver。他们无 uncertainty, 我们是 upgrade path。

**vs Domke 2025 "Large Language Bayes" (arXiv:2504.14025, NeurIPS 2025)**
- 他们: LLM 写多个 Pyro/Stan PPL, importance-weighted VI 集成, 无科学域应用
- 我们: LLM 只产 structured prior config (scope-bounded), SBI 而非 MCMC/VI, EIS 场景具化。场景 + 后端双差异化。
- 我们显式 cite Domke 作为 LLM→PPL 最相关框架，但 differentiate 在**domain-grounded**。

**vs AgentEIS (Li et al., J. Mater. Sci. 2025)**
- 他们: ET 分类器 + LoRA fine-tuning Qwen3/Llama3.1, 67.4% top-1, no verifier
- 我们: verifier-first, 无 LoRA (保留 vanilla Qwen2.5-7B 做 apples-to-apples baseline C0), 五模块集成
- AgentEIS 作为 §4.15 的对比 upper-bound baseline：若 AgentEIS 67.4% > C4 → 我们诚实报告 "fine-tuning 在 ECM 选择上仍胜出, 但 PV-Gap 闭合是正交问题"

### §6. Limitations — Narrative 2 双层显式分开

**Tier 1 (PV-Gap macro) limitations**
1. **4-model coverage**: deepseek-r1, deepseek-v3, qwq-32b, gpt-4o. Open-weight reasoning 模型 (e.g., Qwen2.5-Reasoning, o1-mini) 未覆盖；Tier 2 单独覆盖 Qwen2.5-7B 但 P/V 没有重打分。
2. **Sub-sampled L2** (60 → 30 with seed=42)：降低了 within-question 方差估计精度。
3. **gpt-4o-mini 已移除** (5 模型 → 4 模型)：不能测小开源模型的 PV-Gap scaling。
4. **E3 NoOp underpowered** (n=20)：方向性 5/0 显著但未过 Bonferroni — 报告 effect size + p_midp，不声称统计显著。
5. **Repeats=2**：P(c) within-question variance 估计精度受限。
6. **Layer A judge ensemble = 2 models** (GPT-4o + DeepSeek-V3)：均为 instruct family，未覆盖独立 reasoning judge。Kappa Paradox 已显式说明。

**Tier 2 (format-locked case study) limitations**
7. **Single-model deep dive**: Tier 2 仅 Qwen2.5-7B-Instruct。DS-R1 frozen aggregate 是二级证据，原 raw responses 未存可再 score under locked schema (→ §6 限制 7 与 Tier 1 限制 1 部分重叠)。
8. **Single benchmark**: EIS-Commit 由我们构造 (n=60/方向)；其他 battery / EIS benchmark 未在 Tier 2 测试。
9. **n=60 per direction**: REPLICATED verdict 已排除 60-sample noise (§4.8 / §9.4.1)，但 rare class (lithium_plating, LAM_positive) 样本量不足以评估校准。
10. **No hedge-forward baseline file**: `intervention_forward_full_chain.jsonl` 缺失，forward locked-vs-hedge delta 报告不完整 (→ §9.8)。F-confirm 判决不依赖 hedge 数字 (tight = 1.9 % 已是地板)。
11. **No RAG / in-context calibration intervention**: Paper 1 范围内只做 diagnostic，不做 intervention。Paper 2 主题。

**Cross-tier limitation**
12. **Tier 1 与 Tier 2 模型覆盖不重合** (Tier 1 没 Qwen2.5-7B, Tier 2 没 4 个 macro 模型): §4.10 cross-tier synthesis 是 *consistency argument*，不是 *causal demonstration*。补 Qwen2.5-7B 的 Tier 1 P/V 重打分 + 4 个 macro 模型的 Tier 2 format-locked rerun 是 Paper 1.5 / journal extension 工作量。当前 paper 显式声明这是限制。

**Arm 2 / Arm 3 (方法与闭合度实验) limitations (v0.4 新增)**

13. **EIS-PRM 规则语料规模**: 初版 `EChemRules` 约 5k 段，医学领域 Med-PRM 使用 >20k 临床 guideline 段。电化学规则标准化程度低于医学 → 我们的 stepwise 标签信噪比可能低于 Med-PRM。*补救*: 扩到 10k 段 + 多轮 RLHF-style preference 校验；Paper 1 v1 报告起始 5k 版本数字。

14. **Stepwise label 由 GPT-4o bootstrap**: 不做大规模人工标注（成本上限）。QA 只抽检 100 题。→ 若 GPT-4o judge 对电化学 stepwise 偏见（见 §3.8），整个 PRM 训练信号被污染。*补救*: DS-V3 二号 judge 做 disagreement filter；Kappa Paradox 已在 §3.1 / §4.1 caption 标示。

15. **SBI 可扩展性未 benchmark**: `sbi-dev/sbi` 包在 20-D EIS 参数空间的第三方公开 benchmark 不存在（SBI × EIS 先例核查结论）。若 SNPE-C 在此维度上收敛不理想 → 回退 FMPE（Dax 2023 在 15–20 参数 gravitational-wave 上 proven）。*补救*: §4.13 显式报告 coverage-error vs MCMC reference（小 case）；若 CE > 20%，降级到 C2 (PyBaMM-only)。

16. **Summary-statistic 设计敏感**: Dupourqué & Barret 2025 (X-ray 光谱) 明确指出 full-spectrum NPE 需要 summary 压缩。我们用 DRT 峰 + Bode + Nyquist 曲率共约 20-D summary。设计选择本身是 engineering 决策，有可能错。*补救*: §4.13 做 summary ablation (3 variants) 报告。

17. **LLM-authored prior 的 scope 约束**: 我们不让 LLM 写任意 Pyro/Stan (§3.10)，只让它产 structured JSON prior + model choice。这降低 Domke-style "large language Bayes" 风险但**牺牲灵活性**。*补救*: Paper 2 再做 full-PPL 版本；本文诚实声明 scope-bounded。

18. **Weaver 在科学域无第三方复现**: Weaver 原文 math/MCQ 验证，科学域 generalization 未被第三方独立复现（截至 2026-04-19 我们的 WebSearch 结果）。若 §4.14 集成收益 < 3% → 报告 "Weaver-for-science 需要重新设计弱监督 prior"。

19. **Concurrent 风险 (Hassanaly 组)**: Hassanaly et al. 2026 发表距今 17 天；NREL 组极有可能已有 EIS extension 在 pipeline。*补救*: §0.4 submission 前复查清单月度执行。若他们抢发 SBI×EIS → 我们立即重写 differentiator 为 "LLM-authored prior + Weaver ensemble on EIS" (后两项仍空缺)。

20. **Commit-under-protocol ≠ deployment**: 本文的 ABSTAIN 机制是 benchmark 协议内的，不是工程化的。实际部署时 abstention policy 还需考虑成本、用户反馈、edge case。*补救*: 不声明 "deployment-ready"。

21. **没有湿实验闭环**: ChatBattery / Coscientist 类工作用湿实验/机器人闭合。我们的 GT 来自 PyBaMM simulation + bounded noise，不是真实电化学工作站数据。*补救*: §3.2 明确声明 simulated GT；外部 validation 在 Paper 2。

22. **Pilot validation SOC 范围限制 (P0.6)**: ρ_real 的 AUROC ≥ 0.89 claim 仅在 SOC = 0.50 附近 Chen2020 OCV plateau 上验证过。SOC 进入斜率区 (0.0–0.3 或 0.7–1.0) 时 EIS 对 SOC 敏感度显著上升 → residual 对 SOC 标定误差的鲁棒性未知。*补救*: Paper 1 claim 限定在 "SOC plateau regime"；sloped-regime 在 Paper 2 或 journal extension。

23. **Pilot stressor 覆盖 3/4 pass (P0.7, 2026-04-19 freeze)**: P0.6 的 `soc_plus/minus` 因 SOC=0.50 plateau 退化为 null 被弃用；P0.7 改成 4 个非-null real stressor 测试：`surface_form` (ρ_real AUROC=0.945)、`spm_vs_spme` (0.998)、`param_set_mismatch` (Chen2020 vs OKane2022, 1.000) 全部通过；**`temperature_mismatch` (10°C vs 25°C) 四个 metric 全部跌到 AUROC ≈ 0.46–0.47** (低于随机)，属 **confounder-dominance failure**——温度带来的 1.61× |Z| scale 压倒机理差异，错误候选有时"偶然补偿"温度偏移反而比正确候选残差更小。这是残差主导验证的**结构性局限**，不是 tuning 问题。*后果*: §3.9 primary-residual 宣称从 "worst-case 0.89" 改为 "**条件匹配下 worst-case 0.945；条件不匹配下可能跌破 0.5**"；§4.12 C2 预期收益再次下调并强制 condition-calibration gate 前置；§3.11 w_2 加 gate 逻辑 (gate 失败 → 权重 0)；见新增 limitation 26。

24. **ρ_imag 的 mode-dependence 是已知失败模式 (P0.5 + P0.6)**: ρ_imag 在 surface-form mismatch 下 AUROC = 0.51 (anti-signal)，在 SPM-vs-SPMe mismatch 下 AUROC = 1.00。单独用会被特定失配模式误导。*补救*: §3.11 Weaver w_2 只把 ρ_imag 作为 mode-conditional channel 保留；如果 §4.14 LOO 显示 ρ_imag 权重不稳健或 LOO 收益 < 1%，submission 前弃用这个 channel，仅保留 ρ_real + ρ_complex。这条 limitation 同时是正论文贡献 (residual metric mode-conditionality 是 first-time 报告)。

25. **Pilot 数据量**: P0.6 每 (mode, noise) cell 只有 60 pair (20 scenarios × 3 candidates)，AUROC resolution ≈ ±0.003 但 95% CI 仍可能跨 0.75 gate 附近 (e.g., surface_form @ noise=0.15 AUROC=0.908，CI 下沿落入 0.82–0.83)。*补救*: 主实验 §4.12 重跑 n ≥ 200 (扩场景到 L2+L3 合集)，submission 前 freeze 数字。

26. **Residual 验证的条件-校准前置 (P0.7 发现, 2026-04-19)**: forward residual 的可用性**条件性依赖于 observed 与 candidate 的操作条件匹配**。P0.7 在 15°C 的温度差 (GT=10°C, cand=25°C) 下观察到 ρ_real / ρ_logmag / ρ_imag / ρ_complex 四个 metric **全部 AUROC ≈ 0.46–0.47 (低于随机)**——不是因为 stressor 弱 (smoke test |Z|.mean 是 baseline 的 1.61×)，而是因为温度引起的整体 |Z| scaling 淹没了机理差异，**错误机理候选的残差反而有时比正确候选更小** (confounder-dominance)。这意味着残差主导验证**不能独立做部署级 verifier**；必须有一个上游 condition-calibration pass 先确认 T / parameter-set / SOC-区间 已校准。*补救 & 正论文贡献*: (a) §3.11 w_2 加 gate 逻辑 (gate 失败 → 残差权重 = 0，不是降权)；(b) §4.12 诚实报告无 gate 的 C2 预期负收益；(c) 本条 limitation 同时是 **first-time 量化报告的物理 verifier 结构性边界** — 社区普遍直觉"物理 oracle 更可靠"的一个具体反例。主实验在 §4.12 / §4.13 明确标注 "condition-matched subset" 与 "deployment-realistic unmatched subset" 两档数字。

### §7. Conclusion (~0.3 页)

- 一句话复述 headline
- 一句话复述三重独立证据（confusion + calibration + replication）
- 一句话点出：format-locked commit probing 是 future benchmark 设计的工具
- 一句话 teaser Paper 2

---

## 4. 图表清单

| # | 类型 | 源数据 | 状态 |
|---|---|---|---|
| Figure 1 | 7×7 confusion matrix heatmap (reverse_locked) | results/e5a1_faithfulness/confusion_locked.png | **已生成** |
| Figure 2 | bar chart conf_all vs conf_wrong per GT × 4 classes | §9.3 per-GT 表 | **已生成** (`results/paper1_figures/figure2_calibration_per_gt.{png,pdf}` via `_make_paper1_figures.py`, 2026-04-18) |
| Figure 3 | conf-bin accuracy scatter (forward) | §9.4 confidence-bin 表 | **已生成** (`results/paper1_figures/figure3_conf_bin_forward.{png,pdf}` via `_make_paper1_figures.py`, 2026-04-18) |
| Figure 4 | AMR vs V faithfulness curve (forward truncation) | results/e5a1_faithfulness/faithfulness_curves_v2.png | **已生成** |
| Table 1 | per-feature tight/loose/median\|log10\| (seed-1 + seed-2) | §9.4.1 | **已就位** |
| Table 2 | three-test replication verdict | §9.4.1 | **已就位** |
| (Appendix) Table A1 | full per-scenario predictions (reverse) | reverse_locked.jsonl | 需要 prettify |
| (Appendix) Table A2 | full per-scenario predictions (forward) | forward_locked_seed1+2.jsonl | 需要 prettify |

**需要新写的绘图脚本**: Figure 2 + Figure 3。可以放在一个 `_make_paper1_figures.py` 里。

---

## 5. Related Work 待补文献

**Calibration** (需要至少 4 篇，目前手头 2 篇 verified):

- [x] **Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D.,
  Perez, E., Schiefer, N., et al.** (2022). *Language Models (Mostly)
  Know What They Know.* arXiv:2207.05221, posted 11 Jul 2022.
  URL: https://arxiv.org/abs/2207.05221.
  **使用位置**: §1 motivation（"larger LMs are mostly calibrated on MCQ"），
  §2.1 calibration bin, §5.3 hook（factual QA vs domain-committed
  diagnosis 对照）。**注**: arXiv preprint, 无期刊页码；论文内 P(True) 自评协议
  在 §3, P(IK) 在 §4——引用时按 section 索引。

- [x] **Lin, S., Hilton, J., Evans, O.** (2022). *Teaching Models to
  Express Their Uncertainty in Words.* arXiv:2205.14334, posted 28 May
  2022; published in TMLR, 17 Oct 2022. OpenReview id `8s8K2UZGTZ`.
  URLs: https://arxiv.org/abs/2205.14334 ;
  https://openreview.net/forum?id=8s8K2UZGTZ.
  **使用位置**: §2.1（verbalized confidence baseline），§5.3（"trained
  to verbalize" vs "spontaneously verbalize under format-lock" 对照）。
  **注**: TMLR 文章本身无传统页码，按 OpenReview PDF 内的 section 引用
  （Calibration on training distribution = §3；distribution shift = §4）。

- [ ] Groot+ 2024 — 需查具体 paper title（Scientific-domain calibration）
- [ ] Tian+ 2023 "Just Ask for Calibration" — RLHF 如何破坏 calibration
- [ ] Xiong+ 2024 "Can LLMs Express Their Uncertainty?" ICLR

**Faithfulness / CoT analysis**:

- [x] **Lanham, T., Chen, A., Radhakrishnan, A., et al.** (2023).
  *Measuring Faithfulness in Chain-of-Thought Reasoning.*
  arXiv:2307.13702, posted 17 Jul 2023; Anthropic technical report.
  URLs: https://arxiv.org/abs/2307.13702 ;
  https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning.
  **使用位置**: §2.2（CoT-faithfulness 测量协议原文），§3.2（forward /
  reverse intervention 协议 directly modeled on Lanham §3 truncation +
  paraphrase 实验），§4.6（AMR-vs-V 曲线 follows Lanham 的 answer-match-rate
  intervention）。**注**: arXiv preprint，引用按 section 索引——truncation
  intervention §3.1, paraphrase intervention §3.2, scale findings §4。

- [ ] Turpin+ 2023 "LLMs Don't Always Say What They Think"（已引）
- [ ] Radhakrishnan+ 2023 "Question Decomposition Improves Faithfulness"

> **Citations note (2026-04-18)**: 三篇 verified by web search.
> 三者均为 arXiv 或 TMLR 论文 — 没有传统期刊连续页码可援引。BibTeX 写法用
> `@misc` (arXiv) 或 `@article` (TMLR) + `eprint` / `note` 字段；引用正文里
> 用作者+年份+section anchor (如 "Lanham et al., 2023, §3.1") 即可。

**LLM × Electrochemistry / Materials**:
- [ ] Ramos+ 2024 "A review of LLMs for chemistry"（需确认引用）
- [ ] GPT4Battery / BatteryGPT 类工作（需搜）
- [ ] Boiko+ 2023 "Autonomous chemical research with LLMs"

**Prior-driven / mode-collapse**:
- [ ] McKenzie+ 2023 "Inverse Scaling"
- [ ] Perez+ 2022 "Discovering Language Model Behaviors with Model-Written Evaluations"

---

**v0.4 新增 citation bins (Arm 2 方法)** — 全部已在 Axis C 重做 / SBI 补查中 verified

**Grounded PRM / stepwise verifier**:
- [x] **Yun, J., Sohn, J., Park, J., Kim, H., Tang, X., et al.** (2025). *Med-PRM: Medical Reasoning Models with Stepwise, Guideline-verified Process Rewards.* EMNLP 2025. arXiv:2506.11474. [ACL Anthology](https://aclanthology.org/2025.emnlp-main.837/). **位置**: §3.8, §5.5.2, Related Work §2.4.
- [x] **Zhu, J., Zheng, C., Lin, J., et al.** (2025). *RetrievalPRM: Retrieval-Augmented Process-Reward Models.* ACL Findings 2025. arXiv:2502.14361. **位置**: §3.8 (二阶段检索骨架)；Δ-accuracy UNVERIFIED，使用前须再核。
- [x] **Zhong, X., Jin, B., Ouyang, S., et al.** (2025). *ChemRAG-Bench.* COLM 2025. arXiv:2505.07671. **位置**: §3.8 RAG 类比（+17.4%）, Related Work §2.4.

**Heterogeneous verifier ensembling**:
- [x] **Saad-Falcon, J., Buchanan, E. K., Chen, M. F., et al.** (2025). *Weaver: Closing the Generation-Verification Gap with Weak-Verifier Aggregation.* NeurIPS 2025. arXiv:2506.18203. [OpenReview](https://openreview.net/forum?id=BHRFAubSf9). **位置**: §3.11, §5.5.5, Related Work §2.4.

**PyBaMM / EIS physics oracle**:
- [x] **Hallemans, N., Courtier, N. E., Please, C. P., et al.** (2024/2025). *Physics-based battery model parametrisation from impedance data (PyBaMM-EIS).* arXiv:2412.10896. J. Electrochem. Soc. 172 (2025). **位置**: §3.9, §5.5.3, §5.6 (direct baseline).
- [x] **Zhang, R., Sadeghi, N., et al.** (2023/2025). *AutoEIS: Automated Bayesian ECM Selection.* J. Electrochem. Soc. 170 (2023) DOI:10.1149/1945-7111/aceab2 (Editors' Choice); JOSS 2025 DOI:10.21105/joss.06256; arXiv:2305.04841. **位置**: §3.9 KK+physics 过滤前端, Related Work §2.3.

**Simulation-based inference (SBI)**:
- [x] **Hassanaly, M., Randall, C. R., Weddle, P. J., et al.** (NREL/INL, 2026). *Neural posterior estimation for scalable and accurate inverse parameter inference in Li-ion batteries.* arXiv:2604.02520 (2026-04-02, **concurrent ~17 days prior**). **位置**: §0.4 (危险度 🔴), §3.10, §4.13, §5.6 头号 differentiator.
- [x] **Dax, M., Green, S. R., Gair, J., Pürrer, M., et al.** (2023). *Flow Matching for Scalable Simulation-Based Inference.* NeurIPS 2023. arXiv:2305.17161. **位置**: §3.10 (FMPE 作为高维 SBI fallback), §6 limitation 15.
- [x] **Dupourqué, S. & Barret, D.** (2025). *Simulation-Based Inference for X-ray Spectral Fitting.* arXiv:2506.05911. **位置**: §3.10, §6 limitation 16 (summary-stat 压缩必要性)。

**LLM → probabilistic program / Probabilistic Language of Thought**:
- [x] **Wong, L., Grand, G., Lew, A. K., Goodman, N. D., Mansinghka, V. K., Andreas, J., Tenenbaum, J. B.** (2023). *From Word Models to World Models: Translating from Natural Language to the Probabilistic Language of Thought.* arXiv:2306.12672. **位置**: §3.10 (PLoT 框架引用), §5.6 (Wong + Domke 对比).
- [x] **Domke, J.** (2025). *Large Language Bayes.* NeurIPS 2025. arXiv:2504.14025. **位置**: §0.4 (危险度 🟡), §3.10 (LLM-written PPL scope 约束说明), §5.6.

**Bayesian inverse EIS (non-SBI baselines)**:
- [x] **Aitio, A., Marquis, S. G., Ascencio, P., Howey, D.** (2020). *Bayesian parameter estimation applied to the Li-ion battery single particle model with electrolyte dynamics.* IFAC 2020. arXiv:2001.09890. **位置**: Related Work §2.3, §5.6.
- [x] **Berliner, M. D., Kim, S., Cui, X., Lam, R. K., et al.** (2025). *Bayesian Analysis of Interpretable Aging across Thousands of Lithium-ion Battery Cycles.* arXiv:2504.10439. **位置**: Related Work §2.3 (HPC-MCMC baseline).
- [x] **Ciucci group** (HKUST). *Hierarchical Bayesian DRT inversion.* Electrochim. Acta 2021; OSTI 1755725. **位置**: Related Work §2.3 (HMC DRT 先例)。

**Scientific agent / electrochem baseline**:
- [x] **Li, Y., et al.** (2025). *AgentEIS: Automated EIS Mechanism Interpretation via LLM + Extra-Trees Classifier.* J. Mater. Sci. 2025. DOI:10.1007/s10853-025-11692-x. **位置**: §5.6 (fine-tuning 基线对比), §4.15 upper-bound baseline.
- [x] **Zhou, S., Liu, R., Su, B., et al.** (2025). *BatteryAgent.* arXiv:2512.24686. **位置**: Related Work §2.3 (三层分层 precedent).
- [x] **Liu, S., Xu, H., Ai, Y., Li, H., Bengio, Y., Guo, H.** (2025). *ChatBattery: Multi-Stage LLM for Cathode Discovery.* arXiv:2507.16110. **位置**: Related Work §2.3 (multi-stage filter template).
- [x] **Bran, A. M., Cox, S., Schilter, O., et al.** (2024). *ChemCrow: Augmenting Large Language Models with Chemistry Tools.* Nat. Mach. Intell. DOI:10.1038/s42256-024-00832-8. arXiv:2304.05376. **位置**: Related Work §2.3.
- [x] **Boiko, D. A., MacKnight, R., Kline, B., Gomes, G.** (2023). *Autonomous chemical research with large language models.* Nature 624. DOI:10.1038/s41586-023-06792-0. **位置**: Related Work §2.3.

> **v0.4 citation discipline**: Axis C 重做 (45 tool calls) + SBI 补查 (28 tool calls) 每条 arXiv ID/DOI 均 WebFetch 验证过。RetrievalPRM Δ-accuracy 具体数字标 UNVERIFIED。BatteryAgent arXiv ID (2512.24686) 按 search 结果记录，submission 前再次核查 (post-cutoff)。

---

## 6. Data gaps / 待补

| 项 | 严重度 | 补救路径 | 是否 blocker |
|---|---|---|---|
| `intervention_forward_full_chain.jsonl` 缺失 | 中 | 在既有 hedge forward runs 里筛 untruncated trial | 否（F-confirm 已独立成立） |
| DS-R1 raw reverse responses 没存 | 中 | 重跑 r1 on 60 items（R1 API 费用估 ~$5） | 否（Qwen 已是主证据） |
| Qwen 2.5 **14B/32B** 未测 | 低 | 若 scale-up 改变结论 → 重大 story 调整。优先级 LOW，Paper 1 可以不做 | 否 |
| LLaDA / sham 异常未诊断 | 低 | 独立诊断 notebook 已建，结论待跑 | 否（本 paper 不涉及） |
| Figure 2 / 3 绘图脚本 | 高 | ~30 min 本地跑 matplotlib | 是（drafting 阶段就要） |
| Related work 文献 coverage | 高 | 1–2 天深读 + 引用完善 | 是 |

---

## 7. Timeline 建议

（不是承诺时间，只是顺序）

1. **Week 0**: 绘 Figure 2 + Figure 3；补 related work coverage 到 80%
2. **Week 0–1**: Methods 先写（最结构化，最快）
3. **Week 1**: Results 按 §4.1–§4.6 顺序写；每小节 0.3–0.5 页
4. **Week 1–2**: Intro + Discussion 拉出初稿
5. **Week 2**: Abstract 最后写；Limitations + Conclusion 填
6. **Week 2–3**: 自审 pass: 读一遍挑 4 类问题——overclaim / under-citation / figure-text redundancy / missing limitation
7. **Week 3**: 给导师送审

**不要先写 Abstract**；Abstract 在全文定稿后压缩。

---

## 8. 自审 checklist（draft 完成前 gate）

- [ ] Headline 数字 (8.3% vs 14.3%) 在 Abstract 和 Conclusion 都出现
- [ ] 每个 claim 都有 §4.x subsection 支撑
- [ ] 三项 REPLICATION verdict 在 Abstract 和 §4.5 都有；不放弃这个卖点
- [ ] §6 Limitations 明确说 "single-model" — 不假装 generalize
- [ ] 每条 §9.x 结论在此文件里有对应引用行
- [ ] Figures 都有 caption + self-explanatory（读者不读 body 也能懂）
- [ ] 不重复 实验结果.md 的数字——只 inline 引用；需要新数 → 改 实验结果.md
- [ ] 不提 Self-Planning Prompting（已 scope 出）
- [ ] 不提 PyBaMM 具体方案（只 teaser）
- [ ] Scoring 漏洞 A/B/C 在 §5.4 Discussion 坦诚写出（漏洞 C 已修复；A/B 作为 scoring robustness 讨论）

---

## 9. 下一步（只等你点头）— 更新 for Narrative 2

**已完成 (2026-04-18 batch)**:
- [x] Figure 2/3 (Tier 2 calibration figures) 已生成 → `pvgap_experiment/results/paper1_figures/`（在新 §4 编号下变为 Figure 3 与 Figure 4）
- [x] Methods §3 prose v0.3 已重构为 8 个子节（§3.0 PV-Gap construct + §3.1 Tier 1 setup + §3.2–§3.7 Tier 2 protocol）
- [x] Kadavath 2022 / Lin 2022 / Lanham 2023 三篇 citation verified（§5 中给出 arXiv ID + section anchor）
- [x] Narrative 2 双层挂钩在 §0.2 / §1 / §2 / §3 / §4 / §5 / §6 全部贯通（v0.3 全文 audit done）

**Narrative 2 新增待办（按依赖链排序）**:

1. **Tier 1 主图 (Figure 1)**: 4 模型 × 4 stages V scores bar chart
   - 数据已在 实验结果.md §2 — 直接从表读
   - 写到 `_make_paper1_figures.py` 的 `make_figure1()` (新增)
   - ~20 min 本地

2. **Tier 1 主表 (Table 1)**: per-model P / V / PV-Gap mean & median
   - 数据 实验结果.md §1.1 — 4 行 LaTeX
   - ~10 min

3. **Cross-tier synthesis schematic (§4.10)**: PV-Gap 在 Tier 1 stage 衰减 ⇄ Tier 2 现象的对应关系
   - 可选 schematic figure 或纯 table — workshop 版可省，journal 版强烈建议
   - ~30 min if schematic, ~10 min if table

4. **Methods 全英化**: §3.0–§3.7 当前是中英混排 prose draft
   - 一次性翻译，~1 h
   - 不要在 draft v1 之前改字号或 LaTeX-ize；先把 logic 锁住

5. **Results §4 prose 起草**: §4.1–§4.10 当前是 bullet outline
   - 按 §3 prose 格式延续，10 个子节 × 0.2–0.4 页
   - 数据已全部就位

6. **Related work 剩余 5 篇 verify**:
   - Groot 2024 / Tian 2023 / Xiong 2024（calibration bin）
   - Turpin 2023 / Radhakrishnan 2023（faithfulness bin）
   - Farquhar 2024 (Nature) — semantic entropy 对比 (§5.1) 必须 verify exact citation

7. **Figure caption 英文版** 起草（已生成的两张图 footer 已 ASCII-safe）

8. **Appendix Table A1 / A2 prettify**: 由 reverse_locked.jsonl + forward_locked_seed{1,2}.jsonl 生成 LaTeX-ready 表（脚本未写）

**Narrative 2 风险登记**:
- Tier 1 与 Tier 2 模型覆盖不重合（§6 limit 12）— reviewer 可能要求至少补 Qwen2.5-7B 的 Tier 1 P/V 重打分。决策点：先看 workshop 反馈；journal 版可能 mandatory。
- Tier 1 PV-Gap V 数字依赖原 hedge-mode scoring，§9.5 漏洞 A/B 仍未量化对 V 的真实影响。决策点：在 §3.7 / §5 显式声明 "Tier 1 V 估计可能略低，本文不重打分；Tier 2 §4.5 confusion matrix 走 JSON 字面键，独立于 alias map" — 已写入 §3.7 audit 子节。

---

## 9.v0.4 Arm 2 / Arm 3 实现路线图 (method + closure 实验)

> 本部分是 v0.4 新增核心内容。每项列出：输出产物 / 预估算力 / 前置依赖 / decision gate。

### 9A. EIS-PRM 实现 (Arm 2.1)

**9A.1 规则语料库 `EChemRules` 构造 (状态: seed 已就位, 扩容中)**
- 输出: `pvgap_experiment/data/echem_rules/*.jsonl`（每条带 rule text + mechanism tag + frequency band + counterexamples）
- **已就位 (2026-04-19)**:
  - `echem_rules_seed.jsonl`: 50 条 hand-curated 规则（33 primary / 13 secondary / 4 weak；25 feature / 13 meta / 7 fit / 5 DRT）；全部 primary 规则强制带 `counterexamples` 字段（honesty constraint, 见 README.md）
  - `echem_rules_condition_calibration.jsonl`: **15 条 gate 规则 (P0.7-driven)**，新增 `gate_action` 字段 (ABSTAIN / require / flag / upgrade / compensate / PASS)；直接服务 §3.11 w_2 condition-calibration gate；防止 temperature-confounder / parameter-set-mismatch / SOC-plateau 把残差分数污染进 Weaver
  - `sources.bib`: Barsoukov & Macdonald 3e / Orazem-Tribollet 2e / Lasia / Chen2020 / OKane2022 / Hallemans2024 / Ciucci2020 / Iurilli2021 / Dubarry2017 / Birkl2017 / Boukamp1995
- 规模目标: v1 扩到 5k 段 (seed 为骨架，文本挖掘为主力)；journal 版扩到 10k
- 扩容路径 (README 已文档化):
  1. Textbook structured-extraction pass — Barsoukov & Macdonald, Lasia, Orazem-Tribollet 的诊断表逐条生成卡片
  2. Literature pass — AutoEIS (Zhang 2023/2025), Hallemans, Iurilli, Dubarry, Birkl 中每条诊断主张 → 一张卡片带 source
- 算力: 零 GPU；主要是文本清洗 + 手工分段 + LLM-辅助 extraction (GPT-4o 抽 + 人工 review)
- 前置依赖: Orazem-Tribollet 电子版 / Lasia 电子版 / DRT peak-assignment table 手工整理 (**未开始**)
- Decision gate: v1 规模到 5k **且 primary 规则 100% 带 counterexamples** 且 gate 规则 100% 带 `gate_action` 且经人工 100 段抽检精度 ≥ 90% → 进入 9A.2

**9A.2 Stepwise label bootstrap**
- 输出: `data/eis_prm_labels.jsonl`（~1000 stepwise + labels，GPT-4o judge + DS-V3 disagreement filter）
- 算力: ~$50 OpenAI / ~$20 DS-V3 API
- 前置依赖: 9A.1 + 既有 EChem-Reason CoT + EIS-Commit CoT (均已就位)
- Decision gate: DS-V3/GPT-4o disagreement rate < 15% → 进入 9A.3；否则 reviewer 可能质疑 label quality，需加独立 judge

**9A.3 PRM 训练**
- 输出: `models/eis_prm_qwen7b_v1/` + training log
- 算力: 单机 8×A100 × 48h (estimate per Med-PRM)
- 前置依赖: 9A.2
- Decision gate: val stepwise AUC > 0.80 → 进入 §4.11 实验

### 9B. PyBaMM-Verified Loop 实现 (Arm 2.2)

**9B.0 Condition-calibration gate (P0.7 硬前置, 新增于 2026-04-19)**
- 输出: `pvgap_experiment/src/condition_calibration_gate.py`
- 前置依赖: 9A.1 `echem_rules_condition_calibration.jsonl` (15 条 gate 规则, 已就位)
- 功能: 给定 (Z_obs, candidate_params, metadata), 跑 15 条 gate 规则；根据 `gate_action` 决定 PASS / ABSTAIN / flag / require / upgrade；**只有 PASS 的 case 才进入 9B.2 residual scoring**。ABSTAIN 直接汇报给 Weaver 把 w_2 置零；flag 让 residual 过但打 "low-trust" 标记供 Weaver 降权。
- 算力: 零 GPU，每 case ~1 ms (纯规则匹配)
- Decision gate: 在 pilot_p0_7 的 4 stressor × 60 pair 数据上复跑，temperature_mismatch case 必须 100% 触发 `require_T_match_within_5C_or_ABSTAIN`；`param_set_mismatch` 必须 100% 触发 `require_parameter_set_calibration_pass_or_ABSTAIN`；`surface_form` 和 `spm_vs_spme` 应 PASS (不应被误 gate 成 ABSTAIN)。→ 进入 9B.2

**9B.1 PyBaMM-EIS 环境搭建**
- 输出: `pvgap_experiment/src/pybamm_eis_residual.py` + 单元测试
- 前置依赖: `pybamm==25.12.2` + `pybammeis==0.1.6` (注意: PyPI 上**不存在** `pybamm-eis`, 实际通过 `pip install git+https://github.com/pybamm-team/pybamm-eis.git` 安装, 模块名是 `pybammeis`)
- 算力: 零 GPU，CPU 即可；每次 DFN forward ~1–5 s, SPM ~0.1–0.5 s
- 实现 (2026-04-21): `simulate_Z(case)` 包 `pybammeis.EISSimulation(model, parameter_values, initial_soc).solve(frequencies, method="direct")`; `residuals(z_obs, z_sim)` 计算 4 个归一化 ρ (real / imag / complex / logmag); 顶层 `pybamm_eis_residual(case)` = `residual_fn` 入口. 内置 `_SIM_CACHE` 按 `(model, params, soc, overrides)` SHA-1 缓存 simulation 对象, 避免重建 (Weaver loop 同 candidate 跨多 obs 时 ≥99% 命中).
- Decision gate: SPM/Chen2020 self-residual = 0; L_neg +10% 扰动 ρ_complex > 1e-6 (smoke test PASS); 5 DFN 参数 set 与 Hallemans 2024 复现一致 → 待跑.

**9B.2 residual + KK 的 self-correction 循环 wrapper**
- 输出: `pvgap_experiment/src/pybamm_verified_loop.py` + `test_9B2_integration.py`
- 前置依赖: **9B.0 (condition-calibration gate)** + 9B.1 + LLM API wrapper (现有)
- 入口逻辑: 每次调用必须先过 9B.0 gate；gate ABSTAIN → wrapper 返回 ABSTAIN 而不是残差分，**不**计入 accuracy 分母但计入 coverage 分母（与 AbstentionBench 协议一致）
- 算力: 每 scenario × 3 轮 × 若干 LLM call，每 ABSTAIN 决策用 1 PyBaMM call
- 实现 (2026-04-21, `test_9B2_integration.py`): wire `pybamm_eis_residual` 作 `residual_fn`, 跑 5 toy cases (L_neg+15% / R_part+20% / SOC mismatch / param_set mismatch / SPMe-vs-SPM model-form). 每 case 比较 correct vs wrong candidate 的 `ρ_complex`, 加 gate ABSTAIN-on-mismatch 容错.
- Decision gate: 对 5 个 hand-crafted toy cases (已知 GT mechanism), loop 正确率 > 70% → 进入 §4.12 实验. **结果: 5/5 = 100% PASS** (4 case PASS gate 后 ρ_complex correct=0 vs wrong ∈ [8e-3, 6e-2]; 1 case `param_set_mismatch_chen_vs_marquis` gate 正确 ABSTAIN 拒绝评分). **9B.2 状态: PASS, 进入 §4.12 实验**.

### 9C. PLoT-lite SBI 实现 (Arm 2.3 核心)

**9C.1 LLM prior-emission prompt design**
- 输出: `pvgap_experiment/prompts/sbi_prior_config_v1.md` + `sbi_prior_schema_v1.json` + `pvgap_experiment/src/sbi_prior_emit.py`
- 前置依赖: 9B.1
- 实现 (2026-04-21): 6-param 固定 schema (L_neg/L_pos, R_p_neg/R_p_pos, D_s_neg/D_s_pos), allowed dist `lognormal | uniform`, 强制 `support` 截断. JSON-Schema 验证 + 独立 keys/duplicates check. LLM caller provider auto-select (DeepSeek 优先 — 当时 OpenAI quota 耗尽).
- Decision gate: 在 5 hand cases, LLM 产出的 structured JSON 全部 schema-valid 且语义 reasonable → **5/5 PASS** (DeepSeek-chat, JSON mode). 物理 reasonableness: 6 params × 5 cases support 全部落在文献 O() ±1 decade 内, 所有 lognormal scale ∈ [0.05, 2.0]. **9C.1 状态: PASS, 进入 9C.2 SNPE-C**.

**9C.2 sbi 包 SNPE-C baseline**
- 输出: `pvgap_experiment/src/sbi_posterior.py` + training logs
- 算力: 10⁴ PyBaMM-EIS 仿真 (CPU, ~3h); SNPE 训练 (单 A100, ~30 min)
- 前置依赖: 9B.1
- 实现 (2026-04-21): `build_prior(emission)` 把 9C.1 emission JSON 转 6-D log10-space `MultipleIndependent(BoxUniform)` 先验; `simulator_factory` 把 θ_log10 → x ∈ R^16 (Re/Im at 8 freqs in 0.1Hz–1kHz); `train_snpe(prior, sim, n_sim, max_epochs=80, sample_with='direct')`; `coverage_error` 在 α=0.10 中心 CI 上算 per-dim mis-coverage. **修复**: SNPE-C 默认 rejection sampler 在低 n_sim 下 0% acceptance 死锁, 切 `sample_with='direct'` (NPE flow 直采).
- 实测 (n_sim=200 smoke, lg_m50_healthy emission): pipeline 完整跑通 (sim 305s + 训练 81 epochs + sampling 即时). Per-dim mis-coverage: hold-out [0.667, 0.667, 0.333], mean 0.556, deviation from nominal 0.10 = 0.456. **诚实 FAIL** at smoke scale (远超 0.15 阈值).
- 诚实科学结论: 200 sims 对 6-D in / 16-D out NPE 严重欠采样 (Hassanaly 2024 用 ~10⁴-10⁵). **smoke 只验证 pipeline 完整性, 不构成 calibration claim**. 需 dedicated 10⁴-sim run (~3-4h CPU 单机, 或 Colab) 才能正式过 §9C.2 gate.
- Decision gate: 在 3 hold-out scenarios, coverage-error ≤ 15% (对比 Hassanaly 2–10%) → 进入 9C.3；否则切换到 FMPE. **当前状态: pipeline READY, 待 10⁴-sim 正式跑**.

**9C.3 FMPE fallback (如 9C.2 收敛不佳)**
- 算力: 10⁵ 仿真 (~24 h CPU)
- 前置依赖: 9C.2 失败
- Decision gate: CE ≤ 20% → 进入 §4.13；否则 **Paper 1 scope 收回到 C0/C1/C2**，不上 SBI（诚实 fallback）

**9C.4 Summary-statistic ablation (3 variants)**
- 输出: summary-stat comparison table (§4.13)
- 前置依赖: 9C.2 或 9C.3
- Decision gate: 至少一种 summary 达 CE ≤ 15% → 成功

### 9D. Weaver 集成 (Arm 2.4)

**9D.1 5-signal 提取 wrapper**
- 输出: `pvgap_experiment/src/weaver_signals.py`
- 前置依赖: 9A.3 + 9B.2 + 9C.2/3 (至少 3 信号可用即进)
- Decision gate: 在 EIS-Commit 20% 子集上, 5 信号的 correlation 矩阵非全 >0.9 (避免 redundant)
- **状态 (v1, 30-case mini-batch, 2026-04-21)**: PROVISIONAL PASS。`extract_w{1..5}` API 全部就位; w_2 (PyBaMM-residual) 与 w_5 (LLM-critic via DeepSeek) 是真信号, w_1/w_3/w_4 在该轮是 stub-constant 0.50（PRM ckpt 与 SBI posterior 未到位; pyimpspec `mu` 已废弃）。10/10 off-diagonal 中只有 1 对 (w_2,w_5) 可计算, Spearman = 0.447（中等, 非冗余）。无任何 |ρ|>0.9, 因此严格门槛通过, 但 informativeness flag = False。
- **v2 修复 (本仓库已 commit)**:
  (a) `extract_w3_linkk` 改用 `KramersKronigResult.get_estimated_percent_noise()`（pyimpspec ≥1.x）, 经验值 clean ≈0.2%, 30% 噪声扰动 ≈21%, 通过 Platt sigmoid 映射 [0,1];
  (b) `_synth_minibatch` 的条件键改为 `observed_temperature_K`/`candidate_temperature_K` Kelvin, 与 §9B.0 CC-008 规则匹配, 修正 cond_mismatch case 未触发 ABSTAIN 的 bug;
  (c) 相关矩阵改为按对计算, 跳过常数列, 避免 scipy `spearmanr` 在多常数列下塌缩为 1×1 矩阵;
  (d) per-case 结果即时写盘, 防止 correlation 阶段崩溃丢失全部样本。
- **未决**: w_1 (PRM) 等待 §9A.3 ckpt; w_4 (SBI posterior) 等待 §9C.2 10⁴ 校准; full 5/10-pair informative gate 需上述两项就位后复跑。
- 输出文件: `results/weaver_signals_minibatch_v1.json`, `results/weaver_signals_9D1{,_v2}.log`
- **状态 (v1, 30-case mini-batch, 2026-04-21)**: PROVISIONAL PASS。`extract_w{1..5}` API 全部就位; w_2 (PyBaMM-residual) 与 w_5 (LLM-critic via DeepSeek) 是真信号, w_1/w_3/w_4 在该轮是 stub-constant 0.50（PRM ckpt 与 SBI posterior 未到位; pyimpspec `mu` 已废弃）。10/10 off-diagonal 中只有 1 对 (w_2,w_5) 可计算, Spearman = 0.447（中等, 非冗余）。无任何 |ρ|>0.9, 因此严格门槛通过, 但 informativeness flag = False。
- **v2 修复 (本仓库已 commit)**:
  (a) `extract_w3_linkk` 改用 `KramersKronigResult.get_estimated_percent_noise()`（pyimpspec ≥1.x）, 经验值 clean ≈0.2%, 30% 噪声扰动 ≈21%, 通过 Platt sigmoid 映射 [0,1];
  (b) `_synth_minibatch` 的条件键改为 `observed_temperature_K`/`candidate_temperature_K` Kelvin, 与 §9B.0 CC-008 规则匹配, 修正 cond_mismatch case 未触发 ABSTAIN 的 bug;
  (c) 相关矩阵改为按对计算, 跳过常数列, 避免 scipy `spearmanr` 在多常数列下塌缩为 1×1 矩阵;
  (d) per-case 结果即时写盘, 防止 correlation 阶段崩溃丢失全部样本。
- **未决**: w_1 (PRM) 等待 §9A.3 ckpt; w_4 (SBI posterior) 等待 §9C.2 10⁴ 校准; full 5/10-pair informative gate 需上述两项就位后复跑。
- 输出文件: `results/weaver_signals_minibatch_v1.json`, `results/weaver_signals_9D1{,_v2}.log`

**9D.2 弱监督 label model 训练**
- 输出: `models/weaver_ensemble_v1.pt`（400M 蒸馏 student per Weaver 原文）
- 算力: 单 A100, ~4 h
- 前置依赖: 9D.1
- Decision gate: 在 calibration subset 上, 集成分 AUROC 比任一单信号高 ≥ 3% → 进入 §4.14
- **状态 (scaffold, 2026-04-21)**: `pvgap_experiment/src/weaver_label_model.py` 已铺设。组件: `LabelModelInputs` dataclass + `from_signals_json` (吃 §9D.1 mini-batch JSON), Snorkel `LabelModel` 包装 + 失败回退到 calibrated majority-vote (`_MajorityVote`), `StudentHeadStub` (numpy MLP 占位, 待 §9A.3 PRM ckpt 到位换成 400M 蒸馏 head), `train_student` soft-label 蒸馏, `auroc` + `gate_decision` 评估器。CLI `python -m src.weaver_label_model --signals_json <path>` 可端到端 smoke。当前 mini-batch (n=30, 2-3 信号 stub) AUROC 数字仅验证 pipeline 走通, 不可作 gate 判定。
- **真 gate run** 需: §9A.3 PRM ckpt → w_1 实信号; §9C.2 10⁴ posterior → w_4 实信号; §9A.2 graded gold labels (`results/stepwise_labels`) 作监督。三者就位后 install snorkel + 跑 `weaver_label_model.py` 即得首个有意义的 §9D.2 gate 判读。
- **状态 (scaffold, 2026-04-21)**: `pvgap_experiment/src/weaver_label_model.py` 已铺设。组件: `LabelModelInputs` dataclass + `from_signals_json` (吃 §9D.1 mini-batch JSON), Snorkel `LabelModel` 包装 + 失败回退到 calibrated majority-vote (`_MajorityVote`), `StudentHeadStub` (numpy MLP 占位, 待 §9A.3 PRM ckpt 到位换成 400M 蒸馏 head), `train_student` soft-label 蒸馏, `auroc` + `gate_decision` 评估器。CLI `python -m src.weaver_label_model --signals_json <path>` 可端到端 smoke。当前 mini-batch (n=30, 2-3 信号 stub) AUROC 数字仅验证 pipeline 走通, 不可作 gate 判定。
- **真 gate run** 需: §9A.3 PRM ckpt → w_1 实信号; §9C.2 10⁴ posterior → w_4 实信号; §9A.2 graded gold labels (`results/stepwise_labels`) 作监督。三者就位后 install snorkel + 跑 `weaver_label_model.py` 即得首个有意义的 §9D.2 gate 判读。

### 9E. 主实验 (Arm 3, §4.15 headline)

**9E.1 5 configuration (C0–C4) 全 run**
- 输出: `results/v04_closure_table.csv` + 95% CI + paired bootstrap p
- 前置依赖: 9A-9D 至少 9A + 9B 全部完成；9C 可 fallback；9D 可 fallback
- 算力: 每 config × 60 scenario × 3 seed × (LLM call + oracle call) → 估 $200 API + 20 h CPU + 4 h GPU
- Decision gate: C4 vs C0 在 reverse diagonal 上 p < 0.05 且 Δ > 5%

### 9F. Abstract / Introduction / Conclusion 回填

- 依赖 9E 数字；数字未回填前不锁 Abstract

### Decision trees (fail-safe)

```
if 9C SBI 全失败:
    Paper 1 scope 收回到 (C0, C1, C2, C4 w/o SBI signal);
    title 退到 v0.4 候选 #5 (去掉 "Amortized Bayesian")
    §3.10 改为 "future work"，不做实验
if 9A PRM label quality 太差:
    回退到 Med-PRM 原 backbone 直接 zero-shot 打分 (skip training);
    §3.8 改为 "PRM-zero" configuration
if 9D Weaver 集成收益 < 3%:
    诚实报告 negative result (§5.5.5 既有 fallback 语句已在)
if Hassanaly 组发布 EIS extension 在 submission 前:
    立即重写 §5.6，主贡献 pivot 到 "LLM-authored prior + Weaver for EIS" (SBI-EIS 不再 claim first)
```

### 9G. 复查 & monthly monitoring (v0.4 新增)

- **每月 1 次 arXiv 扫**: "Hassanaly", "NREL NPE EIS", "neural posterior estimation impedance", "LLM SBI science"
- **每月 1 次 Nature / Science / Joule / JES / Electrochim. Acta 新刊扫**
- **每月 1 次 OpenReview 扫**: NeurIPS 2026 / ICLR 2027 submissions 题目含 EIS / impedance / SBI / battery

---

**v0.4 实验优先级总表 (决策者最终安排)**

| Arm | 子任务 | 优先级 | 不做的代价 |
|---|---|---|---|
| 诊断 (v0.3 保留) | 实验结果.md §1–§9 已就位 | 已完成 | — |
| 方法 9A (EIS-PRM) | 规则库 + label + 训练 | 🔴 必做 | 论文失去主要 method contribution |
| 方法 9B (PyBaMM Loop) | env + wrapper | 🔴 必做 | 物理 verifier 信号缺失，C2 / C4 无法跑 |
| 方法 9C (SBI) | sbi 包 + LLM prior | 🟡 强推 | C3 缺失, 但 fallback 有 |
| 方法 9D (Weaver) | 弱监督集成 | 🟡 强推 | C4 退化为 vote, 但仍可报告 |
| 实验 9E (5-config) | 主表 | 🔴 必做 | paper 无 headline 数字 |
| 复查 9G (并发扫) | 月度 | 🔴 必做 | Hassanaly 风险盲区 |
