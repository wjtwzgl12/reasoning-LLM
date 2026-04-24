


# Reasoning LLM 电化学推理的 Plausibility-Validity Gap：系统性表征与诊断

> 最后更新: 2026-04-13  
> 版本: v5.0 (PV-Gap 统一版 — 与完整实验方案 v2.0 对齐)  
> 状态: 验证阶段  
> 核心转变: 从 v4.1 的"5个模块补丁集合"转变为"1个核心问题驱动的统一方法论"

---

## 0. 版本演进与核心转变

### 为什么需要 v5.0？

v4.1 的 5 个模块（M1-M5）本质上是 5 个不同论文方法的拼凑，每个模块都在修补别人方法的缺陷：

```
v4.1的问题：
KRUX有混淆变量 → 加4个控制组 (修KRUX的方法论)
忠实性指标≈随机基线 → 改名"敏感性分析" (回避问题)
LLM-Modulo不适用 → 降级为"一致性检查" (保住模块)
多轮退化 → 限制K=1 (回避退化)
Math PRM无效 → 降级为对照 (保住模块)
```

**每个模块都在修补别人的方法，不是在提出自己的方法论 → "为了解决问题而解决问题"。**

### v5.0 的核心：1 个问题 + 1 个方法论

> **核心问题：Reasoning LLM 在电化学推理上的 Plausibility-Validity Gap 有多大？什么决定了它能否被弥合？**
> 
> **方法论：Physical-Ground-Truth Probing — 用已知物理 ground truth 的受控变体，系统性探测 LLM 推理的 plausibility-validity gap。**

### 有机一致的论文长什么样

| 论文 | 1个声明 | 1个操作 | 结果是推论 |
|------|--------|--------|----------|
| GSM-Symbolic (ICLR 2025) | "如果LLM真推理，换数字不应影响" | 换数字 | 性能波动±15% → 模式匹配 |
| "Reasoning or Reciting?" (NAACL 2024) | "如果真推理，反事实不应退化" | 翻转假设 | 一致性退化 → 背诵非推理 |
| "Faith and Fate" (NeurIPS 2023) | "推理=计算图遍历" | 增加图复杂度 | 性能归零 → 线性化匹配 |

**共同特征：1个可证伪声明 → 1个实验范式 → 结果是自然推论。v5.0 遵循这个模式。**

---

## 1. 三个关键词的有机关系

### 为什么不是"为了组合而组合"？

```
    电化学可解释性 ←——————————————→ 大模型可解释性
         ↑                              ↑
         |     reasoning LLM            |
         |     (研究对象)               |
         |                              |
    LLM帮助解释电化学          电化学帮助解释LLM
    (生成可读诊断)             (提供ground truth验证推理)
```

| 连接 | 逻辑 | 文献依据 |
|------|------|---------|
| 电化学 → 大模型可解释性 | 电化学有严格物理定律(Nernst、Butler-Volmer、KK)，提供**可验证的ground truth**。大多数LLM可解释性研究在数学/代码上做，科学领域几乎空白 | KRUX (arXiv:2508.19202); Anthropic circuit tracing用数学做探测 → 电化学是未探索的科学测试床 |
| 大模型可解释性 → 电化学 | 理解LLM的**失败模式和知识边界**，才能知道何时信任LLM诊断 | ChemBench (Nature Chemistry 2025): LLM平均优于人类但在特定子任务灾难性失败且过度自信 |
| Reasoning LLM的角色 | Reasoning LLM产生**显式推理链**(extended thinking)，可以被分析和验证。Base LLM不产生显式推理过程 | DeepSeek-R1 (arXiv:2501.12948): `<think>`标签显式区分推理过程 |

**测试：去掉任何一个组件，方向是否仍然成立？**

| 去掉 | 结果 | 说明 |
|------|------|------|
| 去掉"电化学" | 变成通用LLM可解释性研究 | 失去**科学领域ground truth**的独特价值 |
| 去掉"大模型可解释性" | 变成"用LLM做电化学"(老路) | 面临所有已知的致命问题 |
| 去掉"reasoning LLM" | 用base LLM | 失去**显式推理链**，无法做步骤级分析 |

**三个组件都不可去除 → 组合是有机的。**

---

## 2. 核心概念：Plausibility-Validity Gap

### 2.1 什么是 PV-Gap

**定义：** LLM产生的科学推理在表面层面（语法正确、术语准确、逻辑通顺）通过合理性检查，但在深层物理约束上（数值精确性、因果正确性、机制有效性）存在错误的现象。

**来源与理论根基：**

| 来源 | 贡献 | 引用 |
|------|------|------|
| 术语命名 | 化学合成中首次命名"Plausibility-Validity Gap" | arXiv:2507.07328 (IEEE MCSoC 2025) |
| 现象观察 | LLM平均超越人类但在特定子任务**灾难性失败且过度自信** | ChemBench (Nature Chemistry 2025) |
| 根因分析 | next-token prediction优化linguistic plausibility而非constraint satisfaction | arXiv:2511.10381 |
| 理论基础 | plausible conjecture ≠ rigorous proof; LLM只做conjecture | Polya "Mathematics and Plausible Reasoning" (1954) |
| 量化先例 | format adherence 96.3% vs synthesis feasibility 74.4% — 差距21.9个百分点 | arXiv:2507.07328 |
| 评估理论 | face validity vs construct validity 的系统性区分 | arXiv:2511.04703 (NeurIPS 2025 Workshop) |

### 2.2 PV-Gap vs 一般幻觉

| | 一般幻觉 | Plausibility-Validity Gap |
|---|---------|-------------------------|
| 表面线索 | 常可检测(自相矛盾、无中生有) | **无法从表面检测**(术语、逻辑、格式全对) |
| 检测方式 | 自动化(semantic entropy等) | **需要领域专业知识** |
| 来源 | 训练数据缺失/噪音 | **训练目标本身**(plausibility ≠ validity) |
| 危险性 | 用户可能发现 | **用户几乎无法发现** → 更危险 |
| 文献 | Farquhar Nature 2024 (semantic entropy) | arXiv:2402.04614 (Faithfulness vs. Plausibility) |

### 2.3 PV-Gap 的操作性定义

```
对每条推理链 c:
  P(c) = plausibility_score(c)  ∈ [0, 1]  (表面合理性)
  V(c) = validity_score(c)      ∈ [0, 1]  (物理有效性)
  
  PV-Gap(c) = P(c) - V(c)
  
  如果 PV-Gap > 0: 推理比实际更"看起来对" → plausible but invalid
  如果 PV-Gap ≈ 0: 表面和深层一致 → 可信推理
  如果 PV-Gap < 0: 推理看起来差但实际对 → (不太常见)
  
  模型级 PV-Gap = mean(PV-Gap(c)) for all c
```

### 2.4 为什么电化学是度量 PV-Gap 的理想测试床

| 特性 | 数学/代码 | 通用科学 | **电化学** |
|------|---------|---------|----------|
| Ground truth | ✅ 计算器/执行器 | ❌ 通常没有 | ✅ 物理定律(Nernst, Butler-Volmer, KK) |
| 推理过程可观测 | ✅ 步骤可验证 | ❌ 推理结构不固定 | ✅ 固定分析框架(数据质量→特征→机制→鉴别) |
| 连续量值 | ❌ 离散对错 | 部分 | ✅ 参数有物理范围和单位 |
| 可控模拟 | ✅ 换数字 | ❌ | ✅ PyBaMM精确模拟 |
| LLM训练覆盖 | 高(大量数学/代码) | 中 | **低** → gap可能更大更可见 |
| 参考文献 | GPQA, GSM-Symbolic | ChemBench, SciReas | **空白** → 贡献空间 |

---

## 3. 已知事实 vs 待验证假设

### 3.1 已证明的事实

#### A. 关于 LLM 科学推理的 PV-Gap 现象

| 编号 | 事实 | 来源 | 与PV-Gap的关系 |
|------|------|------|-------------|
| F1 | 化学合成中format adherence(96.3%) vs synthesis feasibility(74.4%)存在21.9pp差距 | arXiv:2507.07328, IEEE MCSoC 2025 | **PV-Gap在化学中已被观察到** |
| F2 | ChemBench: LLM平均超越人类但在特定子任务灾难性失败且过度自信 | Nature Chemistry 2025 | 过度自信 = plausibility高而validity低 |
| F3 | next-token prediction优化linguistic plausibility而非constraint satisfaction | arXiv:2511.10381 | **PV-Gap的根因是训练目标** |
| F4 | Reasoning LLM的CoT不总是忠实的: Claude 25%提及注入hint, R1 39% | Anthropic 2025, arXiv:2505.05410 | 推理链表面合理不等于真实反映计算 |
| F5 | GSM-Symbolic: LLM在语义等价数学变体上性能波动±15% | arXiv:2410.05229, ICLR 2025 | **受控扰动可以暴露推理的非鲁棒性** |

#### B. 关于电化学领域的 LLM 能力

| 编号 | 事实 | 来源 | 与PV-Gap的关系 |
|------|------|------|-------------|
| F6 | o1在电化学编码任务85%成功率 | Angewandte Chemie 2025 | LLM有电化学能力但不完美 |
| F7 | BatteryAgent: 物理特征+GBDT+SHAP+LLM达AUROC 98.6%，但LLM核心贡献是文本解释 | arXiv:2512.24686 | LLM解释的plausibility vs validity未被检验 |
| F8 | 不存在电化学推理benchmark | 本研究调查 | **Gap: 需要创建** |
| F9 | EIS诊断存在ECM非唯一性——多个等效电路可完美拟合同一阻抗谱 | Vadhva et al. 2021 ChemElectroChem | 验证validity需要物理约束而非单一答案 |
| F10 | ChatBattery: LLM驱动发现3种新阴极材料(容量+18-29%) | arXiv:2507.16110 | LLM+电化学有正面结果 |

#### C. 关于方法论工具的实际效果（⚠️ 经深度文献验证）

| 编号 | 事实 | 来源 | ⚠️ 关键限制 |
|------|------|------|------------|
| F11 | KRUX: 知识注入提升LLM科学推理性能 | arXiv:2508.19202 | ⚠️ **方法论有严重混淆变量**：无无关知识控制、位置偏好、格式效应。见§6.1 |
| F12 | ThinkPRM: ProcessBench 86.5% F1 | arXiv:2504.16828 | ⚠️ **仅在数学benchmark验证**；LLM验证LLM存在循环依赖 (Huang ICLR 2024) |
| F13 | VersaPRM: 数学PRM在化学MC上+2.00pp (58.67%→60.67%) | VersaPRM, ICML 2025 | ⚠️ **效果微弱**；Math PRM在Biology/Philosophy近零或负 |
| F14 | Prompt格式变化可造成最高76pp准确率差异 | Sclar et al. ICLR 2024 | KRUX的KI注入无法排除格式效应 |
| F15 | 多轮LLM反馈：仅23%平滑收敛，41%振荡，36%混沌；5轮后退化37.6% | arXiv:2506.11022, IEEE-ISTAS 2025 | 多轮反馈循环不可靠 |
| F16 | 忠实性指标在因果验证下≈随机基线 | Zaman & Srivastava, EMNLP 2025 | "忠实性"概念本身有争议 |
| F17 | LLM先验知识可压倒证据(锚定偏差17.8-57.3%) | PMC 2025; arXiv:2412.06593 | 推理可能是prior-driven而非data-grounded |

### 3.2 待验证假设（围绕PV-Gap）

| 编号 | 假设 | 对应实验 | 预期方向 |
|------|------|---------|---------|
| H1 | 电化学推理中PV-Gap > 0（推理表面合理但物理上有错） | E1 | **预期正面** — ChemBench已观察到此现象 |
| H2 | PV-Gap在机制识别(mechanism_id)阶段最大 | E2 | **有依据的推测** — 需要深层领域知识的阶段gap应最大 |
| H3 | LLM电化学推理部分是prior-driven而非data-grounded | E3 | **预期正面** — 锚定偏差文献(F17)支持 |
| H4 | 注入正确领域知识可缩小PV-Gap | E4 | **待定** — KRUX有混淆变量，需控制组验证 |

**关键特性：无论H1-H4的结果如何，characterization本身就是positive contribution。**

---

## 4. 方法论：Physical-Ground-Truth Probing

### 4.1 核心方法论

> 受 GSM-Symbolic (ICLR 2025) 启发但适配到科学领域：用**已知物理 ground truth**的受控变体，系统性探测 LLM 推理的 plausibility-validity gap。

| GSM-Symbolic (数学) | 本方法论 (电化学) |
|-------------------|-----------------|
| 换数字 → 看性能是否不变 | 换物理参数 → 看推理是否跟随物理变化 |
| NoOp(加无关条件) → 看性能是否下降 | 加无关电化学信息 → 看推理是否被干扰 |
| 测量: 答案正确率变化 | 测量: **推理过程中每步的物理有效性变化** |
| 贡献: 证明数学推理是模式匹配 | 贡献: **量化科学推理的plausibility-validity gap** |

**为什么这不是"为了组合而组合"：**
- GSM-Symbolic只测答案层面(outcome-level) → 我们测推理过程(process-level)
- GSM-Symbolic在数学上不需要领域知识判断 → 电化学需要物理ground truth
- GSM-Symbolic无法解耦"缺知识"vs"推理错误" → 我们用知识注入做诊断

### 4.2 双层评估框架（方法论核心）

```
Layer A: Plausibility 评估 (表面合理性)
  ├── Relevance: 推理步骤是否与问题相关
  ├── Coherence: 步骤之间是否逻辑连贯
  ├── Completeness: 是否覆盖所有必要分析阶段
  ├── Fluency: 语言是否专业流畅
  └── → 评估者: GPT-4o (非领域专家也能判断)
  └── → 来源: CaSE框架 (arXiv:2510.20603)

Layer B: Validity 评估 (深层有效性)
  ├── Feature Citation Accuracy: 引用的数据特征是否存在于输入中
  ├── KK Consistency: EIS数据质量判断是否正确 (impedance.py linKK)
  ├── Numerical Accuracy: 引用的参数值是否与输入一致
  ├── Causal Correctness: 因果推理方向是否正确
  ├── Mechanism Match: 最终诊断是否与ground truth一致
  └── → 评估者: 物理规则(hard constraints) + 领域专家

PV-Gap Score = Plausibility Score - Validity Score
```

**Validity 层的每条规则的 soundness 分析：**

| 规则 | Sound? | 来源 | 说明 |
|------|--------|------|------|
| Feature Citation Accuracy | ✅ | ACM SIGIR ICTIR 2025 | 自动匹配，57%的引用是post-rationalized |
| KK Consistency | ✅ | impedance.py linKK; Boukamp 1995 | 纯数学检查(Hilbert变换) |
| Numerical Accuracy | ✅ | 自动比对 | 数值与输入的直接对照 |
| Causal Correctness | ❌(需人工) | 领域专家 | 条件性因果难以自动编码 |
| Mechanism Match | ✅(PyBaMM题) | PyBaMM ground truth | 模拟题有确定答案 |

### 4.3 普适性论证

**方法论的可迁移性测试：如果把"电化学"换成其他有物理 ground truth 的科学领域，方法论是否仍然成立？**

| 替换为... | ground truth来源 | 可控模拟 | 方法论成立？ |
|---------|----------------|---------|------------|
| 热力学 | 热力学定律 | COMSOL | ✅ |
| 流体力学 | Navier-Stokes | CFD | ✅ |
| 药物化学 | 反应机理 | RDKit | ✅ |
| 材料科学 | 相图/状态方程 | CALPHAD | ✅ |
| 天文学 | Kepler定律 | — | ✅ |

**方法论不依赖于电化学特殊性 → 普适。电化学是 instantiation，不是方法论本身。**

---

## 5. 三层贡献（自然从1个问题推导出）

```
核心问题: Plausibility-Validity Gap 有多大？
    │
    ├── 贡献1: 定义和测量这个gap (方法论贡献)
    │   └── PV-Gap操作性定义 + 双层评估框架 + EChem-Reason Benchmark
    │
    ├── 贡献2: 探测gap的来源 (科学贡献)  
    │   └── 扰动测试 → data-grounded vs prior-driven？
    │   └── 知识注入诊断 → 知识缺失 vs 推理缺陷？
    │
    └── 贡献3: 量化gap的可弥合性 (实践贡献)
        └── 知识注入是否缩小gap → 改进方向指引
```

**所有实验服务于同一个问题，不是独立模块的拼凑。**

---

## 6. 实验设计（从1个问题自然展开）

### 6.0 全局架构

```
核心问题: Plausibility-Validity Gap 有多大？
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
 测量gap          探测gap来源      测试gap可弥合性
 (E1+E2)          (E3)             (E4)
    │               │               │
    ▼               ▼               ▼
EChem-Reason     受控扰动         知识注入
Benchmark        (PyBaMM变体)     (KRUX-EC+控制组)
    │               │               │
    ▼               ▼               ▼
process-level    data-grounded    knowledge gap
validity scoring vs prior-driven  vs reasoning gap
```

---

### 6.1 EChem-Reason Benchmark（为所有实验提供数据基础）

**解决什么问题：** 不存在电化学推理benchmark(F8)。没有有ground truth的评估标准就无法量化PV-Gap。

**设计：**

```
EChem-Reason Benchmark (350题):
├── Level 1: 知识题 (100题)
│   ├── 电化学原理 (Nernst方程, Butler-Volmer, 电双层)
│   ├── EIS基础 (Nyquist图解读, 等效电路含义)
│   └── 退化机制知识 (SEI, 锂析出, LAM的特征)
│
├── Level 2: 单步推理 (100题)
│   ├── "给定DRT峰在τ=10⁻³s且面积增大45%, 最可能的物理过程是什么?"
│   ├── "R_ct从5Ω增大到15Ω, 对应什么退化模式?"
│   └── 每题有step-level ground truth annotation
│
├── Level 3: 多步诊断推理 (100题)
│   ├── 完整EIS诊断场景(结构化文本表征)
│   ├── 需要多步推理: 数据质量→特征提取→机制识别→排除替代
│   ├── 标注: 每阶段的正确推理 + ground truth诊断
│   └── 来源: PyBaMM模拟(ground truth已知) + 文献case study
│
└── Level 4: 对抗性题目 (50题)
    ├── 包含误导信息的诊断场景(NoOp, 反事实)
    ├── 测试LLM是否被先验偏见影响
    └── 方法: 改编GSM-Symbolic (ICLR 2025)
```

**数据来源：**

| 来源 | 用途 | 规模 | 可行性 |
|------|------|------|--------|
| PyBaMM模拟 (pybamm-eis, SPMe/DFN) | Level 2-3, ground truth精确可知 | 150题 | ✅ 已验证可行 |
| 电化学教科书+综述 | Level 1知识题 | 100题 | ✅ 人工整理 |
| 发表论文中的EIS案例 | Level 3真实场景 | 50题 | ✅ 需人工收集 |
| 对抗性构造 (PyBaMM参数变体) | Level 4 | 50题 | ✅ 参考GSM-Symbolic方法 |

**构建方法的结构性缺陷与缓解：**

| 缺陷 | 底层原理 | 缓解 |
|------|---------|------|
| PyBaMM模拟与真实数据的gap | PyBaMM基于P2D/SPMe模型，假设均匀电极、理想化SEI | 混合数据：70%模拟+30%文献真实案例；明确声明适用范围 |
| 题目规模限制(350题) | 人工标注推理步骤耗时(Level 3每题~2h) | 聚焦质量；350题已与ChemBench电化学子集(~200题)相当 |
| 专家标注主观性 | 同一诊断可有多条有效推理路径 | 多专家标注(≥2人)；允许多条正确路径；报告annotator agreement |
| 步骤边界划分无标准 | PRM800K按换行分割过于简单，自由文本中"一步"边界模糊 | 利用电化学固定分析框架(数据质量→特征提取→机制识别→鉴别诊断)作为步骤模板；post-hoc映射而非强制输出 |

**文献先例：**
- ChemBench (Nature Chemistry 2025): 2700+化学题 → EChem-Reason是电化学子领域深化
- SciReas/SciReas-Pro (arXiv:2508.19202): 多学科科学推理benchmark → 补充电化学维度
- GSM-Symbolic (ICLR 2025): 对抗性数学benchmark → Level 4借鉴其方法论
- GPQA (NeurIPS 2024 D&B): "Google-proof"题目方法论 → Level 2-3质量控制参考

---

### 6.2 实验1: 测量 Plausibility-Validity Gap (E1)

**问题：** 电化学推理中，有多少比例的推理"看起来对但物理上错"？

**方法：**

1. 对5个模型收集EChem-Reason全部350题的推理链
2. 双层评估每条推理链的 P(c) 和 V(c)
3. 计算 PV-Gap(c) = P(c) - V(c)

**模型选择：**

| 模型 | 类型 | 为什么选 |
|------|------|---------|
| DeepSeek-R1 | Reasoning (开源) | 有显式`<think>`链; 开源可分析 |
| QwQ-32B | Reasoning (开源) | 轻量reasoning model |
| o1-mini | Reasoning (API) | 商业reasoning model对照 |
| GPT-4o | Base (API) | 非reasoning model对照 |
| DeepSeek-V3 | Base (开源) | 与R1同系列的非reasoning对照 |

**Plausibility 评估维度 (来自 CaSE, arXiv:2510.20603)：**

| 维度 | 定义 | 评估方式 |
|------|------|---------|
| Relevance | 推理步骤是否与问题相关 | LLM judge(无需领域知识) |
| Coherence | 步骤之间是否逻辑连贯 | LLM judge |
| Completeness | 是否覆盖所有必要分析阶段 | 模板匹配 |
| Fluency | 语言是否专业流畅 | LLM judge |

**Validity 评估维度 (需要物理 ground truth)：**

| 维度 | 定义 | 评估方式 | Sound? |
|------|------|---------|--------|
| Feature Citation Accuracy | 引用的数据特征是否存在于输入中 | 自动匹配 | ✅ |
| KK Consistency | EIS数据质量判断是否正确 | impedance.py linKK | ✅ |
| Numerical Accuracy | 引用的参数值是否与输入一致 | 自动比对 | ✅ |
| Causal Correctness | 因果推理方向是否正确 | 领域专家 | ❌(需人工) |
| Mechanism Match | 最终诊断是否与ground truth一致 | 自动(PyBaMM题)或人工(文献题) | ✅(PyBaMM题) |

**预期结果(有文献支撑)：**
- PV-Gap > 0 for most models → gap存在 (ChemBench: 过度自信; arXiv:2507.07328: 96% format vs 74% feasibility)
- Reasoning LLM的PV-Gap < Base LLM → reasoning model产出更valid的推理(但需实测)
- Level 4(对抗性)的PV-Gap最大 → 对抗性输入放大gap
- **首次在电化学领域系统性量化PV-Gap = inherently positive contribution**

**统计方法 (来源: arXiv:2412.00868 DBPA + statsforevals.com)：**
- Bootstrap confidence intervals (而非假设检验)
- 每个模型×Level组合≥50个数据点
- 报告effect size + CIs，不只报p值

---

### 6.3 实验2: 推理过程的 PV-Gap 阶段分析 (E2)

**问题：** PV-Gap在推理的哪个阶段最大？

**方法：** 对E1的推理链按电化学分析阶段分解，分别计算每阶段的PV-Gap。

```
对每个分析阶段 s ∈ {data_quality, feature_extraction, mechanism_id, differential}:
  PV-Gap(s) = mean(P(step) - V(step)) for all steps in stage s
  
→ 产出: PV-Gap Profile (哪个阶段gap最大)
```

**阶段划分方法：** 利用电化学推理的固定分析框架，post-hoc将自由文本推理映射到预定义阶段模板。不强制LLM用模板输出(F21: 外部结构约束对reasoning model有害)。

| 阶段 | 内容 | PV-Gap预期 |
|------|------|-----------|
| data_quality | KK验证、数据完整性检查 | 小(KK是数学检查) |
| feature_extraction | 识别DRT峰、阻抗弧参数 | 中(需要数值准确性) |
| mechanism_id | 将特征映射到物理退化机制 | **最大**(需要深层领域知识) |
| differential | 排除替代诊断 | 中-高(需要比较推理) |

**预期(有依据的推测)：**
- mechanism_id阶段PV-Gap最大 → 需要将DRT峰位置/幅值映射到SEI增长/锂析出等物理过程，这是最需要领域知识的环节
- **这种阶段级PV-Gap profile是首次被测量 → inherently positive contribution**

**注意：E2使用E1的推理链，不需要额外LLM推理调用(CPU处理)。**

---

### 6.4 实验3: 扰动探测——推理是 data-grounded 还是 prior-driven？(E3)

**问题：** 当输入数据改变时（物理上应该改变诊断），推理链是否跟随改变？如果不跟随 → 推理是prior-driven → PV-Gap来自"先验压倒证据"。

**方法（改编自GSM-Symbolic，适配物理约束）：**

```
对每道Level 3题目 q:
  q_original: R_SEI = 25Ω → 正确诊断: SEI增长
  q_perturbed: 用PyBaMM重新模拟 R_SEI = 5Ω → 正确诊断: 无明显SEI退化
  
  cot_original = Generate(model, q_original)
  cot_perturbed = Generate(model, q_perturbed)
  
  if diagnosis(cot_perturbed) 仍然是 "SEI增长":
    → prior-driven (推理没跟随数据变化)
    → PV-Gap来源: 先验偏见
  
  if diagnosis(cot_perturbed) 改为 "无明显退化":
    → data-grounded (推理跟随数据)
    → PV-Gap来源(如果有): 物理推理错误而非先验
```

**三种扰动类型：**

| 类型 | 操作 | 物理预期 | 测量什么 |
|------|------|---------|---------|
| **参数扰动** | 改变退化程度(R_SEI: 25→5Ω) | 诊断应改变 | 推理是否跟随数据 |
| **NoOp** | 添加无关信息("电池曾经历运输振动") | 诊断不应改变 | 推理是否被无关信息干扰 |
| **反事实** | 数据与描述矛盾 | 应检测到矛盾 | 推理依赖数据还是描述 |

**与GSM-Symbolic的关键区别：**
- PyBaMM保证扰动后所有物理量联动一致(不是简单换数字)
- 测量推理过程(process-level)而非仅答案(outcome-level)
- 电化学参数扰动有物理意义(GSM-Symbolic换数字无物理意义)

**结构性缺陷与缓解：**

| 缺陷 | 底层原理 | 缓解 |
|------|---------|------|
| GSM-Symbolic的"模式匹配"结论可能过强 | 性能波动可能来自数值敏感性(arXiv:2502.06753) | PyBaMM保证物理一致性——扰动导致的变化是物理的而非数值的 |
| 扰动幅度的校准问题 | 扰动太大→推理自然会变；扰动太小→变化不够显著 | 用PyBaMM设定"物理上应改变诊断"的最小扰动阈值 |
| 电化学参数联动 | 改变一个参数(如温度)所有相关量都应联动 | PyBaMM的精确物理模型自动处理参数联动 |

**统计方法：**
- 每种扰动≥100对(paired)
- Bootstrap CIs
- McNemar's test (诊断变/不变)

---

### 6.5 实验4: 知识注入诊断——gap 来自缺知识还是推理缺陷？(E4)

**问题：** 如果注入正确的电化学知识，PV-Gap是否缩小？

```
如果 PV-Gap(+knowledge) << PV-Gap(baseline):
  → gap来自知识缺失 → 可通过RAG/知识库弥合
  
如果 PV-Gap(+knowledge) ≈ PV-Gap(baseline):
  → gap来自推理本身 → 知识注入无法解决 → 需要不同策略
```

**实验条件（4条件，含2控制组）：**

| 条件 | 内容 | 目的 |
|------|------|------|
| Baseline | 无注入 | 基线PV-Gap |
| +Knowledge | 电化学KI(来自教科书，如DRT峰→物理过程映射表、退化机制→EIS特征对应关系) | 知识是否缩小gap |
| +Irrelevant | 有机化学KI(等长) | ⚠️ 控制：排除priming/长度效应 |
| +Random | 等长随机文本 | ⚠️ 控制：排除纯长度效应 |

**为什么是4条件而非v4.1的8条件：**
- v4.1的8条件是为了"修KRUX方法论" → 为了解决问题而解决问题
- v5.0只需回答1个问题："知识注入是否缩小PV-Gap？"
- 2个控制组足以排除最大混淆变量(priming和长度)
- 更多控制（格式C4、位置C3）可在后续工作中做，不在本论文中堆积

**结果解读：**

| 场景 | PV-Gap变化 | +Irrelevant效果 | 解读 |
|------|----------|----------------|------|
| A | 缩小 | 无效果 | **知识是瓶颈** → 可弥合 |
| B | 缩小 | 也缩小 | **priming效应** → 不是知识 |
| C | 不变 | — | **推理是瓶颈** → 知识注入无法弥合 |

**无论哪个场景都是positive contribution：**
- 场景A: 电化学知识增强有效的因果证据(比KRUX更严谨)
- 场景B: 揭示KRUX方法论的局限性(控制组实验是KRUX原文缺失的)
- 场景C: 领域差异发现——电化学推理瓶颈不在知识

**KRUX方法论深层缺陷（本实验的改进意义）：**

KRUX注入KI时**至少同时改变了4个变量**：

```
1. 知识内容 (intended manipulation)
2. Prompt长度 → 改变attention分布
3. 文本位置 → 触发position bias (开头最优, Liu et al. TACL 2024)
4. 领域关键词 → priming效应，激活相关参数知识 (Allen-Zhu ICML 2024)
```

KRUX原文承认"不区分两种解释"。本实验的控制组设计是对KRUX方法论的必然改进。

---

## 7. 算法框架与技术路线

### 7.1 整体算法流程

```python
# 阶段1: Benchmark构建
benchmark = build_echem_reason(
    pybamm_sim=150,     # PyBaMM模拟题
    textbook=100,        # 教科书知识题
    literature=50,       # 文献案例
    adversarial=50       # 对抗性题
)

# 阶段2: E1 — 测量PV-Gap
for model in [R1, QwQ, o1_mini, GPT4o, V3]:
    for question in benchmark:
        cot = model.generate(question)
        p_score = plausibility_eval(cot)   # Layer A: LLM judge
        v_score = validity_eval(cot, question.ground_truth)  # Layer B: 物理规则
        pv_gap = p_score - v_score

# 阶段3: E2 — 阶段级PV-Gap分析 (CPU, 无额外LLM调用)
for cot in all_cots:
    stages = post_hoc_stage_mapping(cot)  # 映射到分析阶段
    for stage in stages:
        stage_pv_gap = compute_stage_pv_gap(stage)

# 阶段4: E3 — 扰动探测
for question in level3_questions:
    q_perturbed = pybamm_perturb(question)  # 物理一致的参数扰动
    cot_orig = model.generate(question)
    cot_pert = model.generate(q_perturbed)
    sensitivity = compare_diagnosis(cot_orig, cot_pert)

# 阶段5: E4 — 知识注入诊断
for question in benchmark:
    for condition in [baseline, +knowledge, +irrelevant, +random]:
        cot = model.generate(question, condition)
        pv_gap = compute_pv_gap(cot)
    delta_gap = pv_gap_baseline - pv_gap_knowledge
```

### 7.2 关键技术组件

#### PyBaMM-EIS 模拟协议

```python
import pybamm
import pybamm_eis  # pybamm-eis包

# 基于SPMe模型的EIS模拟
model = pybamm.lithium_ion.SPMe()

# 可控参数（用于E3扰动和题目生成）
params = {
    "SEI resistance [Ohm.m2]": [0.01, 0.05, 0.1, 0.5],  # 控制SEI退化程度
    "Electrode height [m]": 0.137,
    "Current function [A]": 5,
}

# 生成EIS谱
eis_sim = pybamm_eis.EISSimulation(model, parameter_values=params)
impedance = eis_sim.solve(f_eval=np.logspace(-3, 4, 50))

# 提取ground truth
ground_truth = {
    "R_SEI": params["SEI resistance [Ohm.m2]"],
    "dominant_mechanism": "SEI_growth" if params["SEI resistance [Ohm.m2]"] > 0.1 else "healthy",
    "expected_drt_peak": "τ~10⁻³s"
}
```

#### KK 验证 (Validity Layer)

```python
from impedance.validation import linKK

# 验证LLM对EIS数据质量的判断
f, Z = question.eis_data
M, mu, Z_fit, res_real, res_imag = linKK(f, Z)

# 如果LLM说"数据通过KK验证"但实际未通过 → validity错误
kk_pass = (np.mean(np.abs(res_real)) < 0.05) and (np.mean(np.abs(res_imag)) < 0.05)
```

#### Post-hoc 步骤提取与阶段映射 (E2用)

```python
# 不强制LLM用模板输出(F21)
# 而是post-hoc将自由文本映射到预定义阶段

STAGE_KEYWORDS = {
    "data_quality": ["KK", "Kramers-Kronig", "data quality", "noise", "数据质量"],
    "feature_extraction": ["DRT", "peak", "R_ct", "impedance arc", "特征", "峰"],
    "mechanism_id": ["SEI", "lithium plating", "degradation", "退化", "机制"],
    "differential": ["exclude", "differential", "alternative", "排除", "鉴别"]
}

def map_to_stages(cot_text):
    """将自由文本推理链映射到分析阶段"""
    sentences = split_sentences(cot_text)
    stages = {}
    for sent in sentences:
        best_stage = max(STAGE_KEYWORDS, key=lambda s: keyword_match(sent, STAGE_KEYWORDS[s]))
        stages.setdefault(best_stage, []).append(sent)
    return stages
```

---

## 8. 数据集与评估指标

### 8.1 数据集总览

| 数据 | 用途 | 来源 | 规模 |
|------|------|------|------|
| EChem-Reason Benchmark | E1-E4全部实验 | PyBaMM模拟 + 教科书 + 文献 | 350题 |
| PyBaMM扰动变体 | E3扰动探测 | PyBaMM重新模拟 | 100题×3扰动类型 |
| 知识要素(KI)库 | E4知识注入 | Ciucci iScience 2025; Barsoukov教科书 | ~50个KI条目 |
| 控制文本 | E4控制组 | 有机化学教科书(Irrelevant) + 随机文本(Random) | 等长匹配 |
| NASA Battery Aging | 可选: 真实EIS数据验证 | data.nasa.gov | 补充 |

### 8.2 评估指标体系

| 指标 | 定义 | 对应实验 | 自动/人工 |
|------|------|---------|---------|
| **PV-Gap(c)** | P(c) - V(c) per chain | E1核心指标 | 半自动 |
| **PV-Gap(s)** | 阶段级PV-Gap | E2 | 半自动 |
| **Diagnosis Flip Rate** | 扰动后诊断改变的比例 | E3 | 自动 |
| **NoOp Sensitivity** | NoOp扰动导致诊断改变的比例(应为0) | E3 | 自动 |
| **Feature Citation Accuracy** | 引用数据特征的正确率 | E1 Validity层 | 自动 |
| **ΔPV-Gap(knowledge)** | 知识注入后PV-Gap变化量 | E4 | 半自动 |
| **ΔPV-Gap(irrelevant)** | 无关知识注入后PV-Gap变化量(应≈0) | E4控制 | 半自动 |

### 8.3 统计方法

| 方法 | 用途 | 来源 |
|------|------|------|
| Bootstrap CIs | 所有指标的置信区间 | statsforevals.com |
| McNemar's test | 配对诊断变化(E3) | 标准非参数检验 |
| Effect size (Cohen's d) | 模型间PV-Gap差异 | — |
| DBPA framework | Benchmark评估的统计严谨性 | arXiv:2412.00868 |
| ≥50 samples/condition | 每个模型×Level组合的最小样本量 | — |

---

## 9. 从 v4.1 去掉了什么（以及为什么）

| v4.1模块 | 去掉的内容 | 为什么去掉 |
|----------|-----------|----------|
| **M3: EChem-PRM** (作为独立模块) | 不再是独立模块 | → **吸收到PV-Gap的Validity评估层中**。物理规则(KK, 参数范围, 因果方向)不是独立的"验证器模块"，而是度量PV-Gap的工具 |
| **M3.2: 学习型PRM (ThinkPRM/VersaPRM)** | 作为独立实验 | Math PRM在化学上+2pp，不值得作独立实验。如需可在supplementary |
| **M4: 忠实性评估** (整个模块) | 作为核心模块 | 忠实性指标≈随机基线(Zaman EMNLP 2025)。扰动测试的真实测量是"sensitivity"，已整合到E3 |
| **M4.2: 内部表征分析** | circuit tracing | 计算不可行(R1-671B)；且不直接度量PV-Gap |
| **M5: 反馈循环** (整个模块) | 反馈改善推理 | **典型的"为了解决问题而加"**。本论文是characterization(量化gap)，不是engineering(减小gap)。且37.6%退化率 |
| **M5.2: 多轮修正** | K=3迭代 | 仅23%收敛；unsound verifier放大错误 |
| **M5.3: 上界分析** | verifier ceiling | 服务于反馈循环；反馈循环已去掉 |
| **8条件因式实验** | KRUX的8组 | 简化为4条件(含2控制)。8条件是为了"修KRUX方法论" |

**这些不是不好，而是不服务于核心问题"PV-Gap有多大"。它们可以是future work。**

---

## 10. 深度反思：v4.1 的5个关键错误及其修正

### 10.1 错误1: KRUX"知识是瓶颈"作为已证事实

| v4.1声明 | 实际 | 来源 |
|---------|------|------|
| "KRUX证明知识是瓶颈" | KRUX的KI注入是纯prompt拼接，未做无关知识控制 | arXiv:2508.19202 原文承认 |
| "知识注入在所有模型上提升" | 格式变化可达76pp差异 | Sclar ICLR 2024 |
| — | 随机知识≈对齐知识(<1.0 F1 gap) | arXiv:2311.01150 |

**v5.0修正：** KRUX不再是独立模块(M2)，其核心操作(知识注入)整合为E4的一个实验条件，并补充2个控制组。"知识是否是瓶颈"变成E4要回答的诊断问题之一。

### 10.2 错误2: LLM-Modulo 6x 作为反馈有效性证据

| v4.1声明 | 实际 | 来源 |
|---------|------|------|
| "LLM-Modulo 6x提升" | 实际4.6x，在旅行规划上(可判定约束) | Gundawar arXiv:2405.20625 |
| "EChem-PRM是LLM-Modulo实现" | 0篇论文将LLM-Modulo用于科学推理 | 文献搜索 |
| "验证器sound" | 电化学规则是经验指南，provably unsound | Vadhva 2021 |

**v5.0修正：** 反馈循环(M5)整体去掉。物理规则不作为"验证器"独立存在，而是吸收到PV-Gap的Validity评估层。

### 10.3 错误3: "忠实性评估"作为核心模块

| v4.1声明 | 实际 | 来源 |
|---------|------|------|
| "Anthropic方法论成熟" | 所有忠实性指标≈随机基线 | Zaman EMNLP 2025 |
| "首次测量电化学忠实性" | "忠实性"概念本身可能ill-defined | Barez Oxford 2025 |
| "Anthropic 25% + Arcuschin 0.37%作为基线" | 测量完全不同的构念 | 方法论错误 |

**v5.0修正：** 忠实性模块(M4)去掉。扰动测试实际测量的是"推理对输入的敏感性"——这个操作性定义整合到E3(data-grounded vs prior-driven)。

### 10.4 错误4: VersaPRM "+2.00" 暗示 PRM 在化学有效

| v4.1声明 | 实际 | 来源 |
|---------|------|------|
| "+2.00"暗示有效 | 2个百分点MC，绝对~60% | VersaPRM Table 1 |
| Math PRM可迁移到科学 | Biology +0.00~+0.31, Philosophy **-0.13** | 同上 |

**v5.0修正：** 学习型PRM不再出现在实验设计中。如需对比可在supplementary。

### 10.5 错误5: 多轮反馈"预期正面"

| v4.1声明 | 实际 | 来源 |
|---------|------|------|
| "最多K=3轮" | 仅23%平滑收敛，41%振荡，36%混沌 | LLM feedback dynamics 2025 |
| "任何正面改善publishable" | 5轮后退化37.6% | arXiv:2506.11022 |

**v5.0修正：** 反馈循环整体去掉。本论文聚焦characterization(测量gap)而非engineering(减小gap)。

---

## 11. 与现有工作的区别

| 现有工作 | 做了什么 | 本方案的区别 |
|---------|---------|------------|
| ChemBench (Nature Chemistry 2025) | 评估LLM化学知识(outcome-level) | 我们: **process-level** PV-Gap量化 |
| GSM-Symbolic (ICLR 2025) | 对抗性数学benchmark(outcome-level) | 我们: **科学领域** + **process-level** + 物理ground truth |
| KRUX (arXiv:2508.19202) | 知识-推理解耦(无控制组) | 我们: **带控制组的知识注入诊断** |
| BatteryAgent (arXiv:2512.24686) | 物理+SHAP+LLM诊断 | 我们: 关注**推理过程的PV-Gap**(BatteryAgent不分析推理过程) |
| arXiv:2507.07328 | 命名PV-Gap(化学合成) | 我们: **形式化PV-Gap定义** + **双层评估框架** + 电化学instantiation |
| Anthropic CoT faithfulness (2025) | 通用任务的CoT忠实性 | 我们: 避开contested "faithfulness"，用物理ground truth测量sensitivity |
| AgentEIS (J.Mater.Sci. 2025) | ML+LLM EIS电路识别 | 我们: 关注**推理可解释性**(不只是任务性能) |

---

## 12. 论文结构（符合顶会方法论论文模式）

```
1. Introduction: LLM在科学推理中"听起来对但物理上错"的问题
2. Related Work: 幻觉 / construct validity / 科学LLM评估 / 受控扰动方法论
3. Plausibility-Validity Gap: 形式定义 + 操作化度量 + 双层评估框架
4. EChem-Reason Benchmark: 构建方法(电化学instantiation)
5. Experiments:
   E1: 测量PV-Gap (across models × levels)
   E2: PV-Gap Profile (across reasoning stages)
   E3: Perturbation Probing (data-grounded vs prior-driven)
   E4: Knowledge Injection Diagnosis (knowledge gap vs reasoning gap)
6. Results & Analysis
7. Discussion: 普适性 + 对其他科学领域的implications + limitations
8. Conclusion
```

---

## 13. 计算量估计

| 实验 | 规模 | Token估计 |
|------|------|----------|
| E1: PV-Gap测量 | 350题 × 5模型 | ~3.5M |
| E2: 阶段级分析 | (E1的推理链分解，CPU) | 0 |
| E3: 扰动探测 | 100题 × 3扰动类型 × 2模型 | ~1.8M |
| E4: 知识注入 | 350题 × 4条件 × 2模型 | ~5.6M |
| **总计** | | **~11M tokens** |

vs v4.1的~41M tokens → **减少73%** (去掉了不服务于核心问题的模块)

API成本: ~$15-30 (DeepSeek-R1 API)

---

## 14. 资源与时间线

| 阶段 | 时长 | 内容 | 资源 |
|------|------|------|------|
| P1 | 3周 | EChem-Reason benchmark构建 (PyBaMM模拟+题目设计+标注) | CPU + 人工 |
| P2 | 2周 | E1: PV-Gap测量 (5模型×350题) | API/GPU |
| P3 | 1周 | E2: 阶段级分析 (CPU处理E1推理链) | CPU |
| P4 | 2周 | E3: 扰动探测 (PyBaMM变体生成+模型推理) | API/GPU |
| P5 | 2周 | E4: 知识注入诊断 (4条件×2模型) | API/GPU |
| P6 | 2周 | 分析+论文撰写 | — |
| **总计** | **~12周** | | |

---

## 15. 自检：这个方向是否"为了解决问题而解决问题"？

| 测试 | 结果 |
|------|------|
| 去掉任何一个实验，核心问题还能回答吗？ | E1(测量gap)不可去掉；E2-E4可独立省略但每个都给核心问题增加一个维度 |
| 方法论可迁移到其他领域吗？ | ✅ 只需替换物理规则和模拟器 |
| 有1个统一的核心问题吗？ | ✅ "PV-Gap有多大？" |
| 结果是自然推论还是bolted-on？ | ✅ E1→E2→E3→E4是"测量→分析→诊断"的自然递进 |
| 是在提出自己的方法论还是修补别人的？ | ✅ PV-Gap双层评估是自己的框架 |
| v4.1的5个模块中被保留了什么？ | M1→EChem-Reason; M2.1部分→E4; M3.1部分→Validity层; M4.1部分→E3 |
| v4.1的5个模块中被去掉了什么？ | M3.2(PRM)、M4.2(内部表征)、M4.3(校准)、M5(反馈循环)全部 |

---

## 16. 完整引用清单

### PV-Gap 概念来源

| 引用 | Venue | 角色 |
|------|-------|------|
| arXiv:2507.07328 "Bridging the Plausibility-Validity Gap" | IEEE MCSoC 2025 | 命名PV-Gap |
| arXiv:2402.04614 "Faithfulness vs. Plausibility" | 2024 | plausible ≠ faithful |
| arXiv:2511.04703 "Measuring what Matters: Construct Validity" | NeurIPS 2025 Workshop | face validity vs construct validity |
| Farquhar et al. "Semantic Entropy" | Nature 2024 | 幻觉分类学 |
| arXiv:2511.10381 "Plausibility vs Correctness" | 2025 | next-token prediction → plausibility非correctness |
| Polya "Mathematics and Plausible Reasoning" | 1954 | plausible conjecture ≠ rigorous proof |

### 方法论范式来源

| 引用 | Venue | 角色 |
|------|-------|------|
| GSM-Symbolic (arXiv:2410.05229) | ICLR 2025 | 受控扰动范式 |
| "Reasoning or Reciting?" (Wu et al.) | NAACL 2024 | 反事实探测范式 |
| "Faith and Fate" (Dziri et al.) | NeurIPS 2023 | 计算图复杂度范式 |
| CaSE (arXiv:2510.20603) | 2025 | process-level评估框架(relevance+coherence) |
| Binz & Schulz "Using Cognitive Psychology" | PNAS 2023 | LLM作为实验对象范式 |
| CheckList (Ribeiro et al.) | ACL 2020 Best Paper | 行为测试方法论 |
| KRUX (arXiv:2508.19202) | 2025 | 知识-推理解耦(E4参考) |
| "Correctness ≠ Faithfulness in RAG" | ACM SIGIR ICTIR 2025 | Feature Citation Accuracy |

### 电化学与工具

| 引用 | Venue | 角色 |
|------|-------|------|
| ChemBench, Mirza et al. | Nature Chemistry 2025 | LLM科学能力基线 |
| PyBaMM, Sulzer et al. | JORS 2021 | 可控EIS模拟 |
| pybamm-eis | — | EIS仿真接口 |
| impedance.py | JOSS 2020 | KK验证 |
| Barsoukov & Macdonald | Wiley教科书 | 物理约束来源 |
| Ciucci et al. DRT综述 | iScience 2025 | DRT知识 |
| NASA Battery Aging | data.nasa.gov | 真实EIS数据 |
| BatteryAgent arXiv:2512.24686 | 2025 | 最接近的LLM+电化学工作 |
| AgentEIS | J.Mater.Sci. 2025 | LLM+EIS先例 |
| ChatBattery arXiv:2507.16110 | 2025 | LLM+电池正面结果 |
| Zheng et al. | Angewandte Chemie 2025 | LLM+电化学编码 |
| Vadhva et al. | ChemElectroChem 2021 | ECM非唯一性 |

### 深度反思文献（证明v4.1方法的缺陷）

| 引用 | Venue | 证明了什么 |
|------|-------|----------|
| Sclar et al. | ICLR 2024 | prompt格式76pp差异 → KRUX混淆变量 |
| Liu et al. "Lost in the Middle" | TACL 2024 | context位置偏好 → KRUX混淆变量 |
| arXiv:2311.01150 | 2023 | 随机知识≈对齐知识 → KRUX结论可能不成立 |
| Allen-Zhu & Li | ICML 2024 | KI可能是检索线索而非新知识 |
| Zaman & Srivastava | EMNLP 2025 | 忠实性指标≈随机基线 |
| Barez et al. "CoT Is Not Explainability" | Oxford 2025 | CoT非必要非充分 |
| Huang et al. "LLMs Cannot Self-Correct" | ICLR 2024 | ThinkPRM循环依赖 |
| Setlur et al. | arXiv:2411.17501 | imperfect verifier ceiling |
| arXiv:2506.11022 | IEEE-ISTAS 2025 | 多轮退化37.6% |
| Gundawar et al. | arXiv:2405.20625 | LLM-Modulo实际4.6x，依赖可判定验证 |
| Vadhva et al. | ChemElectroChem 2021 | ECM非唯一性 → verifier unsound |

### LLM推理限制文献

| 引用 | Venue | 角色 |
|------|-------|------|
| DeepSeek-R1 arXiv:2501.12948 | Nature 2025 | Reasoning model基础 |
| Anthropic arXiv:2505.05410 | 2025 | CoT不忠实现象 |
| arXiv:2504.06564 "Thinking Out Loud" | 2025 | Reasoning model过度自信 |
| Ahn et al. arXiv:2402.14903 | 2024 | Tokenizer数值问题 |
| arXiv:2412.06593 | 2024 | LLM锚定偏差 |
| arXiv:2502.06753 | 2025 | GSM-Symbolic部分波动可由计算难度解释 |
| GPQA, Rein et al. | NeurIPS 2024 | "Google-proof"题目方法论 |
| arXiv:2412.00868 | 2024 | DBPA统计框架 |

---

## 17. v4.1 → v5.0 变更日志

```
核心转变: 5个独立模块 → 1个核心问题(PV-Gap) + 1个方法论(Physical-Ground-Truth Probing)

结构变更:
- 删除: M2(KRUX-EC作为独立模块) → 知识注入整合为E4
- 删除: M3(EChem-PRM作为独立模块) → 物理规则吸收为Validity评估层
- 删除: M4(忠实性评估) → 扰动测试整合为E3
- 删除: M5(反馈循环) → 完全去掉
- 新增: §2(PV-Gap核心概念)、§4(Physical-Ground-Truth Probing方法论)、§5(三层贡献)
- 重组: 实验从5个(E1-E5)精简为4个(E1-E4)，每个直接服务核心问题

事实表更新:
- 重组为围绕PV-Gap的3类(现象、能力、方法论工具)
- 所有⚠️标记保留
- 新增F1-F3(PV-Gap专门的文献支撑)

计算量: ~41M → ~11M tokens (减少73%)
时间线: ~13周 → ~12周

核心逻辑变化:
v4.1: "理解LLM电化学推理 → 验证 → 改善" (5个模块各有各的方法)
v5.0: "PV-Gap有多大？" → 测量 → 分析来源 → 诊断可弥合性 (1个问题4个自然实验)
```
