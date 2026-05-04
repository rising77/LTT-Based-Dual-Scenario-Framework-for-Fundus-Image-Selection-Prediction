# LTT-Based-Dual-Scenario-Framework-for-Fundus-Image-Selection-Prediction
基于LTT框架下眼底分级影像的双场景风险控制方法
# LTT双场景眼底图像分级风险控制框架

## 📌 项目简介

本项目针对眼底图像分级任务中的**高风险误判问题**，提出一种基于 LTT 框架的双场景风险控制方法。

方法结合统计推断与深度学习模型，在 Clopper–Pearson（CP）置信区间理论基础上，构建**并联–串联双重决策机制**，实现对模型预测结果的风险约束与可靠性控制。

该方法适用于医学图像分类中对**低误诊率、高可靠性**要求的应用场景。

---

## 🧠 核心思想

传统深度学习方法仅输出类别预测，缺乏严格的统计风险保证。

本项目提出：

> 基于 Clopper–Pearson 置信区间的双场景风险控制框架（LTT）

---

### 🔹 1. Clopper–Pearson 置信区间

用于对分类正确率进行**严格统计下界估计**：

用于保证在给定置信水平下模型误差不超过设定阈值。

---

### 🔹 2. 双场景决策机制

#### ✔ 并联机制（Parallel Rule）
- 多个条件同时满足时接受预测
- 强化高置信样本筛选
- 提高整体可靠性

#### ✔ 串联机制（Serial Rule）
- 逐级过滤不确定样本
- 降低误判风险
- 提升安全性

---

### 🔹 3. 双场景风险控制

构建：

- **高覆盖率场景（Coverage-oriented）**
- **高安全性场景（Risk-oriented）**

实现不同应用需求下的自适应决策策略。

---

## 🏗 方法框架

整体流程如下：

1. 输入眼底图像
2. CNN（EfficientNet-B3）提取特征
3. 输出类别概率
4. 构建 Clopper–Pearson 置信区间
5. 进行并联 / 串联风险控制决策
6. 输出最终预测或拒绝（defer）

---

## ⚙️ 模型结构

- Backbone：EfficientNet-B3（timm实现）
- 分类任务：眼底图像分级
- Loss：Focal Loss（γ = 2.0）
- 推理策略：选择性预测（Selective Prediction）

---

## 📊 风险控制机制

### 📌 Clopper–Pearson 区间
用于估计分类准确率的统计置信界：

保证在置信水平 \(1 - \delta\) 下满足风险约束。

---

### 📌 决策输出

模型输出三种结果：

- ✔ Accept（接受预测）
- ✔ Reject（拒绝预测 / 转人工）
- ✔ Conditional Accept（条件接受）

---

## 📁 项目结构
LTT双场景/
│
├── src/
│ ├── model.py # EfficientNet + FocalLoss
│ ├── data.py # 数据加载
│ ├── train.py # 训练与评估
│ ├── uncertainty.py # 不确定性计算（可扩展）
│
├── scripts/
│ ├── 01_train_baseline.py
│ ├── 02_compute_uncertainties.py
│ └── 03_eval_only.py
│
├── config.py # 全局配置
├── outputs/ # 实验结果与图像
└── README.md
