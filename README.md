# 标普500股票的分析师评级与目标价准确性研究

本项目针对标普 500 (S&P 500) 分析师评级与目标价数据集，建立了完整的计量经济学研究流水线，包含：
1. **数据清洗与特征工程脚本** (`01_data_acquisition_firm.py`)
2. **控制机构与年份双重固定效应的非线性回归与 Lind & Mehlum 拐点检验脚本** (`02_data_analysis_firm.py`)
3. **学术可视化图表渲染与行业分组描述性统计导出脚本** (`03_data_visualization_firm.py`)

项目重点探讨了分析师目标价“预期看涨程度”与未来 180 天实际超额收益率之间的**非线性关系与倒 U 型形态**，并结合 **Lind & Mehlum 拐点检验** 评估了**半导体高成长板块 vs. 传统价值行业**的异质性调节效应。

---

## 目录

1. [项目概述与变量定义表](#1-项目概述与变量定义表)
2. [数据清洗与特征工程脚本 (`01_data_acquisition_firm.py`)](#2-数据清洗与特征工程脚本-01_data_acquisition_firmpy)
3. [计量建模与板块异质性分析脚本 (`02_data_analysis_firm.py`)](#3-计量建模与板块异质性分析脚本-02_data_analysis_firmpy)
4. [学术可视化与描述统计导出脚本 (`03_data_visualization_firm.py`)](#4-学术可视化与描述统计导出脚本-03_data_visualization_firmpy)
5. [核心实验结果与数据导出清单](#5-核心实验结果与数据导出清单)
6. [环境配置与运行指南](#6-环境配置与运行指南)

---

## 1. 项目概述与变量定义表

### 1.1 项目概述
本研究旨在探讨分析师评级与目标价调整对股票未来超额收益的预测能力及其非线性机制。通过提取分析师目标价变更数据，剔除未满 180 天观测期与非法目标价样本，并在机构-日-股票维度进行聚合处理。在模型中引入**分析师机构固定效应 (`C(firm)`)** 与 **年份固定效应 (`C(event_year)`)** 以消除机构偏好与宏观周期带来的遗漏变量偏差。对清洗后样本进行二次多项式 OLS 回归（采用 **HC1 稳健标准误**），探讨分析师预期看涨程度对未来 180 天实际超额收益率的作用规律，并进一步验证半导体与传统板块的异质性表现。

---

### 1.2 变量表与测量基础

根据模型设定，变量在计量模型中的角色说明如下表所示：

| 变量名称 | 变量符号 | 变量类型 | 数据来源 / 计算公式 | 在模型中的角色与含义 |
| :--- | :--- | :--- | :--- | :--- |
| **实际超额收益** | $Y$ (`actual_excess_return_180d`) | **因变量** | `forward_return_180d_pct` - `Yearly_Market_Mean` | 剔除同年大盘系统性走势后的 180 天个股 Alpha 超额回报率。 |
| **预期看涨程度** | $X$ (`expected_bullishness`) | **自变量** | (`current_price_target` - `prior_price_target`) / `prior_price_target` | 量化分析师给出的相对目标价看涨溢价比例（采用 T-1 日收盘价为基准）。 |
| **看涨二次项** | $X^2$ (`expected_bullishness_sq`) | **自变量** | `expected_bullishness` ** 2 | **捕捉倒 U 型/凹凸非线性关系**（验证极端看涨惩罚机制）。 |
| **机构固定效应** | $FE_{\text{firm}}$ (`C(firm)`) | **固定效应** | 提取自 `firm` 字段（券商/分析师机构） | 控制不同分析师机构的能力差异、评级乐观偏好与机构固定效应。 |
| **年份固定效应** | $FE_{\text{year}}$ (`C(event_year)`) | **固定效应** | 提取自 `event_date` 时间戳年份 | 控制宏观牛熊市等时间固定效应（Year FE）带来的遗漏变量偏差。 |
| **板块分类** | $M$ (`industry_group`) | **异质性分组变量** | 根据 Ticker 精准映射为 `Semiconductors` / `Traditional Industry` / `Other/General` | 检验高成长（半导体）与传统稳健行业对看涨溢价容忍拐点的**异质性差异**。 |

---

## 2. 数据清洗与特征工程脚本 (`01_data_acquisition_firm.py`)

### 2.1 脚本职责说明
`01_data_acquisition_firm.py` 专门负责“**数据清洗、特征构建与行业划分**”。包含严格的数据过滤、180天截断控制、机构-日级均值聚合、双端极值截尾、对齐论文公式的特征计算，最终导出清洗后的数据集 `processed_data.csv`。

---

### 2.2 核心清洗与特征构建逻辑

#### 1. 基础清洗与截断控制 (Data Cleaning & Truncation Control)
- **关键列缺失值剔除**：剔除股票代码 (`ticker`)、发布日期 (`event_date`) 或分析师机构 (`firm`) 缺失的无效记录。
- **180 天时间窗口截断**：剔除距离当前不足 180 天（即 `forward_return_180d_pct` 缺失）的样本，消除回看偏差与前瞻偏误 (Look-ahead Bias)。
- **目标价有效性检验**：要求 `current_price_target > 0` 且 `prior_price_target > 0`，消除数据库录入错漏。

#### 2. 机构-交易日层面聚合 (Daily Mean Aggregation)
- 为防范同一机构在同一交易日发布多份研报造成的信息重叠偏差，同时满足后续模型对机构固定效应（Firm FE）的控制需求，本脚本将观测层面设定为 `(ticker, event_date, firm, event_year)`。
- 对同机构同日同股票的 `current_price_target`、`prior_price_target` 和 `forward_return_180d_pct` 取均值，合并为单一观测值。

#### 3. 核心特征工程 (Feature Engineering)
- **预期看涨程度 ($X$)**：以研报发布前基准价（`prior_price_target`）为分母，计算预期看涨溢价率：
  ```python
  expected_bullishness = (current_price_target - prior_price_target) / prior_price_target
  ```
- **离群值截尾 (Trimming)**：采用双端 0.5% 分位数截尾（保留 $[P_{0.5}, P_{99.5}]$ 区间），防止极端错报数据拉拽 OLS 回归线。
- **非线性二次项 ($X^2$)**：构建 `expected_bullishness_sq = expected_bullishness ** 2`。
- **实际超额收益率 ($Y$)**：计算同年份标普 500 平均 180 天收益率作为大盘基准，个股收益扣除大盘基准：
  ```python
  actual_excess_return_180d = forward_return_180d_pct - yearly_market_return
  ```

#### 4. 板块与行业精准映射
- **半导体板块 (`Semiconductors`)**：包含 14 只高成长、高估值弹性龙头股票：`NVDA`, `AVGO`, `AMD`, `INTC`, `QCOM`, `TXN`, `AMAT`, `LRCX`, `ADI`, `MU`, `KLAC`, `MCHP`, `MPWR`, `ON`。
- **传统行业 (`Traditional Industry`)**：涵盖金融 (`JPM`, `V`, `MA`, `BAC`, `WFC`, `GS`, `MS`, `C`, `BLK`)、能源 (`XOM`, `CVX`, `COP`, `SLB`, `EOG`, `MPC`, `PSX`)、必需消费 (`PG`, `KO`, `PEP`, `COST`, `WMT`, `PM`, `MO`)、工业 (`CAT`, `GE`, `HON`, `LMT`, `BA`, `MMM`, `UNP`) 及公用事业与传统商业 (`NEE`, `DUK`, `SO`, `D`, `MCD`, `HD`) 共 35 只稳定现金流标的。
- **通用组 (`Other/General`)**：其余标普 500 成分股样本。

---

## 3. 计量建模与板块异质性分析脚本 (`02_data_analysis_firm.py`)

### 3.1 脚本职责说明
`02_data_analysis_firm.py` 专门负责“**控制机构与年份固定效应的非线性回归、Lind & Mehlum 倒 U 拐点严格检验、板块异质性分组对比及学术拟合图表导出**”。全部分析结果自动保存至 `output_firm/` 目录。

---

### 3.2 核心分析模块与回归逻辑

#### 1. 全样本二次非线性 OLS 回归 (`模块 2`)
- **计量回归方程**：
  $$Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \sum \delta_j \text{Firm}_j + \sum \gamma_t \text{Year}_t + \epsilon$$
- **标准误修正**：采用 White 异方差稳健标准误 (`cov_type='HC1'`)，修正金融截面数据中的异方差问题。
- **固定效应控制**：同时包含机构固定效应 (`C(firm)`) 和年份固定效应 (`C(event_year)`)，有效控制券商乐观偏误差异与宏观经济周期。

#### 2. Lind & Mehlum 倒 U 型极值拐点严谨检验 (Turning Point Verification)
- **极值点推导**：对回归方程求一阶导数：
  $$\frac{dY}{dX} = \beta_1 + 2\beta_2 X = 0 \implies X^* = -\frac{\beta_1}{2\beta_2}$$
- **判定与检验条件**：
  1. 二次项系数 $\beta_2 < 0$ 且 $p\text{-value} < 0.05$。
  2. 验证拐点 $X^*$ 是否位于样本自变量实际区间 $[X_{\min}, X_{\max}]$ 内。
  3. 检验区间左端点斜率 $\text{Slope}_{\min} = \beta_1 + 2\beta_2 X_{\min} > 0$ 以及右端点斜率 $\text{Slope}_{\max} = \beta_1 + 2\beta_2 X_{\max} < 0$。

#### 3. 板块异质性分组回归 (`模块 3`)
- 分别对半导体板块 (`Semiconductors`) 和传统行业 (`Traditional Industry`) 进行独立回归，导出包含估计系数、标准误 (SE)、95% 置信区间 (Confidence Intervals) 与拐点百分比的对比汇总表。

#### 4. 学术拟合图表渲染 (`模块 4`)
- 绘制双板块倒 U 型二次拟合曲线叠加散点对比图以及各板块平均超额收益柱状图，保存为 `semicon_vs_traditional_chart.png`。

---

## 4. 学术可视化与描述统计导出脚本 (`03_data_visualization_firm.py`)

### 4.1 脚本职责说明
`03_data_visualization_firm.py` 专门负责“**全套学术分析图表 (图1~图6 全中文渲染) 的高分辨率绘制**”以及“**行业分组及全样本描述性统计表格导出**”。

---

### 4.2 核心图表与数据表清单

| 图表 / 文件名称 | 模块类型 | 内容与学术含义说明 |
| :--- | :--- | :--- |
| `图1_行业样本数量柱状图.png` | 图表渲染 | 半导体与传统行业观测样本数量柱状图（直观展示两大板块样本分布）。 |
| `图2_看涨溢价分布直方图.png` | 图表渲染 | 半导体与传统行业看涨溢价（Expected Bullishness）分布直方图与 KDE 密度曲线。 |
| `图3_多周期收益相关系数热力图.png` | 图表渲染 | 看涨溢价与多周期超额收益率（30天、90天、180天、365天）相关系数热力图。 |
| `图4_全样本二次拟合图.png` | 图表渲染 | 全样本分析师看涨预期溢价与 180 天超额收益二次拟合关系图及极值拐点标记。 |
| `图5_行业异质性双面板分析图.png` | 图表渲染 | 板块异质性双面板分析图（左图：半导体与传统行业拟合曲线对比；右图：各板块平均超额收益柱状图）。 |
| `图6_看涨溢价十分位数分组折线图.png` | 图表渲染 | 看涨溢价十分位数（Decile Quantiles）分组平均 180 天超额收益折线图。 |
| `行业分组描述统计.csv` | 数据表格 | 包含半导体、传统行业、通用组及全样本在看涨溢价与 180 天超额收益上的样本量、均值、标准差、最小值、分位数与最大值（保存至 `output_firm/` 目录）。 |

---

## 5. 核心实验结果与数据导出清单

执行全部脚本后，`output_firm/` 目录下生成的关键结果文件如下：

| 文件名 | 文件格式 | 内容描述 |
| :--- | :--- | :--- |
| `regression_results.csv` | CSV | 全样本 OLS 二次回归系数明细表（包含 Coefficient, Std_Error, t_stat, p_value，涵盖机构与年份固定效应系数）。 |
| `semicon_vs_traditional_comparison.csv` | CSV | 全样本 vs. 半导体 vs. 传统行业 样本量、线性 Beta1 (含 SE 与 95% 置信区间)、二次项 Beta2 (含 SE 与 95% 置信区间)、P 值及拐点百分比对比汇总表。 |
| `半导体与传统行业学术拟合对比图.png` | PNG | 学术对比图：双板块二次拟合曲线对比与平均超额收益柱状图（全中文版）。 |
| `图1_行业样本数量柱状图.png` ~ `图6_看涨溢价十分位数分组折线图.png` | PNG | 6 张高清全中文学术分析可视化图表。 |
| `行业分组描述统计.csv` | CSV | 行业分组及全样本描述性统计汇总表格。 |

---

## 6. 环境配置与运行指南

### 6.1 依赖包安装

本项目基于 Python 3.8+ 开发，所需依赖库记录于 `requirements.txt` 中。请在终端执行：

```bash
pip install -r requirements.txt
```

---

### 6.2 运行顺序

为确保数据依赖正常，请按如下步骤依次运行三个脚本：

```bash
# Step 1: 执行数据清洗与特征工程 (生成 processed_data.csv)
python 01_data_acquisition_firm.py

# Step 2: 执行计量回归与板块异质性分析 (导出 regression_results.csv 及 semicon_vs_traditional_comparison.csv)
python 02_data_analysis_firm.py

# Step 3: 执行学术可视化渲染与描述性统计导出 (生成 图1~图6 全中文图表 及 行业分组描述统计.csv)
python 03_data_visualization_firm.py
```

---

### 6.3 项目完整目录树结构

运行所有脚本后，项目的完整目录结构如下：

```text
.
├── 01_data_acquisition_firm.py                    # 1. 数据清洗与特征工程脚本 (Step 1)
├── 02_data_analysis_firm.py                       # 2. 计量回归与板块分析脚本 (Step 2)
├── 03_data_visualization_firm.py                  # 3. 学术可视化与描述统计脚本 (Step 3)
├── sp_500_analyst_rating_and_price_target_accuracy.csv # 4. 原始输入数据集
├── processed_data.csv                            # 5. 清洗与聚合后数据集
├── requirements.txt                              # 6. Python 依赖库清单
├── README.md                                     # 7. 项目说明文档
└── output_firm/                                   # 📂 结果导出目录
    ├── regression_results.csv                    # 全样本 regression 系数明细表
    ├── semicon_vs_traditional_comparison.csv    # 异质性板块对比汇总表 (含 SE 与置信区间)
    ├── 半导体与传统行业学术拟合对比图.png        # 双板块学术拟合与收益对比图 (中文主图)
    ├── 行业分组描述统计.csv                         # 行业分组描述统计表
    ├── 图1_行业样本数量柱状图.png                  # 图1：样本数量柱状图
    ├── 图2_看涨溢价分布直方图.png                  # 图2：看涨溢价分布直方图
    ├── 图3_多周期收益相关系数热力图.png            # 图3：相关系数热力图
    ├── 图4_全样本二次拟合图.png                    # 图4：全样本二次拟合图
    ├── 图5_行业异质性双面板分析图.png              # 图5：异质性双面板分析图
    └── 图6_看涨溢价十分位数分组折线图.png          # 图6：十分位数分组折线图
```
