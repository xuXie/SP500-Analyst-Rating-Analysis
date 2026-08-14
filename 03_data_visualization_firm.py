"""
【代码功能目录 / Table of Contents】:
----------------------------------------------------------------------------------------
1 环境初始化与全局配置 (Global Visual Configurations)
    ├── 1.1 导出路径配置 (Output Directory Setup)
    ├── 1.2 Seaborn / Matplotlib 字体与样式配置 (Style Setup)
    └── 1.3 核心变量与极值拐点参数定义 (Variables & Inflexion Parameters)
2 图表渲染模块 (Academic Visualization Modules - 调整后的生成顺序)
    ├── 2.1 图1: 半导体、传统行业观测样本数量柱状图 (Fig1_Sample_Count.png)
    ├── 2.2 图2: 半导体、传统行业看涨溢价分布直方图 (Fig2_Dist_Hist.png)
    ├── 2.3 图3: 看涨溢价与多周期超额收益相关系数热力图 (Fig3_Correlation_Heatmap.png)
    ├── 2.4 图4: 分析师看涨预期溢价图 (Fig4_FullSample_U.png)
    ├── 2.5 图5: 半导体与传统行业异质性分析图 (Fig5_Industry_DualPanel.png)
    └── 2.6 图6: 看涨溢价十分位数分组平均超额收益折线图 (Fig6_Quantile_Line.png)
3 描述性统计导出模块 (Descriptive Statistics Export)
    └── 3.1 行业分组及全样本描述统计表格导出 (行业分组描述统计.csv)
4 流程主入口 (Main Execution Pipeline)
----------------------------------------------------------------------------------------
"""

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1 环境初始化与全局配置 (Global Setup)
# ==========================================
INPUT_FILE = "processed_data.csv"
OUTPUT_DIR = "output_firm"

# 核心变量定义
COL_X = "expected_bullishness"
COL_Y = "actual_excess_return_180d"
COL_GROUP = "industry_group"

# 预设拐点数值 (基于 OLS 二次拟合模型计算)
INFLECTION_ALL = 3.87
INFLECTION_SEMI = 0.85
INFLECTION_TRAD = 14.74

# 分组标签定义
LABEL_SEMI = "Semiconductors"
LABEL_TRAD = "Traditional Industry"

def setup_environment():
    """设置 Seaborn 与 Matplotlib 样式和字体"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig_both_locations(fig, filename):
    """保存图表至output_firm"""
    path_output = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path_output, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ==========================================
# 2 图表渲染模块 (Visualizations - 按新顺序)
# ==========================================

def generate_fig1_sample_count(df):
    """图1：半导体、传统行业观测样本数量柱状图"""
    print("  🎨 渲染 图1: 半导体、传统行业观测样本数量柱状图 (Fig1_Sample_Count.png)...")
    fig, ax = plt.subplots(figsize=(7, 4))

    sample_counts = df.groupby(COL_GROUP, observed=False).size()
    sample_groups = [LABEL_SEMI, LABEL_TRAD]
    sample_nums = [sample_counts[LABEL_SEMI], sample_counts[LABEL_TRAD]]

    ax.bar(sample_groups, sample_nums, color=["#0044cc", "#ff7722"])
    ax.set_title("Number of Observations per Industry", fontsize=14)
    ax.set_ylabel("Observation Count")

    for bar in ax.patches:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 100,
            f"{int(h)}",
            ha="center",
            fontsize=9
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig_both_locations(fig, "Fig1_Sample_Count.png")


def generate_fig2_dist_hist(df_plot):
    """图2：半导体、传统行业看涨溢价分布直方图"""
    print("  🎨 渲染 图2: 半导体、传统行业看涨溢价分布直方图 (Fig2_Dist_Hist.png)...")
    fig, ax = plt.subplots(figsize=(10, 6))

    df_hist = df_plot[df_plot[COL_GROUP].isin([LABEL_SEMI, LABEL_TRAD])]
    sns.histplot(data=df_hist, x=COL_X, hue=COL_GROUP, kde=True, alpha=0.5, ax=ax)

    ax.set_title("Distribution of Analyst Bullish Premium", fontsize=14)
    ax.set_xlabel("Bullish Expectation Premium")
    ax.set_ylabel("Frequency")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig_both_locations(fig, "Fig2_Dist_Hist.png")


def generate_fig3_correlation_heatmap(df):
    """图3：看涨溢价与多周期超额收益相关系数热力图"""
    print("  🎨 渲染 图3: 看涨溢价与多周期超额收益相关系数热力图 (Fig3_Correlation_Heatmap.png)...")
    fig, ax = plt.subplots(figsize=(8, 6))

    corr_vars = [
        COL_X,
        "forward_return_30d_pct",
        "forward_return_90d_pct",
        COL_Y,
        "forward_return_365d_pct"
    ]
    corr_df = df[corr_vars].corr()

    sns.heatmap(corr_df, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix of Bullish Premium & Multi-period Returns", fontsize=14)
    ax.set_xticklabels(["Premium", "30d Ret", "90d Ret", "180d Ret", "365d Ret"], fontsize=9)
    ax.set_yticklabels(["Premium", "30d Ret", "90d Ret", "180d Ret", "365d Ret"], fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig_both_locations(fig, "Fig3_Correlation_Heatmap.png")


def generate_fig4_full_sample_u(df, df_plot):
    """图4：分析师看涨预期溢价图 (全样本 U型二次拟合图)"""
    print("  🎨 渲染 图4: 分析师看涨预期溢价图 (Fig4_FullSample_U.png)...")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(data=df_plot, x=COL_X, y=COL_Y, alpha=0.3, ax=ax)

    # 拟合二次多项式 (使用全样本数据)
    z = np.polyfit(df[COL_X], df[COL_Y], 2)
    poly_func = np.poly1d(z)
    x_line = np.linspace(df[COL_X].min(), df[COL_X].max(), 200)

    ax.plot(x_line, poly_func(x_line), color="#2255bb", lw=3, label="Quadratic U-curve Fit")
    ax.axvline(x=INFLECTION_ALL, color="red", linestyle="--", label=f"Inflection Point: {INFLECTION_ALL}")

    ax.set_title("U-shaped Relationship: Bullish Expectation vs 180d Excess Return (Full Sample)", fontsize=14)
    ax.set_xlabel("Analyst Bullish Expectation Premium")
    ax.set_ylabel("180-day Excess Return")
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig_both_locations(fig, "Fig4_FullSample_U.png")


def generate_fig5_industry_dual_panel(df, df_plot):
    """图5：半导体与传统行业异质性分析图 (双面板图)"""
    print("  🎨 渲染 图5: 半导体与传统行业异质性分析图 (Fig5_Industry_DualPanel.png)...")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.32})

    df_semi_raw = df[df[COL_GROUP] == LABEL_SEMI]
    df_trad_raw = df[df[COL_GROUP] == LABEL_TRAD]
    df_semi_plot = df_plot[df_plot[COL_GROUP] == LABEL_SEMI]
    df_trad_plot = df_plot[df_plot[COL_GROUP] == LABEL_TRAD]

    x_semi_min, x_semi_max = df_semi_raw[COL_X].min(), df_semi_raw[COL_X].max()
    x_trad_min, x_trad_max = df_trad_raw[COL_X].min(), df_trad_raw[COL_X].max()

    x_range_semi = np.linspace(x_semi_min, x_semi_max, 300)
    x_range_trad = np.linspace(x_trad_min, x_trad_max, 300)

    # 左面板散点
    ax_left.scatter(df_trad_plot[COL_X], df_trad_plot[COL_Y], color="#949494", alpha=0.22, s=8, label="Traditional Industry")
    ax_left.scatter(df_semi_plot[COL_X], df_semi_plot[COL_Y], color="#d48b59", alpha=0.22, s=8, label="Semiconductors")

    # 拟合曲线
    z_semi = np.polyfit(df_semi_raw[COL_X], df_semi_raw[COL_Y], 2)
    poly_semi = np.poly1d(z_semi)
    ax_left.plot(x_range_semi, poly_semi(x_range_semi), color="#c26000", lw=2, label="Semiconductors Fit")

    z_trad = np.polyfit(df_trad_raw[COL_X], df_trad_raw[COL_Y], 2)
    poly_trad = np.poly1d(z_trad)
    ax_left.plot(x_range_trad, poly_trad(x_range_trad), color="#003370", lw=2, label="Traditional Industry Fit")

    ax_left.set_ylim(-40, 340)
    ax_left.set_yticks([0, 100, 200, 300])
    ax_left.set_title("U-Curve: Semiconductors vs. Traditional Industry", fontsize=9)
    ax_left.set_xlabel("Expected Bullishness (Delta Target Price %)", fontsize=8.5)
    ax_left.set_ylabel("Actual Excess Return 180d (%)", fontsize=8.5)
    ax_left.legend(loc="upper right", fontsize=7, frameon=True)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.grid(True, alpha=0.35)

    # 右面板柱状图
    df_other = df[~df[COL_GROUP].isin([LABEL_SEMI, LABEL_TRAD])]
    mean_other = df_other[COL_Y].mean()
    mean_semi = df.loc[df[COL_GROUP] == LABEL_SEMI, COL_Y].mean()
    mean_trad = df.loc[df[COL_GROUP] == LABEL_TRAD, COL_Y].mean()

    show_groups = ["Other/General", "Semiconductors", "Traditional Industry"]
    avg_values = [mean_other, mean_semi, mean_trad]

    bars = ax_right.bar(show_groups, avg_values, color=["#c26000", "#6b62a1", "#003370"])
    ax_right.set_title("Mean 180d Excess Return by Industry Group", fontsize=9)
    ax_right.set_ylabel("Mean Excess Return 180d (%)", fontsize=8.5)

    for bar in bars:
        height = bar.get_height()
        disp_h = 0.00 if abs(height) < 1e-4 else height
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.2,
            f"{disp_h:.2f}",
            ha="center",
            fontsize=7.5
        )

    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.grid(True, alpha=0.35)

    save_fig_both_locations(fig, "Fig5_Industry_DualPanel.png")


def generate_fig6_quantile_line(df):
    """图6：看涨溢价十分位数分组平均超额收益折线图"""
    print("  🎨 渲染 图6: 看涨溢价十分位数分组平均超额收益折线图 (Fig6_Quantile_Line.png)...")
    fig, ax = plt.subplots(figsize=(9, 5))

    df_copy = df.copy()
    df_copy["premium_bin"] = pd.qcut(df_copy[COL_X], q=10, duplicates="drop")
    bin_result = df_copy.groupby("premium_bin", observed=False)[COL_Y].mean().reset_index()

    ax.plot(
        range(len(bin_result)),
        bin_result[COL_Y],
        marker="o",
        c="#2255bb",
        linewidth=2
    )
    ax.set_title("Average Excess Return by Premium Quantile", fontsize=14)
    ax.set_xlabel("Premium Quantile (10 Groups)")
    ax.set_ylabel("Mean 180-day Excess Return")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig_both_locations(fig, "Fig6_Quantile_Line.png")

# ==========================================
# 3 描述性统计表格导出模块 (Export Table)
# ==========================================
def export_descriptive_stats(df):
    """分组描述性统计表格计算与导出"""
    print("\n📊 导出行业分组与全样本描述性统计表格...")

    desc_table = df.groupby(COL_GROUP, observed=False)[[COL_X, COL_Y]].describe()
    desc_table.columns = [f"{var}_{stat}" for var, stat in desc_table.columns]
    desc_table = desc_table.reset_index()

    # 计算全样本统计量
    total_stats = df[[COL_X, COL_Y]].describe().T
    total_row = {COL_GROUP: "Full Sample"}
    for var in [COL_X, COL_Y]:
        total_row[f"{var}_count"] = int(total_stats.loc[var, "count"])
        total_row[f"{var}_mean"] = total_stats.loc[var, "mean"]
        total_row[f"{var}_std"] = total_stats.loc[var, "std"]
        total_row[f"{var}_min"] = total_stats.loc[var, "min"]
        total_row[f"{var}_25%"] = total_stats.loc[var, "25%"]
        total_row[f"{var}_50%"] = total_stats.loc[var, "50%"]
        total_row[f"{var}_75%"] = total_stats.loc[var, "75%"]
        total_row[f"{var}_max"] = total_stats.loc[var, "max"]

    total_df = pd.DataFrame([total_row])
    desc_table = pd.concat([total_df, desc_table], ignore_index=True)

    rename_map = {
        "industry_group": "行业分组",
        "expected_bullishness_count": "看涨溢价样本量",
        "expected_bullishness_mean": "看涨溢价均值",
        "expected_bullishness_std": "看涨溢价标准差",
        "expected_bullishness_min": "看涨溢价最小值",
        "expected_bullishness_25%": "看涨溢价25分位数",
        "expected_bullishness_50%": "看涨溢价中位数",
        "expected_bullishness_75%": "看涨溢价75分位数",
        "expected_bullishness_max": "看涨溢价最大值",
        "actual_excess_return_180d_count": "180日超额收益样本量",
        "actual_excess_return_180d_mean": "180日超额收益均值",
        "actual_excess_return_180d_std": "180日超额收益标准差",
        "actual_excess_return_180d_min": "180日超额收益最小值",
        "actual_excess_return_180d_25%": "180日超额收益25分位数",
        "actual_excess_return_180d_50%": "180日超额收益中位数",
        "actual_excess_return_180d_75%": "180日超额收益75分位数",
        "actual_excess_return_180d_max": "180日超额收益最大值",
    }

    desc_table = desc_table.rename(columns=rename_map)

    path_root = "行业分组描述统计.csv"
    path_output = os.path.join(OUTPUT_DIR, "行业分组描述统计.csv")

    desc_table.to_csv(path_root, index=False, encoding="utf-8-sig")
    desc_table.to_csv(path_output, index=False, encoding="utf-8-sig")

    print(f"✅ 统计表格导出成功: {path_root} (同时保存副本至 {path_output})")

# ==========================================
# 4 流程主入口 (Main Execution)
# ==========================================
def main():
    setup_environment()

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：未找到输入文件 {INPUT_FILE}，请先运行 01_data_acquisition_firm.py 生成清洗后数据。")
        return

    print("🚀 [3/3] 开始按新顺序执行学术可视化图表渲染与统计表格导出...")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    print(f"📊 读取数据集成功，总记录数: {len(df)} 条")

    # 绘图缩尾数据（1%-99%分位数过滤，仅用于散点/直方图可视化展现，不修改统计计算）
    q_low = df[COL_Y].quantile(0.01)
    q_high = df[COL_Y].quantile(0.99)
    df_plot = df[(df[COL_Y] >= q_low) & (df[COL_Y] <= q_high)].copy()

    # 渲染全部 6 个图表 (按新顺序)
    generate_fig1_sample_count(df)
    generate_fig2_dist_hist(df_plot)
    generate_fig3_correlation_heatmap(df)
    generate_fig4_full_sample_u(df, df_plot)
    generate_fig5_industry_dual_panel(df, df_plot)
    generate_fig6_quantile_line(df)

    # 导出描述统计表格
    export_descriptive_stats(df)

    print("\n🎉 [完成] 所有可视化学术图表 (Fig1~Fig6) 及描述统计表格按新顺序处理完毕！")

if __name__ == "__main__":
    main()
