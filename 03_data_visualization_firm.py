"""
========================================================================================
项目名称: Team16_标普500股票的分析师评级与目标价准确性研究 (Step 3)
          学术可视化图表绘制与行业分组描述性统计导出 (全中文版)
========================================================================================

【代码功能目录 / Table of Contents】:
----------------------------------------------------------------------------------------
1 环境初始化与全局配置 (Global Visual Configurations)
    ├── 1.1 导出路径配置 (Output Directory Setup)
    ├── 1.2 Seaborn / Matplotlib 中文字体与样式配置 (Style Setup)
    └── 1.3 核心变量与标签定义 (Variables & Label Definitions)
2 图表渲染模块 (Academic Visualization Modules - 全中文学术绘制)
    ├── 2.1 图1: 各行业板块观测样本数量柱状图 (图1_行业样本数量柱状图.png)
    ├── 2.2 图2: 看涨溢价分布直方图 (图2_看涨溢价分布直方图.png)
    ├── 2.3 图3: 多周期收益相关系数热力图 (图3_多周期收益相关系数热力图.png)
    ├── 2.4 图4: 全样本二次拟合关系图 (图4_全样本二次拟合图.png)
    ├── 2.5 图5: 行业异质性分析双面板图 (图5_行业异质性双面板分析图.png)
    └── 2.6 图6: 看涨溢价十分位数分组折线图 (图6_看涨溢价十分位数分组折线图.png)
3 描述性统计导出模块 (Descriptive Statistics Export)
    └── 3.1 行业分组及全样本描述统计表格导出 (行业分组描述统计.csv)
4 流程主入口 (Main Execution Pipeline)
----------------------------------------------------------------------------------------
"""

import os
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

# 分组标签定义
LABEL_SEMI = "Semiconductors"
LABEL_TRAD = "Traditional Industry"

LABEL_SEMI_CN = "半导体板块"
LABEL_TRAD_CN = "传统行业"

def setup_environment():
    """设置 Seaborn 与 Matplotlib 中文字体样式 (注意：set_theme 必须在 plt.rcParams 设置之前调用)"""
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "PingFang SC", "Arial Unicode MS", "SimHei"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fig, filename_cn):
    """保存中文图表至 output_firm 目录"""
    path_cn = os.path.join(OUTPUT_DIR, filename_cn)
    fig.savefig(path_cn, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ==========================================
# 2 图表渲染模块 (Visualizations - 全中文学术绘制)
# ==========================================

def generate_fig1_sample_count(df):
    """图1：半导体、传统行业观测样本数量柱状图"""
    print("  🎨 渲染 图1: 各行业板块观测样本数量柱状图 (图1_行业样本数量柱状图.png)...")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sample_counts = df.groupby(COL_GROUP, observed=False).size()
    sample_groups = [LABEL_SEMI_CN, LABEL_TRAD_CN]
    sample_nums = [sample_counts[LABEL_SEMI], sample_counts[LABEL_TRAD]]

    bars = ax.bar(sample_groups, sample_nums, color=["#0044cc", "#ff7722"], width=0.45)
    ax.set_title("各行业板块观测样本数量分布", fontsize=13, fontweight='bold')
    ax.set_ylabel("观测样本数量 (条)", fontsize=10)
    ax.set_xlabel("行业板块", fontsize=10)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 120,
            f"{int(h):,} 条",
            ha="center",
            fontsize=9.5,
            fontweight='bold'
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig(fig, "图1_行业样本数量柱状图.png")


def generate_fig2_dist_hist(df_plot):
    """图2：半导体、传统行业看涨溢价分布直方图"""
    print("  🎨 渲染 图2: 看涨溢价分布直方图 (图2_看涨溢价分布直方图.png)...")
    fig, ax = plt.subplots(figsize=(10, 6))

    df_hist = df_plot[df_plot[COL_GROUP].isin([LABEL_SEMI, LABEL_TRAD])].copy()
    df_hist['industry_group_cn'] = df_hist[COL_GROUP].map({
        LABEL_SEMI: LABEL_SEMI_CN,
        LABEL_TRAD: LABEL_TRAD_CN
    })

    sns.histplot(
        data=df_hist,
        x=COL_X,
        hue='industry_group_cn',
        kde=True,
        alpha=0.45,
        palette={LABEL_SEMI_CN: "#0044cc", LABEL_TRAD_CN: "#ff7722"},
        ax=ax
    )

    ax.set_title("分析师预期看涨溢价分布直方图与密度估计", fontsize=13, fontweight='bold')
    ax.set_xlabel("分析师预期看涨溢价 (Expected Bullishness)", fontsize=10)
    ax.set_ylabel("样本频数 (Frequency)", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig(fig, "图2_看涨溢价分布直方图.png")


def generate_fig3_correlation_heatmap(df):
    """图3：看涨溢价与多周期超额收益相关系数热力图"""
    print("  🎨 渲染 图3: 看涨溢价与多周期超额收益相关系数热力图 (图3_多周期收益相关系数热力图.png)...")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    possible_vars = [
        COL_X,
        "forward_return_30d_pct",
        "forward_return_90d_pct",
        COL_Y,
        "forward_return_365d_pct"
    ]
    corr_vars = [var for var in possible_vars if var in df.columns]
    corr_df = df[corr_vars].corr()

    labels_map = {
        COL_X: "看涨溢价",
        "forward_return_30d_pct": "30日收益率",
        "forward_return_90d_pct": "90日收益率",
        COL_Y: "180日超额收益",
        "forward_return_365d_pct": "365日收益率"
    }
    display_labels = [labels_map.get(col, col) for col in corr_vars]

    sns.heatmap(
        corr_df,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={'label': '相关系数 Correlation'},
        ax=ax
    )
    ax.set_title("看涨溢价与多周期收益率相关系数矩阵热力图", fontsize=13, fontweight='bold')
    ax.set_xticklabels(display_labels, fontsize=9.5)
    ax.set_yticklabels(display_labels, fontsize=9.5, rotation=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig(fig, "图3_多周期收益相关系数热力图.png")


def generate_fig4_full_sample_u(df, df_plot):
    """图4：全样本分析师看涨预期溢价与超额收益拟合图"""
    print("  🎨 渲染 图4: 全样本二次拟合关系图 (图4_全样本二次拟合图.png)...")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(data=df_plot, x=COL_X, y=COL_Y, alpha=0.25, color="#2b5c8f", s=15, ax=ax)

    # 动态加载 02 脚本中估算的回归极值点结果 (若存在)
    comp_file = os.path.join(OUTPUT_DIR, "semicon_vs_traditional_comparison.csv")
    turning_point_val = None
    if os.path.exists(comp_file):
        try:
            comp_df = pd.read_csv(comp_file)
            all_row = comp_df[comp_df['Industry_Group'] == 'All Sample']
            if not all_row.empty and 'Turning_Point_Pct' in all_row.columns:
                tp = all_row['Turning_Point_Pct'].values[0]
                if not pd.isna(tp) and str(tp).strip() != '':
                    turning_point_val = float(tp)
        except Exception:
            pass

    # 二次多项式拟合曲线
    z = np.polyfit(df[COL_X], df[COL_Y], 2)
    poly_func = np.poly1d(z)
    x_line = np.linspace(df[COL_X].min(), df[COL_X].max(), 200)

    ax.plot(x_line, poly_func(x_line), color="#d95f02", lw=3, label="全样本二次拟合曲线")

    if turning_point_val is not None:
        ax.axvline(
            x=turning_point_val,
            color="red",
            linestyle="--",
            lw=1.8,
            label=f"固定效应模型极值拐点: {turning_point_val:.2%}"
        )

    ax.set_title("分析师看涨预期溢价与 180 天超额收益关系 (全样本)", fontsize=13, fontweight='bold')
    ax.set_xlabel("分析师预期看涨溢价 (Expected Bullishness)", fontsize=10)
    ax.set_ylabel("180 天实际超额收益率 (%)", fontsize=10)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig(fig, "图4_全样本二次拟合图.png")


def generate_fig5_industry_dual_panel(df, df_plot):
    """图5：半导体与传统行业异质性分析双面板图"""
    print("  🎨 渲染 图5: 行业异质性分析双面板图 (图5_行业异质性双面板分析图.png)...")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.32})

    df_semi_raw = df[df[COL_GROUP] == LABEL_SEMI]
    df_trad_raw = df[df[COL_GROUP] == LABEL_TRAD]
    df_semi_plot = df_plot[df_plot[COL_GROUP] == LABEL_SEMI]
    df_trad_plot = df_plot[df_plot[COL_GROUP] == LABEL_TRAD]

    x_semi_min, x_semi_max = df_semi_raw[COL_X].min(), df_semi_raw[COL_X].max()
    x_trad_min, x_trad_max = df_trad_raw[COL_X].min(), df_trad_raw[COL_X].max()

    x_range_semi = np.linspace(x_semi_min, x_semi_max, 300)
    x_range_trad = np.linspace(x_trad_min, x_trad_max, 300)

    # 左面板: 散点与二次拟合对比
    ax_left.scatter(df_trad_plot[COL_X], df_trad_plot[COL_Y], color="#949494", alpha=0.20, s=8, label="传统行业样本")
    ax_left.scatter(df_semi_plot[COL_X], df_semi_plot[COL_Y], color="#d48b59", alpha=0.20, s=8, label="半导体板块样本")

    z_semi = np.polyfit(df_semi_raw[COL_X], df_semi_raw[COL_Y], 2)
    poly_semi = np.poly1d(z_semi)
    ax_left.plot(x_range_semi, poly_semi(x_range_semi), color="#c26000", lw=2.5, label="半导体板块二次拟合")

    z_trad = np.polyfit(df_trad_raw[COL_X], df_trad_raw[COL_Y], 2)
    poly_trad = np.poly1d(z_trad)
    ax_left.plot(x_range_trad, poly_trad(x_range_trad), color="#003370", lw=2.5, label="传统行业二次拟合")

    ax_left.set_ylim(-40, 340)
    ax_left.set_yticks([0, 100, 200, 300])
    ax_left.set_title("拟合曲线对比: 半导体板块 vs. 传统行业", fontsize=10.5, fontweight='bold')
    ax_left.set_xlabel("预期看涨程度 (目标价变更比例)", fontsize=9)
    ax_left.set_ylabel("实际 180 天超额收益率 (%)", fontsize=9)
    ax_left.legend(loc="upper right", fontsize=8, frameon=True)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.grid(True, alpha=0.35)

    # 右面板: 行业分组平均收益率柱状图
    df_other = df[~df[COL_GROUP].isin([LABEL_SEMI, LABEL_TRAD])]
    mean_other = df_other[COL_Y].mean()
    mean_semi = df.loc[df[COL_GROUP] == LABEL_SEMI, COL_Y].mean()
    mean_trad = df.loc[df[COL_GROUP] == LABEL_TRAD, COL_Y].mean()

    show_groups = ["通用/其他", "半导体板块", "传统行业"]
    avg_values = [mean_other, mean_semi, mean_trad]

    bars = ax_right.bar(show_groups, avg_values, color=["#c26000", "#6b62a1", "#003370"], width=0.45)
    ax_right.set_title("各行业分组平均 180 天超额收益率", fontsize=10.5, fontweight='bold')
    ax_right.set_ylabel("平均 180 天超额收益率 (%)", fontsize=9)

    for bar in bars:
        height = bar.get_height()
        disp_h = 0.00 if abs(height) < 1e-4 else height
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.2,
            f"{disp_h:.2f}%",
            ha="center",
            fontsize=8.5,
            fontweight='bold'
        )

    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.grid(True, alpha=0.35)

    save_fig(fig, "图5_行业异质性双面板分析图.png")


def generate_fig6_quantile_line(df):
    """图6：看涨溢价十分位数分组平均超额收益折线图"""
    print("  🎨 渲染 图6: 看涨溢价十分位数分组折线图 (图6_看涨溢价十分位数分组折线图.png)...")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    df_copy = df.copy()
    df_copy["premium_bin"] = pd.qcut(df_copy[COL_X], q=10, duplicates="drop")
    bin_result = df_copy.groupby("premium_bin", observed=False)[COL_Y].mean().reset_index()

    ax.plot(
        range(1, len(bin_result) + 1),
        bin_result[COL_Y],
        marker="o",
        markersize=6,
        c="#2255bb",
        linewidth=2.2,
        label="分位数平均收益"
    )

    for x_val, y_val in zip(range(1, len(bin_result) + 1), bin_result[COL_Y]):
        ax.text(x_val, y_val + 0.15, f"{y_val:.2f}%", ha="center", fontsize=8.5)

    ax.set_title("按分析师看涨溢价十分位数分组的 180 天平均超额收益率", fontsize=12.5, fontweight='bold')
    ax.set_xlabel("看涨溢价十分位数分组 (第 1 组到第 10 组)", fontsize=9.5)
    ax.set_ylabel("平均 180 天超额收益率 (%)", fontsize=9.5)
    ax.set_xticks(range(1, len(bin_result) + 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig(fig, "图6_看涨溢价十分位数分组折线图.png")

# ==========================================
# 3 描述性统计表格导出模块 (Export Table)
# ==========================================
def export_descriptive_stats(df):
    """分组描述性统计表格计算与导出"""
    print("\n📊 导出行业分组与全样本描述性统计表格...")

    desc_table = df.groupby(COL_GROUP, observed=False)[[COL_X, COL_Y]].describe()
    desc_table.columns = [f"{var}_{stat}" for var, stat in desc_table.columns]
    desc_table = desc_table.reset_index()

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

    path_output = os.path.join(OUTPUT_DIR, "行业分组描述统计.csv")
    desc_table.to_csv(path_output, index=False, encoding="utf-8-sig")

    print(f"✅ 统计表格导出成功: {path_output}")

# ==========================================
# 4 流程主入口 (Main Execution)
# ==========================================
def main():
    setup_environment()

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：未找到输入文件 {INPUT_FILE}，请先运行 01_data_acquisition_firm.py 生成清洗后数据。")
        return

    print("🚀 [3/3] 开始按全中文规范渲染学术可视化图表与统计表格导出...")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    print(f"📊 读取数据集成功，总记录数: {len(df)} 条")

    # 绘图缩尾数据 (1%-99%分位数过滤)
    q_low = df[COL_Y].quantile(0.01)
    q_high = df[COL_Y].quantile(0.99)
    df_plot = df[(df[COL_Y] >= q_low) & (df[COL_Y] <= q_high)].copy()

    # 渲染全部 6 个图表 (仅生成中文图片)
    generate_fig1_sample_count(df)
    generate_fig2_dist_hist(df_plot)
    generate_fig3_correlation_heatmap(df)
    generate_fig4_full_sample_u(df, df_plot)
    generate_fig5_industry_dual_panel(df, df_plot)
    generate_fig6_quantile_line(df)

    # 导出描述统计表格
    export_descriptive_stats(df)

    print("\n🎉 [完成] 所有可视化学术图表 (图1~图6 中文版) 及描述统计表格全量处理完毕！")

if __name__ == "__main__":
    main()
