"""
========================================================================================
项目名称: Team16_标普500股票的分析师评级与目标价准确性研究  (Step 2)
          倒 U 型非线性假设检验与“半导体 vs 传统行业”异质性对比
========================================================================================

【代码功能目录 / Table of Contents】:
----------------------------------------------------------------------------------------
1 全局环境与绘图配置 (Global Canvas & Export Setup)
    ├── 1.1 导出路径配置 (Output Directory Setup)
    └── 1.2 Seaborn/Matplotlib 字体与白格样式 (Style Setup)
2 全样本二次非线性 OLS 回归 (Base Quadratic Regression Model)
    ├── 2.1 估计参数: Y = β0 + β1*X + β2*X^2 + Year_FE (HC1 Robust Std Error)
    ├── 2.2 倒 U 拐点公式推导: Turning Point = -β1 / (2 * β2)
    └── 2.3 结果导出: regression_results.csv
3 板块异质性分组回归 (Industry Heterogeneity Analysis)
    ├── 3.1 半导体板块回归 (Semiconductors Model & Turning Point)
    ├── 3.2 传统行业回归 (Traditional Industry Model & Turning Point)
    └── 3.3 对比汇总导出: semicon_vs_traditional_comparison.csv
4 学术拟合图表渲染 (Academic Visualizations)
    ├── 4.1 子图 1: 双板块倒 U 型二次拟合曲线叠加散点对比图
    └── 4.2 子图 2: 板块平均超额收益柱状图
5 流程结束与报告汇总 (Execution Summary)
----------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1 全局环境与绘图配置 (Setup)
# ==========================================
# 1.1 导出路径
OUTPUT_DIR = "output_firm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1.2 设置学术论文风格图表 (Whitegrid) 与中文字体设置
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Songti SC', 'STHeiti', 'PingFang SC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

INPUT_FILE = "processed_data.csv"

def run_analysis():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：未找到数据文件 {INPUT_FILE}，请先运行 01_data_acquisition.py")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"📖 [2/2] 读取清洗后数据集，总样本量: {len(df)} 条\n")

# ==========================================
# 2 全样本二次非线性 OLS 回归 (Base Model)
# ==========================================
    print("="*60)
    print("📌 模块 2: 全样本非线性 OLS 回归 (HC1 稳健标准误 + 年份固定效应)")
    print("="*60)
    
# 2.1 回归模型估计参数
    formula_base = 'actual_excess_return_180d ~ expected_bullishness + expected_bullishness_sq + C(firm) + C(event_year)'
    model_base = smf.ols(formula=formula_base, data=df).fit(cov_type='HC1')
    
    print(model_base.summary().tables[1])

# 2.2 倒 U 型极值拐点 (Turning Point) 导出与 Lind & Mehlum 严格检验
    beta1 = model_base.params['expected_bullishness']
    beta2 = model_base.params['expected_bullishness_sq']
    p_val_sq = model_base.pvalues['expected_bullishness_sq']

    x_min = df['expected_bullishness'].min()
    x_max = df['expected_bullishness'].max()

    print("\n" + "-"*50)
    print("🔍 [全样本倒 U 型极值拐点严谨检验]")
    print(f"• 样本自变量范围 (X_min ~ X_max): [{x_min:.2%}, {x_max:.2%}]")
    print(f"• 一次项系数 (β1): {beta1:.4f}")
    print(f"• 二次项系数 (β2): {beta2:.4f} (p-value: {p_val_sq:.4e})")

    if beta2 < 0 and p_val_sq < 0.05:
        turning_point = -beta1 / (2 * beta2)
        
        # 2.2.1 验证拐点是否在样本区间内
        is_inside = x_min < turning_point < x_max
        print(f"• 计算拐点 (Turning Point): {turning_point:.2%}")
        
        # 2.2.2 检验区间两端的斜率符号 (Slope at bounds)
        slope_min = beta1 + 2 * beta2 * x_min
        slope_max = beta1 + 2 * beta2 * x_max
        
        print(f"• 左端点 ({x_min:.2%}) 处斜率: {slope_min:.4f}")
        print(f"• 右端点 ({x_max:.2%}) 处斜率: {slope_max:.4f}")

        if is_inside and slope_min > 0 and slope_max < 0:
            print("✅ 【通过检验】: 存在显著且严谨的倒 U 型非线性关系！")
        else:
            print("⚠️ 【警告】: 二次项虽显著，但拐点超出实际样本区间，本质仅为单调凹函数/假倒 U 形。")
    else:
        print("❌ 二次项不显著或系数非负，未发现倒 U 型关系。")
    print("-" * 50)

# 2.3 导出全样本回归系数明细数据表 (包含 Beta, SE, t值, p值)
    results_df = pd.DataFrame({
        'Coefficient': model_base.params,
        'Std_Error': model_base.bse,
        't_stat': model_base.tvalues,
        'p_value': model_base.pvalues
    })
    results_df.to_csv(os.path.join(OUTPUT_DIR, "regression_results.csv"), encoding='utf-8-sig')
    
# ==========================================
# 3 板块异质性分组回归 (Heterogeneity Analysis)
# ==========================================
    print("\n" + "="*60)
    print("📌 模块 3: 板块异质性分组回归 (半导体 vs. 传统行业)")
    print("="*60)

# 3.1 半导体板块回归 (验证高成长板块的容忍度拐点)
    df_semicon = df[df['industry_group'] == 'Semiconductors'].copy()
    model_semicon = smf.ols(formula=formula_base, data=df_semicon).fit(cov_type='HC1')
    b1_semi = model_semicon.params['expected_bullishness']
    b2_semi = model_semicon.params['expected_bullishness_sq']
    p2_semi = model_semicon.pvalues['expected_bullishness_sq']
    tp_semi = -b1_semi / (2 * b2_semi) if b2_semi < 0 else None

# 3.2 传统行业回归 (验证稳健型板块的容忍度拐点)
    df_trad = df[df['industry_group'] == 'Traditional Industry'].copy()
    model_trad = smf.ols(formula=formula_base, data=df_trad).fit(cov_type='HC1')
    b1_trad = model_trad.params['expected_bullishness']
    b2_trad = model_trad.params['expected_bullishness_sq']
    p2_trad = model_trad.pvalues['expected_bullishness_sq']
    tp_trad = -b1_trad / (2 * b2_trad) if b2_trad < 0 else None

# 3.3 导出板块对比汇总表 (汇总全样本、半导体与传统行业的系数与拐点)
    ci_base = model_base.conf_int(alpha=0.05)
    ci_semi = model_semicon.conf_int(alpha=0.05)
    ci_trad = model_trad.conf_int(alpha=0.05)

    compare_summary = pd.DataFrame({
        'Industry_Group': ['All Sample', 'Semiconductors', 'Traditional Industry'],
        'Sample_Count': [len(df), len(df_semicon), len(df_trad)],
        
        # 一次项 X 的指标
        'Beta1_Linear': [beta1, b1_semi, b1_trad],
        'Beta1_SE': [model_base.bse['expected_bullishness'], model_semicon.bse['expected_bullishness'], model_trad.bse['expected_bullishness']],
        'Beta1_CI_Lower': [ci_base.loc['expected_bullishness', 0], ci_semi.loc['expected_bullishness', 0], ci_trad.loc['expected_bullishness', 0]],
        'Beta1_CI_Upper': [ci_base.loc['expected_bullishness', 1], ci_semi.loc['expected_bullishness', 1], ci_trad.loc['expected_bullishness', 1]],
        
        # 二次项 X^2 的指标
        'Beta2_Squared': [beta2, b2_semi, b2_trad],
        'Beta2_SE': [model_base.bse['expected_bullishness_sq'], model_semicon.bse['expected_bullishness_sq'], model_trad.bse['expected_bullishness_sq']],
        'Beta2_CI_Lower': [ci_base.loc['expected_bullishness_sq', 0], ci_semi.loc['expected_bullishness_sq', 0], ci_trad.loc['expected_bullishness_sq', 0]],
        'Beta2_CI_Upper': [ci_base.loc['expected_bullishness_sq', 1], ci_semi.loc['expected_bullishness_sq', 1], ci_trad.loc['expected_bullishness_sq', 1]],
        
        'Beta2_PValue': [p_val_sq, p2_semi, p2_trad],
        'Turning_Point_Pct': [
            -beta1 / (2 * beta2) if beta2 < 0 else None,
            tp_semi,
            tp_trad
        ]
    })
    
    comp_path = os.path.join(OUTPUT_DIR, "semicon_vs_traditional_comparison.csv")
    compare_summary.to_csv(comp_path, index=False, encoding='utf-8-sig')
    print(f"📄 板块对比结果表(含SE与置信区间)已成功保存至: {comp_path}")

# ==========================================
# 4 学术拟合图表渲染 (Visualizations)
# ==========================================
    print("🎨 正在渲染 4 学术拟合对比图表 (中文)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 4.1 子图 1: 双板块二次拟合曲线对比
    df_plot = df[df['industry_group'].isin(['Semiconductors', 'Traditional Industry'])].copy()
    group_map = {'Semiconductors': '半导体板块', 'Traditional Industry': '传统行业'}
    df_plot['industry_group_cn'] = df_plot['industry_group'].map(group_map)

    sns.scatterplot(
        data=df_plot.sample(n=min(3000, len(df_plot)), random_state=42),
        x='expected_bullishness',
        y='actual_excess_return_180d',
        hue='industry_group_cn',
        alpha=0.25,
        palette={'半导体板块': '#d95f02', '传统行业': '#2b5c8f'},
        ax=axes[0]
    )
    sns.regplot(
        data=df_semicon, x='expected_bullishness', y='actual_excess_return_180d',
        order=2, scatter=False, color='#d95f02', label='半导体板块二次拟合', ax=axes[0]
    )
    sns.regplot(
        data=df_trad, x='expected_bullishness', y='actual_excess_return_180d',
        order=2, scatter=False, color='#2b5c8f', label='传统行业二次拟合', ax=axes[0]
    )
    axes[0].set_title('半导体与传统行业二次拟合曲线对比', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('预期看涨程度 (目标价变更比例)', fontsize=10)
    axes[0].set_ylabel('实际 180 天超额收益率 (%)', fontsize=10)
    axes[0].legend(fontsize=9)

# 4.2 子图 2: 不同板块平均超额收益柱状图
    group_means = df.groupby('industry_group')['actual_excess_return_180d'].mean().reset_index()
    group_means['industry_group_cn'] = group_means['industry_group'].map({
        'Semiconductors': '半导体板块',
        'Traditional Industry': '传统行业',
        'Other/General': '通用/其他'
    })
    sns.barplot(
        data=group_means,
        x='industry_group_cn',
        y='actual_excess_return_180d',
        hue='industry_group_cn',
        palette={'半导体板块': '#d95f02', '传统行业': '#2b5c8f', '通用/其他': '#7570b3'},
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title('各行业分组 180 天平均超额收益率', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('行业分组', fontsize=10)
    axes[1].set_ylabel('平均 180 天超额收益率 (%)', fontsize=10)

    for bar in axes[1].patches:
        h = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.1,
            f"{h:.2f}%",
            ha="center",
            fontsize=9
        )

    plt.tight_layout()
    chart_cn = os.path.join(OUTPUT_DIR, "半导体与传统行业学术拟合对比图.png")
    plt.savefig(chart_cn, dpi=300)
    plt.close(fig)
    print(f"🖼️ 中文学术对比图已成功导出至: {chart_cn}")
# ==========================================
# 5 流程结束与报告汇总 (Execution Summary)
# ==========================================
    print("\n🎉 实证分析流程全线完成！")

if __name__ == "__main__":
    run_analysis()