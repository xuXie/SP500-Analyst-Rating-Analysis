"""
【代码功能目录 / Table of Contents】:
----------------------------------------------------------------------------------------
1 环境初始化与配置 (Global Configurations & Ticker Pools)
    ├── 1.1 极值截尾参数设置 (LOWER_QUANTILE / UPPER_QUANTILE)
    ├── 1.2 半导体板块股票池 (SEMICONDUCTOR_TICKERS)
    └── 1.3 传统价值板块股票池 (TRADITIONAL_TICKERS)
2 基础数据清洗与截断控制 (Data Cleaning & Truncation Control)
    ├── 2.1 剔除关键列缺失值 (Ticker/Event Date/Firm)
    ├── 2.2 剔除 180 天时间窗口截断样本 (Forward Return 180d Null Check)
    └── 2.3 剔除非法目标价 (Target Price > 0 Check)
3 核心变量计算 (Feature Engineering - Aligned with Thesis Formula)
    ├── 3.1 自变量: 预期看涨程度 Expected Bullishness = (Price_Target - Current_Price) / Current_Price
    ├── 3.2 离群值处理: 1% 截尾机制 (Trimming)
    ├── 3.3 非线性二次项: Expected Bullishness Squared (X^2)
    └── 3.4 因变量: 实际超额收益率 Actual Excess Return 180d = 180d Return - Market Mean
4 板块与行业划分 (Industry Group Categorization)
    └── 4.1 精准映射: Semiconductors vs. Traditional Industry vs. Other
5 结果导出与结构校验 (Data Export & Verification)
----------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import os

# ==========================================
# 1 环境初始化与配置 (Configurations & Pools)
# ==========================================
# 校验与读取文件路径配置
INPUT_FILE = "sp_500_analyst_rating_and_price_target_accuracy.csv"
OUTPUT_FILE = "processed_data.csv"

# 1.1 极值处理设置双端 1% 截尾 (Trimming)，用于消除分析师录入极端错误或异常溢价，
# 防止个别离群点 (Outliers) 扭曲 Ordinary Least Squares (OLS) 的回归拟合曲线。
LOWER_QUANTILE = 0.005
UPPER_QUANTILE = 0.995

# 1.2 半导体板块股票 (Semiconductors Industry)
# 代表高成长、高研发投入、高估值弹性特征，市场对其“过度看涨”可能具备更高的容忍度。
SEMICONDUCTOR_TICKERS = [
    'NVDA', 'AVGO', 'AMD', 'INTC', 'QCOM', 'TXN', 
    'AMAT', 'LRCX', 'ADI', 'MU', 'KLAC', 'MCHP', 'MPWR', 'ON'
]

# 1.3 传统价值行业股票 (Traditional Industry)
# 涵盖金融、能源、必需消费、工业与公用事业，代表高股息、稳定现金流、低估值弹性特征。
TRADITIONAL_TICKERS = [
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK',     # 金融
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX',             # 能源
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO',                # 必需消费
    'CAT', 'GE', 'HON', 'LMT', 'BA', 'MMM', 'UNP',               # 工业
    'NEE', 'DUK', 'SO', 'D', 'MCD', 'HD'                        # 公用事业与传统商业
]

def classify_industry_group(ticker):
    """
    将样本划分“半导体、传统行业、通用组”三大维度，
    为后续假设检验中的异质性分组回归 (Grouped Regression) 与拐点对比提供独立因子。
    """
    if ticker in SEMICONDUCTOR_TICKERS:
        return 'Semiconductors'
    elif ticker in TRADITIONAL_TICKERS:
        return 'Traditional Industry'
    else:
        return 'Other/General'

def clean_data():
    """
    数据清洗与特征工程主执行流程
    """
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：未找到输入文件 {INPUT_FILE}")
        return

    print("🚀 [1/2] 开始数据清洗与特征工程 (严格对齐 Word 规范与业务意义)...")
    df = pd.read_csv(INPUT_FILE)
    initial_len = len(df)
    print(f"📊 原始数据总行数: {initial_len} 条")

# ==========================================
# 2 基础数据清洗与截断控制 (Data Cleaning & Truncation Control)
# ==========================================

# 2.1 剔除缺失值：代码、发布日期、分析师机构缺失的无效行，保证样本标识完整性
    df_clean = df.dropna(subset=['ticker', 'event_date', 'firm']).copy()
    
# 2.2 主动剔除距离当前不足 180 天的近期数据。
# 距离当前时间太近的评级发布尚未走完 180 天完整周期，若强行计算收益会导致严重的前瞻偏误 (Look-ahead Bias) 与截断样本偏差。
    df_clean = df_clean.dropna(subset=['forward_return_180d_pct']).copy()
    
# 2.3 剔除非法目标价 (当期与前期目标价必须大于 0，消除数据库录入错漏)
    valid_price_mask = (df_clean['current_price_target'] > 0) & (df_clean['prior_price_target'] > 0)
    df_clean = df_clean[valid_price_mask].copy()

# ==========================================
# 3 核心变量计算 (Feature Engineering - Aligned with Thesis Formula)
# ==========================================
# 3.1 自变量 (X): 预期看涨程度 (Expected Bullishness)
# 公式: (price_target - current_price) / current_price
# 衡量分析师对股票给出的预期溢价看涨比例。
    df_clean['price_target'] = df_clean['current_price_target']
    df_clean['current_price'] = df_clean['prior_price_target']
    
    df_clean['expected_bullishness'] = (
        (df_clean['price_target'] - df_clean['current_price']) / df_clean['current_price']
    )

# 3.2 离群值截尾 (Trimming)
# 剔除上下 1% 极值，确保计量回归不受到误填或极其罕见的过度乐观目标价影响，提升参数估计的稳健性。
    q_low = df_clean['expected_bullishness'].quantile(LOWER_QUANTILE)
    q_high = df_clean['expected_bullishness'].quantile(UPPER_QUANTILE)
    df_clean = df_clean[
        (df_clean['expected_bullishness'] >= q_low) & 
        (df_clean['expected_bullishness'] <= q_high)
    ].copy()

# 3.3 非线性二次项 (X^2)
# 构建预期看涨程度的平方项，用于做倒 U 型 (Quadratic Relation) 非线性关系检验。
    df_clean['expected_bullishness_sq'] = df_clean['expected_bullishness'] ** 2

# 3.4 因变量 (Y): 实际超额收益率 (Actual Excess Return 180d)
# 扣除同期标普 500 指数基准收益。
# 按年份计算全市场标普 500 平均 180 天收益，再用个股收益减去该大盘基准。
# 消除系统性牛熊大盘环境对个股回报的干扰，准确度量分析师评级对超额 Alpha 回报的预测能力。
    df_clean['event_date'] = pd.to_datetime(df_clean['event_date'])
    df_clean['event_year'] = df_clean['event_date'].dt.year
    
    yearly_market_return = df_clean.groupby('event_year')['forward_return_180d_pct'].transform('mean')
    df_clean['actual_excess_return_180d'] = df_clean['forward_return_180d_pct'] - yearly_market_return

# ==========================================
# 4 板块与行业划分 (Categorization)
# ==========================================
    df_clean['industry_group'] = df_clean['ticker'].apply(classify_industry_group)
    df_clean['event_date'] = df_clean['event_date'].dt.strftime('%Y-%m-%d')

# ==========================================
# 5 结果导出 (Export)
# ==========================================
    df_clean.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("="*60)
    print(f"✅ 数据清洗完毕！最终可用样本: {len(df_clean)} 条")
    print(f"💡 半导体板块样本: {(df_clean['industry_group'] == 'Semiconductors').sum()} 条")
    print(f"💡 传统行业样本: {(df_clean['industry_group'] == 'Traditional Industry').sum()} 条")
    print("="*60)

if __name__ == "__main__":
    clean_data()
