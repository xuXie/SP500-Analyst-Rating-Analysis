"""
========================================================================================
项目名称: Team16_标普500股票的分析师评级与目标价准确性研究 (Step 1)
========================================================================================

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
3 核心变量计算与时序聚合 (Feature Engineering & Temporal Aggregation)
    ├── 3.1 预处理: 确保日期格式规范
    ├── 3.2 同日聚合: 多个分析师同日发布目标价取均值 (Daily Mean Aggregation)
    ├── 3.3 自变量: 预期看涨程度 Expected Bullishness = (Price_Target - Current_Price) / Current_Price (T-1 基准)
    ├── 3.4 离群值处理: 0.5% 双侧截尾机制 (Trimming)
    ├── 3.5 非线性二次项: Expected Bullishness Squared (X^2)
    └── 3.6 因变量: 实际超额收益率 Actual Excess Return 180d = 180d Return - Market Mean
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

# 1.1 极值处理设置双端 0.5% 截尾 (Trimming)，用于消除分析师录入极端错误或异常溢价，
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

    print("🚀 [1/2] 开始数据清洗与特征工程 (严格对齐学术规范与前视偏误控制)...")
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
    
# 2.3 剔除非法目标价 (当期与前期基准价必须大于 0，消除数据库录入错漏)
    valid_price_mask = (df_clean['current_price_target'] > 0) & (df_clean['prior_price_target'] > 0)
    df_clean = df_clean[valid_price_mask].copy()

# ==========================================
# 3 核心变量计算与时序聚合 (Feature Engineering & Temporal Aggregation)
# ==========================================
# 3.1 预处理：确保日期格式规范，便于后续聚合与年份提取
    df_clean['event_date'] = pd.to_datetime(df_clean['event_date'])
    df_clean['event_year'] = df_clean['event_date'].dt.year

# 3.2 同日多重覆盖聚合 (Daily Mean Aggregation)
# 为防范信息重叠偏差，同时满足后续模型对机构固定效应（Firm FE）的控制需求，
# 本研究将数据观测层面设定为 (股票代码 ticker, 交易日 event_date, 机构 firm)。
#具体聚合方法如下：
# 若同一机构在同一交易日对同一股票发布了多份包含目标价的研报，则对各项数值型指标取均值，将其合并为单一观测值。
# 不同机构在同日对同股票的发布则分别保留为独立观测值。
# 此方法既确保了同机构同日数据的唯一性，又避免了因全局聚合而遗失具体的机构特征信息。
    
    # 在 (ticker, event_date, firm) 层面聚合，消除同一机构一天发多份报告的冗余
    num_cols = [
        'current_price_target', 'prior_price_target',
        'forward_return_30d_pct', 'forward_return_90d_pct',
        'forward_return_180d_pct', 'forward_return_365d_pct'
    ]
    df_clean = df_clean.groupby(['ticker', 'event_date', 'firm', 'event_year'])[num_cols].mean().reset_index()

# 3.3 自变量 (X): 预期看涨程度 (Expected Bullishness)
# 严格时序规则：目标价涨幅的基准价格 (current_price) 采用研报发布前一交易日 (T-1 日) 的收盘价
# (在此数据集中体现为 prior_price_target，需确保其业务含义为 T-1 价格)，消除日内信息前视偏误。
    df_clean['price_target'] = df_clean['current_price_target']
    df_clean['current_price'] = df_clean['prior_price_target'] # 假设此字段代表T-1收盘价
    
    df_clean['expected_bullishness'] = (
        (df_clean['price_target'] - df_clean['current_price']) / df_clean['current_price']
    )

# 3.4 离群值双侧截尾 (Trimming)
# 剔除上下 0.5% 极值，确保计量回归不受到误填或极其罕见的过度乐观目标价影响，提升参数估计的稳健性。
    q_low = df_clean['expected_bullishness'].quantile(LOWER_QUANTILE)
    q_high = df_clean['expected_bullishness'].quantile(UPPER_QUANTILE)
    df_clean = df_clean[
        (df_clean['expected_bullishness'] >= q_low) & 
        (df_clean['expected_bullishness'] <= q_high)
    ].copy()

# 3.5 非线性二次项 (X^2)
# 构建预期看涨程度的平方项，用于做倒 U 型 (Quadratic Relation) 非线性关系检验。
    df_clean['expected_bullishness_sq'] = df_clean['expected_bullishness'] ** 2

# 3.6 因变量 (Y): 实际超额收益率 (Actual Excess Return 180d)
# 扣除同期标普 500 指数基准收益。
# 按年份计算全市场标普 500 平均 180 天收益，再用个股收益减去该大盘基准。
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
    print(f"✅ 数据清洗与聚合完毕！最终可用样本: {len(df_clean)} 条")
    print(f"💡 半导体板块样本: {(df_clean['industry_group'] == 'Semiconductors').sum()} 条")
    print(f"💡 传统行业样本: {(df_clean['industry_group'] == 'Traditional Industry').sum()} 条")
    print("="*60)

if __name__ == "__main__":
    clean_data()