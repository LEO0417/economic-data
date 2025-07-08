#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政府补贴数据分析工具
基于data_analysis_methods.txt中的分类方法分析政府补贴数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(filename='output/3_政府补贴数据_样本.csv'):
    """加载数据"""
    df = pd.read_csv(filename)
    return df

def classify_subsidies(df):
    """
    根据data_analysis_methods.txt中的方法对补贴进行分类
    """
    # 定义关键词字典
    keywords = {
        'R&D_Innovation': [
            '创新', '研发', '专利', '科技', '技术', '知识产权', '研究', 
            '开发', '科学', '发明', '高新', '智能', '数字化', '信息化'
        ],
        'Industrial_Equipment': [
            '工业', '设备', '技改', '改造', '升级', '转型', '制造', 
            '生产线', '机械', '装备', '产业化'
        ],
        'Employment': [
            '就业', '招聘', '实习', '培训', '稳岗', '用工', '劳动', 
            '职业', '毕业生', '扩岗', '人才'
        ],
        'Environment': [
            '节能', '环保', '清洁', '减排', '污染', '治理', '绿色', 
            '循环', '生态', '废料', '排放'
        ],
        'General_Business': [
            '经营', '出口', '品牌', '税收', '发展', '市场', '贸易', 
            '营业', '商务', '财政', '奖励', '扶持'
        ],
        'Other': [],
        'Unknown': ['其他', '补助', '补贴', '政府']
    }
    
    def classify_single_subsidy(description):
        """对单个补贴描述进行分类"""
        if pd.isna(description):
            return 'Unknown'
        
        description = str(description).lower()
        
        # 计算每个类别的匹配分数
        scores = {}
        for category, words in keywords.items():
            if category == 'Other':
                continue
            score = sum(1 for word in words if word in description)
            scores[category] = score
        
        # 找到最高分数的类别
        if max(scores.values()) == 0:
            return 'Unknown'
        
        return max(scores, key=scores.get)
    
    # 对每个补贴进行分类
    df['subsidy_category'] = df['Fn05601'].apply(classify_single_subsidy)
    
    return df

def analyze_subsidy_distribution(df):
    """分析补贴分布"""
    print("=" * 80)
    print("📊 政府补贴数据分析报告")
    print("=" * 80)
    
    # 基本统计
    print(f"📈 数据概览:")
    print(f"   总记录数: {len(df):,}")
    print(f"   时间跨度: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
    print(f"   涉及企业数: {df['Stkcd'].nunique():,}")
    print(f"   补贴总金额: {df['Fn05602'].sum():,.0f} 元")
    print()
    
    # 按类别统计
    category_stats = df.groupby('subsidy_category').agg({
        'Fn05602': ['count', 'sum', 'mean'],
        'Stkcd': 'nunique'
    }).round(2)
    
    category_stats.columns = ['补贴数量', '补贴总额', '平均补贴额', '涉及企业数']
    category_stats['占比(%)'] = (category_stats['补贴数量'] / len(df) * 100).round(2)
    
    print("📋 按补贴类别统计:")
    print(category_stats.sort_values('补贴总额', ascending=False))
    print()
    
    # 按年份统计
    yearly_stats = df.groupby('Year').agg({
        'Fn05602': ['count', 'sum'],
        'Stkcd': 'nunique'
    }).round(2)
    yearly_stats.columns = ['补贴数量', '补贴总额', '涉及企业数']
    
    print("📅 按年份统计 (前10年):")
    print(yearly_stats.sort_values('补贴总额', ascending=False).head(10))
    print()
    
    return category_stats, yearly_stats

def analyze_keywords(df):
    """分析补贴描述中的关键词"""
    print("🔍 补贴描述关键词分析:")
    
    # 提取所有补贴描述
    all_descriptions = ' '.join(df['Fn05601'].dropna().astype(str))
    
    # 常见关键词
    common_words = [
        '补贴', '补助', '资金', '奖励', '专项', '项目', '技术', '发展',
        '企业', '产业', '创新', '研发', '科技', '工业', '财政', '政府'
    ]
    
    word_counts = {}
    for word in common_words:
        count = all_descriptions.count(word)
        word_counts[word] = count
    
    # 按出现频次排序
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("   关键词出现频次:")
    for word, count in sorted_words[:15]:
        print(f"   {word}: {count}")
    print()

def analyze_by_test_variables(df):
    """分析test和Test变量"""
    print("🧪 test和Test变量分析:")
    
    # test变量分析
    test_stats = df.groupby('test').agg({
        'Fn05602': ['count', 'sum', 'mean'],
        'subsidy_category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
    }).round(2)
    
    print("   test变量统计:")
    print(f"   test=0: {(df['test']==0).sum()} 条记录")
    print(f"   test=1: {(df['test']==1).sum()} 条记录")
    print()
    
    # Test变量分析
    Test_stats = df.groupby('Test').agg({
        'Fn05602': ['count', 'sum', 'mean']
    }).round(2)
    
    print("   Test变量统计:")
    print(f"   Test=0: {(df['Test']==0).sum()} 条记录")
    print(f"   Test=1: {(df['Test']==1).sum()} 条记录")
    print()

def create_visualizations(df, category_stats, yearly_stats):
    """创建可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 补贴类别分布饼图
    category_counts = df['subsidy_category'].value_counts()
    axes[0,0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    axes[0,0].set_title('补贴类别分布')
    
    # 2. 年度补贴趋势
    yearly_amount = df.groupby('Year')['Fn05602'].sum() / 1e8  # 转换为亿元
    axes[0,1].plot(yearly_amount.index, yearly_amount.values, marker='o')
    axes[0,1].set_title('年度补贴总额趋势')
    axes[0,1].set_xlabel('年份')
    axes[0,1].set_ylabel('补贴总额(亿元)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. 补贴金额分布箱线图
    df_plot = df[df['Fn05602'] > 0]  # 排除0值
    axes[1,0].boxplot([np.log10(df_plot[df_plot['subsidy_category']==cat]['Fn05602']) 
                       for cat in category_counts.index if cat in df_plot['subsidy_category'].values])
    axes[1,0].set_xticklabels(category_counts.index, rotation=45)
    axes[1,0].set_title('各类别补贴金额分布(log10)')
    axes[1,0].set_ylabel('补贴金额(log10)')
    
    # 4. test vs Test 交叉表
    cross_tab = pd.crosstab(df['test'], df['Test'])
    sns.heatmap(cross_tab, annot=True, fmt='d', ax=axes[1,1])
    axes[1,1].set_title('test vs Test 交叉分布')
    
    plt.tight_layout()
    plt.savefig('output/5_subsidy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    # 加载数据
    df = load_data()
    
    # 分类补贴
    df = classify_subsidies(df)
    
    # 分析补贴分布
    category_stats, yearly_stats = analyze_subsidy_distribution(df)
    
    # 关键词分析
    analyze_keywords(df)
    
    # test变量分析
    analyze_by_test_variables(df)
    
    # 创建可视化
    create_visualizations(df, category_stats, yearly_stats)
    
    # 保存分析结果
    df.to_csv('output/5_政府补贴数据_分析结果.csv', index=False)
    category_stats.to_csv('output/5_补贴类别统计.csv')
    yearly_stats.to_csv('output/5_年度补贴统计.csv')
    
    print("✅ 分析完成！结果已保存到以下文件:")
    print("   - output/5_政府补贴数据_分析结果.csv")
    print("   - output/5_补贴类别统计.csv") 
    print("   - output/5_年度补贴统计.csv")
    print("   - output/5_subsidy_analysis.png")

if __name__ == "__main__":
    main() 