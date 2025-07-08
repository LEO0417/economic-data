#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预览工具
用于快速查看dta文件转换后的CSV数据
"""

import pandas as pd
import os

def preview_csv(filename='output/3_政府补贴数据_样本.csv'):
    """预览CSV文件"""
    if not os.path.exists(filename):
        print(f"❌ 文件 {filename} 不存在")
        return
    
    try:
        # 读取CSV文件
        df = pd.read_csv(filename)
        
        print("=" * 60)
        print(f"📊 数据文件预览: {filename}")
        print("=" * 60)
        
        # 基本信息
        print(f"📏 数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"💾 内存使用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print()
        
        # 列名信息
        print("📋 列名列表:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        print()
        
        # 数据类型
        print("🔍 数据类型:")
        print(df.dtypes)
        print()
        
        # 缺失值统计
        print("❓ 缺失值统计:")
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        missing_info = pd.DataFrame({
            '缺失数量': missing,
            '缺失比例(%)': missing_percent.round(2)
        })
        print(missing_info[missing_info['缺失数量'] > 0])
        print()
        
        # 前5行数据
        print("👀 前5行数据预览:")
        print(df.head())
        print()
        
        # 基本统计信息（仅数值列）
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print("📈 数值列基本统计:")
            print(df[numeric_cols].describe())
        
        print("=" * 60)
        print("✅ 预览完成")
        
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")

if __name__ == "__main__":
    preview_csv() 