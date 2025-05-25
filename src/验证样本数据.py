#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证样本数据脚本
"""

import pandas as pd
import os

def verify_sample_data():
    """验证样本数据"""
    
    sample_file = "../data/政府补贴数据_样本.dta"
    original_file = "../data/政府补贴数据.dta"
    
    if not os.path.exists(sample_file):
        print(f"错误: 找不到样本文件 {sample_file}")
        return
    
    print("=" * 50)
    print("样本数据验证报告")
    print("=" * 50)
    
    # 读取样本数据
    print("正在读取样本数据...")
    sample_df = pd.read_stata(sample_file)
    
    print(f"样本数据大小: {sample_df.shape}")
    print(f"列名: {list(sample_df.columns)}")
    
    # 显示基本统计信息
    print("\n数值列的基本统计信息:")
    numeric_columns = sample_df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 0:
        print(sample_df[numeric_columns].describe())
    
    # 显示前几行数据
    print("\n样本数据前5行:")
    print(sample_df.head())
    
    # 检查数据类型
    print("\n各列数据类型:")
    print(sample_df.dtypes)
    
    # 文件大小比较
    sample_size = os.path.getsize(sample_file) / (1024*1024)
    original_size = os.path.getsize(original_file) / (1024*1024)
    
    print(f"\n文件大小比较:")
    print(f"原始文件: {original_size:.2f} MB")
    print(f"样本文件: {sample_size:.2f} MB")
    print(f"压缩比例: {(sample_size/original_size)*100:.3f}%")

if __name__ == "__main__":
    verify_sample_data() 