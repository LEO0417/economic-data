#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机提取政府补贴数据的前千分之一样本
作者：自动生成
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def random_sample_dta(input_file, output_file, sample_ratio=0.001):
    """
    从.dta文件中随机提取样本
    
    参数:
    input_file (str): 输入的.dta文件路径
    output_file (str): 输出的.dta文件路径
    sample_ratio (float): 采样比例，默认为0.001（千分之一）
    """
    
    print(f"正在读取数据文件: {input_file}")
    
    try:
        # 读取.dta文件
        df = pd.read_stata(input_file)
        
        print(f"原始数据集大小: {len(df)} 行, {len(df.columns)} 列")
        
        # 计算样本大小
        sample_size = int(len(df) * sample_ratio)
        print(f"将要提取的样本大小: {sample_size} 行")
        
        # 随机设置种子以确保可重现性
        np.random.seed(42)
        
        # 随机采样
        sampled_df = df.sample(n=sample_size, random_state=42)
        
        print(f"随机采样完成，样本大小: {len(sampled_df)} 行")
        
        # 处理中文列名，重命名为英文以兼容Stata格式
        column_mapping = {}
        for i, col in enumerate(sampled_df.columns):
            # 创建英文列名映射
            if any('\u4e00' <= char <= '\u9fff' for char in str(col)):
                new_col_name = f"var_{i+1}"
                column_mapping[col] = new_col_name
                print(f"列名映射: '{col}' -> '{new_col_name}'")
        
        # 重命名列
        if column_mapping:
            sampled_df = sampled_df.rename(columns=column_mapping)
            print(f"已重命名 {len(column_mapping)} 个包含中文的列名")
            
            # 保存列名映射到文件
            mapping_file = "../config/" + os.path.basename(output_file).replace('.dta', '_列名映射.txt')
            with open(mapping_file, 'w', encoding='utf-8') as f:
                f.write("原始列名 -> 新列名\n")
                f.write("=" * 30 + "\n")
                for old_name, new_name in column_mapping.items():
                    f.write(f"{old_name} -> {new_name}\n")
            print(f"列名映射已保存到: {mapping_file}")
        
        # 处理数据中的中文字符
        print("正在处理数据中的中文字符...")
        for col in sampled_df.columns:
            if sampled_df[col].dtype == 'object':
                # 对字符串列进行编码处理
                try:
                    # 尝试将中文字符转换为可以保存到Stata的格式
                    sampled_df[col] = sampled_df[col].astype(str)
                    # 检查是否包含中文字符
                    if sampled_df[col].str.contains('[\u4e00-\u9fff]', na=False).any():
                        print(f"发现列 '{col}' 包含中文字符，将进行编码处理")
                        # 可以选择保留原样或进行其他处理
                except Exception as e:
                    print(f"处理列 '{col}' 时出现问题: {e}")
        
        # 保存为新的.dta文件
        try:
            # 不添加中文标签，避免编码问题
            sampled_df.to_stata(output_file, write_index=False, version=118)
            print("使用Stata版本118格式保存成功")
        except UnicodeEncodeError:
            try:
                # 如果还有问题，尝试将所有字符串列转换为字节
                print("尝试另一种编码方式...")
                df_copy = sampled_df.copy()
                for col in df_copy.columns:
                    if df_copy[col].dtype == 'object':
                        df_copy[col] = df_copy[col].astype(str).str.encode('utf-8', errors='ignore').str.decode('latin-1', errors='ignore')
                df_copy.to_stata(output_file, write_index=False, version=118)
                print("使用UTF-8转Latin-1编码保存成功")
            except Exception as e2:
                print(f"保存失败，尝试保存为CSV格式: {e2}")
                csv_file = output_file.replace('.dta', '.csv')
                sampled_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                print(f"已保存为CSV格式: {csv_file}")
                return sampled_df
        
        print(f"样本数据已保存到: {output_file}")
        
        # 显示基本统计信息
        print("\n样本数据基本信息:")
        print(f"- 数据形状: {sampled_df.shape}")
        print(f"- 列名: {list(sampled_df.columns)}")
        
        return sampled_df
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        return None

def main():
    """主函数"""
    
    # 设置文件路径（相对于项目根目录）
    input_file = "../data/政府补贴数据.dta"
    output_file = "../data/政府补贴数据_样本.dta"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        print("请确保原始数据文件位于 data/ 目录中")
        return
    
    print("=" * 50)
    print("政府补贴数据随机采样程序")
    print("=" * 50)
    
    # 执行随机采样
    sample_data = random_sample_dta(input_file, output_file, sample_ratio=0.001)
    
    if sample_data is not None:
        print("\n采样完成！")
        
        # 显示样本数据的前几行
        print("\n样本数据预览（前5行）:")
        print(sample_data.head())
        
        print(f"\n输出文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
    else:
        print("采样失败！")

if __name__ == "__main__":
    main() 