#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式转换工具
将Stata格式(.dta)转换为CSV格式
"""

import pandas as pd
import os

def convert_dta_to_csv(input_file, output_file):
    """
    将Stata文件转换为CSV格式
    
    参数:
    input_file (str): 输入的.dta文件路径
    output_file (str): 输出的.csv文件路径
    """
    try:
        print(f"正在读取文件: {input_file}")
        
        # 读取dta文件
        df = pd.read_stata(input_file)
        
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 保存为CSV
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"转换完成！文件已保存到: {output_file}")
        print(f"输出文件大小: {os.path.getsize(output_file) / 1024:.2f} KB")
        
        return True
        
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        return False

def ensure_output_directory(output_file):
    """
    确保输出目录存在，如果不存在则创建
    
    参数:
    output_file (str): 输出文件的路径
    """
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def main():
    """主函数"""
    # 设置文件路径
    input_file = "../data/政府补贴数据_样本.dta"
    output_file = "../output/政府补贴数据_样本.csv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        print("请先运行 random_sample.py 生成样本数据")
        return
    
    # 确保输出目录存在
    ensure_output_directory(output_file)
    
    print("=" * 50)
    print("Stata到CSV格式转换工具")
    print("=" * 50)
    
    # 执行转换
    success = convert_dta_to_csv(input_file, output_file)
    
    if success:
        print("\n✅ 转换成功！")
    else:
        print("\n❌ 转换失败！")

if __name__ == "__main__":
    main()
