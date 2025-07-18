# 政府补贴数据分析工具

## 项目简介

这是一个用于政府补贴数据处理和分析的Python工具集，支持数据采样、统计分析和可视化功能。

## 项目结构

```text
economic data/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
├── src/                        # 源代码目录
│   ├── random_sample.py        # 数据随机采样脚本
│   ├── subsidy_analysis.py     # 补贴数据分析脚本
│   ├── preview_data.py         # 数据预览脚本
│   ├── convert_to_csv.py       # 数据格式转换脚本
│   └── 验证样本数据.py          # 样本数据验证脚本
├── data/                       # 数据文件目录
│   ├── 政府补贴数据.dta         # 原始Stata数据文件
│   └── 政府补贴数据_样本.dta    # 采样后的数据文件
├── output/                     # 输出结果目录
│   ├── 政府补贴数据_样本.csv    # CSV格式样本数据
│   ├── 政府补贴数据_分析结果.csv # 数据分析结果
│   ├── 年度补贴统计.csv         # 年度统计结果
│   ├── 补贴类别统计.csv         # 补贴类别统计
│   └── subsidy_analysis.png    # 分析图表
├── docs/                       # 文档目录
│   ├── 政府补贴数据分析报告.md   # 数据分析报告
│   └── 政府补贴数据列标题分析报告.md # 列标题分析报告
└── config/                     # 配置文件目录
    ├── data_analysis_methods.txt # 数据分析方法说明
    └── 政府补贴数据_样本_列名映射.txt # 列名映射记录
```

## 功能特点

- **数据采样**: 从大型Stata数据文件中随机提取样本数据
- **格式转换**: 支持Stata (.dta) 和 CSV 格式之间的转换
- **统计分析**: 提供补贴数据的统计分析功能
- **数据可视化**: 生成分析图表和报告
- **数据验证**: 验证采样数据的完整性和准确性

## 环境要求

- Python 3.6+
- pandas库（用于数据处理）
- numpy库（用于随机数生成）
- matplotlib库（用于数据可视化）

## 环境配置

### 方法1: 使用虚拟环境（推荐）

```bash
# 激活项目虚拟环境
./activate_env.sh

# 或手动激活
source economic_data_env/bin/activate
```

### 方法2: 全局安装

```bash
pip install -r requirements.txt
```

> 💡 **提示**: 项目已配置好专属虚拟环境 `economic_data_env`，包含所有必要的依赖包。详细配置说明请参考 [环境配置说明](docs/环境配置说明.md)。

## 使用方法

### 1. 数据采样

```bash
cd src
python random_sample.py
```

### 2. 数据分析

```bash
cd src
python subsidy_analysis.py
```

### 3. 数据预览

```bash
cd src
python preview_data.py
```

### 4. 数据验证

```bash
cd src
python 验证样本数据.py
```

## 输出说明

- **采样结果**: 生成千分之一的随机样本数据
- **分析报告**: 包含详细的统计分析结果
- **可视化图表**: 补贴数据的图形化展示
- **统计汇总**: 按年度和类别的统计结果

## 注意事项

- 脚本使用固定的随机种子，确保结果可重现
- 处理大文件时可能需要较多内存
- 确保有足够的磁盘空间存储输出文件
- 中文列名会自动转换为英文变量名以确保兼容性

## 数据处理说明

### 原始数据规模

- **原始数据**: 685,169行，9列，文件大小511.64MB
- **样本数据**: 685行（千分之一），9列文件大小0.17MB
- **压缩比例**: 0.033%

### 数据处理特性

1. **中文列名处理**: 自动转换中文列名为英文变量名
2. **编码兼容性**: 使用Stata版本118格式保存
3. **数据完整性**: 保持原始数据的结构和数据类型

## 许可证

本项目采用MIT许可证。
