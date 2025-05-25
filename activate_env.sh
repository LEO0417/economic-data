#!/bin/bash
# 激活政府补贴数据分析项目的虚拟环境

echo "🚀 激活政府补贴数据分析项目虚拟环境..."
source economic_data_env/bin/activate

echo "✅ 虚拟环境已激活！"
echo "📍 当前Python路径: $(which python)"
echo "📦 已安装的包:"
pip list --format=columns

echo ""
echo "💡 使用提示:"
echo "   - 运行脚本: cd src && python script_name.py"
echo "   - 退出环境: deactivate"
echo "   - 查看项目结构: tree 或 ls -la"
echo "" 