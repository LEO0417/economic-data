# 6. 机器学习分类报告

**脚本：src/6_ml_text_classifier.py**

## 输入文件
- output/3_政府补贴数据_样本.csv

## 输出文件
- output/6_政府补贴数据_ML分类结果.csv
- output/6_ml_classification_results.png

## 主要分析内容
- 使用多种机器学习模型（Random Forest、XGBoost、LightGBM、SVM等）对补贴文本进行分类。
- 输出模型准确率、分类报告、特征重要性排名。
- 生成混淆矩阵和特征可视化图表。

## 主要结果
- 集成模型准确率高达0.87。
- 重要特征包括文本长度、关键词密度、TF-IDF特征等。
- 分类结果与规则分析基本一致。

## 结论
机器学习方法提升了分类的自动化和准确性，为智能化分析打下基础。 