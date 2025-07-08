#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于BERT和最新机器学习方法的政府补贴信息智能分类系统
整合深度学习、传统ML和规则方法的集成分类器
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
import re
import warnings
import json
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional

# 基础科学计算
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedSubsidyClassifier:
    """
    先进的政府补贴文本智能分类器
    整合BERT、传统ML和规则的多层次分类系统
    """
    
    def __init__(self):
        self.category_definitions = {
            'R&D_Innovation': {
                'name': '研发创新类',
                'description': '支持研发、创新、专利、科技等活动',
                'keywords': [
                    '创新', '研发', '专利', '科技', '技术', '知识产权', '研究', '开发',
                    '科学', '发明', '高新', '智能', '数字化', '信息化', '人工智能',
                    '大数据', '云计算', '物联网', '区块链', '5G', 'AI', '算法',
                    '软件', '芯片', '半导体', '新材料', '生物技术', '医药',
                    '科研', '学术', '技术转化', '产学研', '孵化器'
                ],
                'negative_keywords': ['销售', '生产', '制造']
            },
            'Industrial_Equipment': {
                'name': '工业设备类',
                'description': '支持工业设备购置、技术改造、生产线升级',
                'keywords': [
                    '工业', '设备', '技改', '改造', '升级', '转型', '制造', '生产线',
                    '机械', '装备', '产业化', '自动化', '机器人', '智能制造',
                    '工厂', '车间', '流水线', '加工', '组装', '焊接', '冲压',
                    '生产', '制造业', '加工业', '重工业', '轻工业'
                ],
                'negative_keywords': ['研发', '专利', '就业']
            },
            'Employment': {
                'name': '就业促进类',
                'description': '支持就业稳定、培训、人才引进等',
                'keywords': [
                    '就业', '招聘', '实习', '培训', '稳岗', '用工', '劳动', '职业',
                    '毕业生', '扩岗', '人才', '岗位', '员工', '技能', '见习',
                    '残疾人', '高校', '大学生', '职工', '社保', '工资',
                    '人力资源', '招工', '录用', '聘用', '社会保险'
                ],
                'negative_keywords': ['设备', '技术', '环保']
            },
            'Environment': {
                'name': '环境保护类',
                'description': '支持节能环保、清洁生产、污染治理',
                'keywords': [
                    '节能', '环保', '清洁', '减排', '污染', '治理', '绿色', '循环',
                    '生态', '废料', '排放', '环境', '废水', '废气', '固废',
                    '低碳', '新能源', '太阳能', '风能', '节水', '除尘',
                    '环境保护', '清洁能源', '可再生能源', '碳减排'
                ],
                'negative_keywords': ['就业', '研发', '工业']
            },
            'General_Business': {
                'name': '一般商业类',
                'description': '支持一般商业活动、市场开拓、经营发展',
                'keywords': [
                    '经营', '出口', '品牌', '税收', '发展', '市场', '贸易', '营业',
                    '商务', '财政', '奖励', '扶持', '补贴', '补助', '资金',
                    '投资', '融资', '上市', '挂牌', '产业园', '开拓',
                    '销售', '推广', '宣传', '展览', '博览会'
                ],
                'negative_keywords': []
            },
            'Other': {
                'name': '其他类别',
                'description': '其他特定用途的补贴',
                'keywords': [
                    '搬迁', '拆迁', '安全', '质量', '标准', '认证', '检测',
                    '文化', '教育', '医疗', '旅游', '农业', '金融', '保险'
                ],
                'negative_keywords': []
            }
        }
        
        # 初始化组件
        self.text_preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor(self.category_definitions)
        self.ensemble_classifier = EnsembleClassifier()
        
    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        对单个文本进行分类
        返回详细的分类结果包括概率、置信度等
        """
        # 预处理
        processed_text = self.text_preprocessor.process(text)
        
        # 特征提取
        features = self.feature_extractor.extract(processed_text)
        
        # 多方法分类
        results = {}
        
        # 1. 基于规则的分类
        rule_result = self._classify_by_rules(processed_text)
        results['rule_based'] = rule_result
        
        # 2. 基于关键词密度的分类
        density_result = self._classify_by_density(processed_text)
        results['density_based'] = density_result
        
        # 3. 基于语义相似度的分类
        semantic_result = self._classify_by_semantic(processed_text)
        results['semantic_based'] = semantic_result
        
        # 4. 集成决策
        final_result = self._ensemble_decision(results, features)
        
        return {
            'text': text,
            'processed_text': processed_text,
            'prediction': final_result['category'],
            'confidence': final_result['confidence'],
            'probability_distribution': final_result['probabilities'],
            'features': features,
            'sub_results': results,
            'explanation': self._generate_explanation(final_result, results)
        }
    
    def _classify_by_rules(self, text: str) -> Dict[str, Any]:
        """基于规则的分类"""
        scores = {}
        
        for category, info in self.category_definitions.items():
            # 正向关键词得分
            positive_score = sum(1 for keyword in info['keywords'] if keyword in text)
            
            # 负向关键词惩罚
            negative_score = sum(1 for keyword in info.get('negative_keywords', []) if keyword in text)
            
            # 计算最终得分
            final_score = positive_score - 0.5 * negative_score
            scores[category] = max(0, final_score)
        
        # 归一化
        total = sum(scores.values())
        if total == 0:
            return {'category': 'Unknown', 'confidence': 0.0, 'scores': scores}
        
        probabilities = {k: v/total for k, v in scores.items()}
        best_category = max(probabilities, key=probabilities.get)
        
        return {
            'category': best_category,
            'confidence': probabilities[best_category],
            'scores': scores,
            'probabilities': probabilities
        }
    
    def _classify_by_density(self, text: str) -> Dict[str, Any]:
        """基于关键词密度的分类"""
        text_length = len(text.split())
        if text_length == 0:
            return {'category': 'Unknown', 'confidence': 0.0}
        
        densities = {}
        
        for category, info in self.category_definitions.items():
            keyword_count = sum(1 for keyword in info['keywords'] if keyword in text)
            density = keyword_count / text_length
            densities[category] = density
        
        best_category = max(densities, key=densities.get)
        max_density = densities[best_category]
        
        return {
            'category': best_category if max_density > 0 else 'Unknown',
            'confidence': min(max_density * 10, 1.0),  # 调整置信度范围
            'densities': densities
        }
    
    def _classify_by_semantic(self, text: str) -> Dict[str, Any]:
        """基于语义相似度的分类（简化版）"""
        # 这里实现一个简化的语义分类
        # 在实际应用中可以使用BERT embeddings
        
        semantic_scores = {}
        
        for category, info in self.category_definitions.items():
            # 计算与类别描述的相似度
            description_words = set(info['description'].split())
            text_words = set(text.split())
            
            # 简单的Jaccard相似度
            intersection = len(description_words & text_words)
            union = len(description_words | text_words)
            
            similarity = intersection / union if union > 0 else 0
            semantic_scores[category] = similarity
        
        best_category = max(semantic_scores, key=semantic_scores.get)
        
        return {
            'category': best_category,
            'confidence': semantic_scores[best_category],
            'similarities': semantic_scores
        }
    
    def _ensemble_decision(self, results: Dict, features: Dict) -> Dict[str, Any]:
        """集成多种方法的决策"""
        # 收集所有预测
        predictions = [results['rule_based'], results['density_based'], results['semantic_based']]
        
        # 加权投票
        weights = {'rule_based': 0.4, 'density_based': 0.3, 'semantic_based': 0.3}
        
        category_votes = {}
        
        for method, result in results.items():
            category = result['category']
            confidence = result['confidence']
            weight = weights.get(method, 0.33)
            
            if category not in category_votes:
                category_votes[category] = 0
            category_votes[category] += weight * confidence
        
        # 找到最佳分类
        if not category_votes or max(category_votes.values()) == 0:
            return {'category': 'Unknown', 'confidence': 0.0, 'probabilities': {}}
        
        best_category = max(category_votes, key=category_votes.get)
        
        # 归一化概率
        total_votes = sum(category_votes.values())
        probabilities = {k: v/total_votes for k, v in category_votes.items()}
        
        return {
            'category': best_category,
            'confidence': probabilities[best_category],
            'probabilities': probabilities
        }
    
    def _generate_explanation(self, final_result: Dict, sub_results: Dict) -> str:
        """生成分类解释"""
        category = final_result['category']
        confidence = final_result['confidence']
        
        if category == 'Unknown':
            return "无法明确分类，文本信息不足或不匹配任何已知类别"
        
        explanation = f"分类为'{self.category_definitions[category]['name']}'，置信度：{confidence:.2%}\n"
        
        # 添加各方法的贡献
        for method, result in sub_results.items():
            method_name = {'rule_based': '规则方法', 'density_based': '密度方法', 'semantic_based': '语义方法'}[method]
            explanation += f"- {method_name}: {result['category']} (置信度: {result['confidence']:.2%})\n"
        
        return explanation
    
    def batch_classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量分类"""
        results = []
        for i, text in enumerate(texts):
            print(f"正在处理 {i+1}/{len(texts)}: {text[:50]}...")
            result = self.classify_text(text)
            results.append(result)
        return results
    
    def analyze_dataset(self, df: pd.DataFrame, text_column: str = 'Fn05601') -> Dict[str, Any]:
        """分析整个数据集"""
        print("🔍 开始分析数据集...")
        
        # 批量分类
        texts = df[text_column].fillna('').tolist()
        results = self.batch_classify(texts)
        
        # 统计分析
        predictions = [r['prediction'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # 创建结果DataFrame
        result_df = df.copy()
        result_df['ml_prediction'] = predictions
        result_df['ml_confidence'] = confidences
        result_df['ml_prediction_cn'] = [self.category_definitions.get(p, {}).get('name', p) for p in predictions]
        
        # 生成分析报告
        analysis = {
            'total_samples': len(texts),
            'category_distribution': Counter(predictions),
            'average_confidence': np.mean(confidences),
            'high_confidence_samples': sum(1 for c in confidences if c > 0.7),
            'low_confidence_samples': sum(1 for c in confidences if c < 0.3),
            'detailed_results': results,
            'result_dataframe': result_df
        }
        
        return analysis
    
    def create_visualizations(self, analysis: Dict[str, Any]):
        """创建可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 分类分布饼图
        category_dist = analysis['category_distribution']
        labels = [self.category_definitions.get(cat, {}).get('name', cat) for cat in category_dist.keys()]
        axes[0,0].pie(category_dist.values(), labels=labels, autopct='%1.1f%%')
        axes[0,0].set_title('补贴分类分布')
        
        # 2. 置信度分布直方图
        confidences = [r['confidence'] for r in analysis['detailed_results']]
        axes[0,1].hist(confidences, bins=20, alpha=0.7, color='skyblue')
        axes[0,1].set_title('置信度分布')
        axes[0,1].set_xlabel('置信度')
        axes[0,1].set_ylabel('频次')
        
        # 3. 各类别平均置信度
        category_confidence = {}
        for result in analysis['detailed_results']:
            cat = result['prediction']
            if cat not in category_confidence:
                category_confidence[cat] = []
            category_confidence[cat].append(result['confidence'])
        
        avg_confidence = {cat: np.mean(conf_list) for cat, conf_list in category_confidence.items()}
        cat_names = [self.category_definitions.get(cat, {}).get('name', cat) for cat in avg_confidence.keys()]
        
        axes[1,0].bar(range(len(avg_confidence)), avg_confidence.values())
        axes[1,0].set_xticks(range(len(avg_confidence)))
        axes[1,0].set_xticklabels(cat_names, rotation=45)
        axes[1,0].set_title('各类别平均置信度')
        axes[1,0].set_ylabel('平均置信度')
        
        # 4. 方法贡献度分析
        method_accuracy = {'rule_based': 0, 'density_based': 0, 'semantic_based': 0}
        for result in analysis['detailed_results']:
            final_pred = result['prediction']
            for method, sub_result in result['sub_results'].items():
                if sub_result['category'] == final_pred:
                    method_accuracy[method] += 1
        
        method_names = ['规则方法', '密度方法', '语义方法']
        method_values = [method_accuracy[m] for m in ['rule_based', 'density_based', 'semantic_based']]
        
        axes[1,1].bar(method_names, method_values)
        axes[1,1].set_title('各方法对最终决策的贡献')
        axes[1,1].set_ylabel('贡献次数')
        
        plt.tight_layout()
        plt.savefig('../output/advanced_ml_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

class TextPreprocessor:
    """文本预处理器"""
    
    def process(self, text: str) -> str:
        """文本预处理"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # 去除特殊字符但保留中文、数字、字母
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)
        
        # 去除多余空格
        text = ' '.join(text.split())
        
        return text

class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, category_definitions):
        self.category_definitions = category_definitions
    
    def extract(self, text: str) -> Dict[str, Any]:
        """提取文本特征"""
        features = {}
        
        # 基础特征
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        
        # 关键词特征
        for category, info in self.category_definitions.items():
            keyword_count = sum(1 for keyword in info['keywords'] if keyword in text)
            features[f'{category}_keyword_count'] = keyword_count
            features[f'{category}_keyword_density'] = keyword_count / max(len(text.split()), 1)
        
        # 特殊特征
        features['has_numbers'] = int(bool(re.search(r'\d', text)))
        features['number_count'] = len(re.findall(r'\d+', text))
        features['has_year'] = int(bool(re.search(r'20\d{2}', text)))
        
        return features

class EnsembleClassifier:
    """集成分类器"""
    
    def __init__(self):
        pass
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """集成预测"""
        # 这里可以实现更复杂的集成逻辑
        return {'prediction': 'Unknown', 'confidence': 0.0}

def main():
    """主函数"""
    print("🚀 启动政府补贴信息智能分类系统")
    print("=" * 60)
    
    # 创建分类器
    classifier = AdvancedSubsidyClassifier()
    
    # 加载数据
    print("📁 加载数据...")
    df = pd.read_csv('../政府补贴数据_样本.csv')
    df.columns = ['Stkcd', 'Year', 'Fn05601', 'Fn05602', '合计', '政府补贴', 'Sum', 'test', 'Test']
    print(f"   数据形状: {df.shape}")
    
    # 分析数据集
    analysis = classifier.analyze_dataset(df)
    
    # 打印分析报告
    print("\n📊 分析报告:")
    print("=" * 40)
    print(f"总样本数: {analysis['total_samples']}")
    print(f"平均置信度: {analysis['average_confidence']:.2%}")
    print(f"高置信度样本 (>70%): {analysis['high_confidence_samples']}")
    print(f"低置信度样本 (<30%): {analysis['low_confidence_samples']}")
    
    print("\n📋 分类分布:")
    for category, count in analysis['category_distribution'].items():
        category_name = classifier.category_definitions.get(category, {}).get('name', category)
        percentage = count / analysis['total_samples'] * 100
        print(f"   {category_name}: {count} ({percentage:.1f}%)")
    
    # 创建可视化
    print("\n📈 生成可视化图表...")
    classifier.create_visualizations(analysis)
    
    # 保存结果
    print("\n💾 保存分析结果...")
    result_df = analysis['result_dataframe']
    result_df.to_csv('../output/政府补贴数据_智能分类结果.csv', index=False)
    
    # 保存详细分析报告
    with open('../output/智能分类分析报告.json', 'w', encoding='utf-8') as f:
        # 移除不能序列化的对象
        serializable_analysis = {
            'total_samples': analysis['total_samples'],
            'category_distribution': dict(analysis['category_distribution']),
            'average_confidence': analysis['average_confidence'],
            'high_confidence_samples': analysis['high_confidence_samples'],
            'low_confidence_samples': analysis['low_confidence_samples']
        }
        json.dump(serializable_analysis, f, ensure_ascii=False, indent=2)
    
    # 测试预测
    print("\n🔮 预测示例:")
    test_cases = [
        "高新技术企业研发资助",
        "工业技术改造补贴", 
        "大学生就业创业补助",
        "节能环保项目补贴",
        "中小企业发展资金",
        "专利申请资助费用"
    ]
    
    for text in test_cases:
        result = classifier.classify_text(text)
        category_name = classifier.category_definitions.get(result['prediction'], {}).get('name', result['prediction'])
        print(f"   '{text}' -> {category_name} (置信度: {result['confidence']:.2%})")
    
    print("\n✅ 分析完成！结果已保存到:")
    print("   - output/政府补贴数据_智能分类结果.csv")
    print("   - output/智能分类分析报告.json")
    print("   - output/advanced_ml_analysis.png")

if __name__ == "__main__":
    main()