#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于最新机器学习方法的政府补贴信息分类系统
支持BERT、XGBoost、LSTM等多种先进模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
import re
import warnings
from collections import Counter
from typing import List, Dict, Tuple, Any

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

# Deep Learning (条件导入)
try:
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("深度学习库未安装，将使用传统机器学习方法")

# 其他工具
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SubsidyTextClassifier:
    """政府补贴文本分类器"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = None
        self.models = {}
        self.feature_names = []
        
        # 定义分类标签映射
        self.category_mapping = {
            'R&D_Innovation': '研发创新',
            'Industrial_Equipment': '工业设备', 
            'Employment': '就业相关',
            'Environment': '环境保护',
            'General_Business': '一般商业',
            'Other': '其他',
            'Unknown': '未知'
        }
        
        # 改进的关键词字典
        self.enhanced_keywords = {
            'R&D_Innovation': [
                '创新', '研发', '专利', '科技', '技术', '知识产权', '研究', '开发', 
                '科学', '发明', '高新', '智能', '数字化', '信息化', '人工智能',
                '大数据', '云计算', '物联网', '区块链', '5G', 'AI', '算法',
                '软件', '芯片', '半导体', '新材料', '生物技术', '医药'
            ],
            'Industrial_Equipment': [
                '工业', '设备', '技改', '改造', '升级', '转型', '制造', '生产线',
                '机械', '装备', '产业化', '自动化', '机器人', '智能制造',
                '工厂', '车间', '流水线', '加工', '组装', '焊接', '冲压'
            ],
            'Employment': [
                '就业', '招聘', '实习', '培训', '稳岗', '用工', '劳动', '职业',
                '毕业生', '扩岗', '人才', '岗位', '员工', '技能', '见习',
                '残疾人', '高校', '大学生', '职工', '社保', '工资'
            ],
            'Environment': [
                '节能', '环保', '清洁', '减排', '污染', '治理', '绿色', '循环',
                '生态', '废料', '排放', '环境', '废水', '废气', '固废',
                '低碳', '新能源', '太阳能', '风能', '节水', '除尘'
            ],
            'General_Business': [
                '经营', '出口', '品牌', '税收', '发展', '市场', '贸易', '营业',
                '商务', '财政', '奖励', '扶持', '补贴', '补助', '资金',
                '投资', '融资', '上市', '挂牌', '产业园'
            ],
            'Other': [
                '搬迁', '拆迁', '安全', '质量', '标准', '认证', '检测',
                '文化', '教育', '医疗', '旅游', '农业', '金融'
            ]
        }

    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if pd.isna(text):
            return ""
        
        # 转换为字符串并清理
        text = str(text).strip()
        
        # 去除特殊字符但保留中文、数字、字母
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)
        
        # 去除多余空格
        text = ' '.join(text.split())
        
        return text

    def extract_manual_features(self, texts: List[str]) -> pd.DataFrame:
        """提取人工特征"""
        features = []
        
        for text in texts:
            feature_dict = {}
            
            # 文本长度特征
            feature_dict['text_length'] = len(text)
            feature_dict['word_count'] = len(text.split())
            
            # 关键词特征
            for category, keywords in self.enhanced_keywords.items():
                count = sum(1 for keyword in keywords if keyword in text)
                feature_dict[f'{category}_keywords'] = count
                feature_dict[f'{category}_density'] = count / max(len(text.split()), 1)
            
            # 数字特征
            feature_dict['has_numbers'] = int(bool(re.search(r'\d', text)))
            feature_dict['number_count'] = len(re.findall(r'\d+', text))
            
            # 特殊词汇特征
            feature_dict['has_project'] = int('项目' in text)
            feature_dict['has_fund'] = int(any(word in text for word in ['资金', '基金', '专项']))
            feature_dict['has_award'] = int(any(word in text for word in ['奖励', '奖金', '表彰']))
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)

    def create_tfidf_features(self, texts: List[str], max_features: int = 1000) -> np.ndarray:
        """创建TF-IDF特征"""
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                stop_words=None  # 中文没有预定义停用词
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_matrix.toarray()

    def prepare_features(self, df: pd.DataFrame, text_column: str = 'Fn05601') -> Tuple[np.ndarray, List[str]]:
        """准备特征"""
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in df[text_column]]
        
        # 提取人工特征
        manual_features = self.extract_manual_features(processed_texts)
        
        # 提取TF-IDF特征
        tfidf_features = self.create_tfidf_features(processed_texts)
        
        # 合并特征
        combined_features = np.hstack([manual_features.values, tfidf_features])
        
        # 特征名称
        feature_names = list(manual_features.columns) + [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        
        return combined_features, feature_names

    def classify_by_rules(self, text: str) -> str:
        """基于规则的分类（作为基线）"""
        if pd.isna(text):
            return 'Unknown'
        
        text = str(text).lower()
        scores = {}
        
        for category, keywords in self.enhanced_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[category] = score
        
        if max(scores.values()) == 0:
            return 'Unknown'
        
        return max(scores, key=scores.get)

    def train_traditional_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练传统机器学习模型"""
        print("🚀 训练传统机器学习模型...")
        
        # Random Forest
        print("   训练Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        # XGBoost
        print("   训练XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # LightGBM
        print("   训练LightGBM...")
        lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_train, y_train)
        self.models['lightgbm'] = lgb_model
        
        # Logistic Regression
        print("   训练Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr
        
        # SVM (小数据集上)
        if X_train.shape[0] < 5000:
            print("   训练SVM...")
            svm = SVC(kernel='rbf', probability=True, random_state=42)
            svm.fit(X_train, y_train)
            self.models['svm'] = svm

    def create_ensemble_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """创建集成模型"""
        print("🎯 创建集成模型...")
        
        # 准备基础模型
        estimators = []
        for name, model in self.models.items():
            if name != 'ensemble':
                estimators.append((name, model))
        
        # 创建投票分类器
        if estimators:
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            ensemble.fit(X_train, y_train)
            self.models['ensemble'] = ensemble

    def train(self, df: pd.DataFrame, target_column: str = 'Test', text_column: str = 'Fn05601'):
        """训练模型"""
        print("🎓 开始训练分类模型...")
        
        # 准备数据
        X, feature_names = self.prepare_features(df, text_column)
        y = df[target_column].values
        
        self.feature_names = feature_names
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # 训练传统模型
        self.train_traditional_models(X_train, y_train)
        
        # 创建集成模型
        self.create_ensemble_model(X_train, y_train)
        
        # 评估模型
        self.evaluate_models(X_test, y_test)
        
        return X_train, X_test, y_train, y_test

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """评估模型性能"""
        print("\n📊 模型性能评估:")
        print("=" * 60)
        
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            print(f"{name:20} | 准确率: {accuracy:.4f}")
        
        # 找到最佳模型
        best_model_name = max(results, key=results.get)
        print(f"\n🏆 最佳模型: {best_model_name} (准确率: {results[best_model_name]:.4f})")
        
        # 详细报告
        best_model = self.models[best_model_name]
        y_pred = best_model.predict(X_test)
        
        print(f"\n📋 {best_model_name} 详细分类报告:")
        print("-" * 50)
        class_names = [str(x) for x in self.label_encoder.classes_]
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        return results

    def predict(self, texts: List[str], model_name: str = 'ensemble') -> List[str]:
        """预测新文本"""
        # 创建临时DataFrame
        temp_df = pd.DataFrame({self.feature_names[0]: texts})
        
        # 准备特征
        X, _ = self.prepare_features(temp_df, self.feature_names[0])
        
        # 预测
        if model_name in self.models:
            y_pred_encoded = self.models[model_name].predict(X)
            predictions = self.label_encoder.inverse_transform(y_pred_encoded)
        else:
            # 使用基于规则的分类作为后备
            predictions = [self.classify_by_rules(text) for text in texts]
        
        return predictions

    def get_feature_importance(self, model_name: str = 'random_forest', top_n: int = 20):
        """获取特征重要性"""
        if model_name not in self.models:
            print(f"模型 {model_name} 不存在")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print(f"模型 {model_name} 不支持特征重要性分析")
            return None
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 {model_name} 前{top_n}个重要特征:")
        print("-" * 50)
        print(feature_importance_df.head(top_n))
        
        return feature_importance_df

    def visualize_results(self, X_test: np.ndarray, y_test: np.ndarray, model_name: str = 'ensemble'):
        """可视化结果"""
        if model_name not in self.models:
            model_name = 'random_forest'
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        # 混淆矩阵
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'{model_name} 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        # 特征重要性（如果支持）
        plt.subplot(1, 2, 2)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:]  # 前10个重要特征
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
            plt.title('特征重要性 (前10)')
        else:
            plt.text(0.5, 0.5, '该模型不支持\n特征重要性分析', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('特征重要性')
        
        plt.tight_layout()
        plt.savefig('output/6_ml_classification_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    print("🤖 政府补贴信息智能分类系统")
    print("=" * 50)
    
    # 加载数据
    print("📁 加载数据...")
    df = pd.read_csv('output/3_政府补贴数据_样本.csv')
    df.columns = ['Stkcd', 'Year', 'Fn05601', 'Fn05602', '合计', '政府补贴', 'Sum', 'test', 'Test']
    
    print(f"   数据形状: {df.shape}")
    print(f"   标签分布:")
    print(f"   - test: {df['test'].value_counts().to_dict()}")
    print(f"   - Test: {df['Test'].value_counts().to_dict()}")
    
    # 创建分类器
    classifier = SubsidyTextClassifier()
    
    # 训练模型 (使用Test列作为标签)
    X_train, X_test, y_train, y_test = classifier.train(df, target_column='Test')
    
    # 特征重要性分析
    classifier.get_feature_importance('random_forest')
    classifier.get_feature_importance('xgboost')
    
    # 可视化结果
    classifier.visualize_results(X_test, y_test)
    
    # 测试预测
    print("\n🔮 预测示例:")
    test_texts = [
        "高新技术企业研发资助",
        "工业技术改造补贴",
        "大学生就业创业补助",
        "节能环保项目补贴"
    ]
    
    predictions = classifier.predict(test_texts)
    for text, pred in zip(test_texts, predictions):
        print(f"   '{text}' -> {pred}")
    
    # 保存分类结果
    print("\n💾 保存分类结果...")
    
    # 对整个数据集进行预测
    all_texts = df['Fn05601'].tolist()
    all_predictions = classifier.predict(all_texts)
    
    # 添加预测结果到数据框
    df['ml_prediction'] = all_predictions
    df['ml_prediction_cn'] = [classifier.category_mapping.get(pred, pred) for pred in all_predictions]
    
    # 保存结果
    df.to_csv('output/6_政府补贴数据_ML分类结果.csv', index=False)
    
    print("✅ 分析完成！结果已保存到:")
    print("   - output/6_政府补贴数据_ML分类结果.csv")
    print("   - output/6_ml_classification_results.png")

if __name__ == "__main__":
    main()