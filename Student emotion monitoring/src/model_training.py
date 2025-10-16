import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import yaml
import os
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.models = {}
        self.best_model = None
        self.best_model_name = None

    def split_data(self, features, labels):
        """划分训练集和测试集"""
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            stratify=labels
        )

        return X_train, X_test, y_train, y_test

    def initialize_models(self):
        """初始化模型"""
        model_configs = self.config['training']['models']

        for model_config in model_configs:
            name = model_config['name']
            if name == 'Random Forest':
                self.models[name] = RandomForestClassifier(random_state=self.config['model']['random_state'])
            elif name == 'Gradient Boosting':
                self.models[name] = GradientBoostingClassifier(random_state=self.config['model']['random_state'])
            elif name == 'SVM':
                self.models[name] = SVC(random_state=self.config['model']['random_state'], probability=True)
            elif name == 'Logistic Regression':
                self.models[name] = LogisticRegression(random_state=self.config['model']['random_state'])

    def train_models(self, X_train, y_train, X_test, y_test):
        """训练多个模型并比较性能"""
        results = {}

        for name, model in self.models.items():
            print(f"训练 {name}...")

            try:
                # 交叉验证
                cv_scores = cross_val_score(model, X_train, y_train, cv=self.config['model']['cv_folds'])

                # 训练模型
                model.fit(X_train, y_train)

                # 预测
                y_pred = model.predict(X_test)

                # 评估
                accuracy = accuracy_score(y_test, y_pred)

                results[name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': accuracy,
                    'predictions': y_pred
                }

                print(f"{name} - 交叉验证: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f}), "
                      f"测试集: {accuracy:.3f}")
            except Exception as e:
                print(f"{name} 训练失败: {e}")
                continue

        return results

    def optimize_best_model(self, results, X_train, y_train):
        """优化最佳模型"""
        if not results:
            print("错误: 没有可用的训练结果")
            return None

        # 选择最佳模型
        self.best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        best_model = results[self.best_model_name]['model']

        print(f"\n优化最佳模型: {self.best_model_name}")

        # 获取参数网格
        model_configs = self.config['training']['models']
        param_grid = None
        for config in model_configs:
            if config['name'] == self.best_model_name:
                param_grid = config['params']
                break

        if param_grid:
            try:
                # 网格搜索
                grid_search = GridSearchCV(
                    best_model, param_grid,
                    cv=self.config['model']['cv_folds'],
                    scoring='accuracy',
                    n_jobs=-1,
                    error_score='raise'
                )

                grid_search.fit(X_train, y_train)

                print(f"最佳参数: {grid_search.best_params_}")
                print(f"最佳交叉验证分数: {grid_search.best_score_:.3f}")

                self.best_model = grid_search.best_estimator_
            except Exception as e:
                print(f"网格搜索失败: {e}")
                print("使用默认参数模型")
                self.best_model = best_model
        else:
            self.best_model = best_model

        return self.best_model

    def save_model(self, model, label_encoder, feature_selector, scaler, selected_features):
        """保存模型和相关组件"""
        try:
            # 确保 selected_features 是列表格式
            if hasattr(selected_features, 'tolist'):
                selected_features_list = selected_features.tolist()
            else:
                selected_features_list = list(selected_features)

            model_data = {
                'model': model,
                'label_encoder': label_encoder,
                'feature_selector': feature_selector,
                'scaler': scaler,
                'selected_features': selected_features_list,  # 保存为列表
                'model_name': self.best_model_name
            }

            # 确保目录存在
            os.makedirs(os.path.dirname(self.config['model']['model_file']), exist_ok=True)

            joblib.dump(model_data, self.config['model']['model_file'])
            print(f"模型已保存到: {self.config['model']['model_file']}")

            # 验证保存的模型
            self._verify_saved_model()

        except Exception as e:
            print(f"模型保存失败: {e}")

    def _verify_saved_model(self):
        """验证保存的模型"""
        try:
            # 尝试重新加载模型以验证
            test_data = joblib.load(self.config['model']['model_file'])
            required_keys = ['model', 'label_encoder', 'feature_selector', 'scaler', 'selected_features']

            for key in required_keys:
                if key not in test_data:
                    print(f"警告: 保存的模型缺少键: {key}")
                elif test_data[key] is None:
                    print(f"警告: 保存的模型中的 {key} 为 None")

            print("模型验证成功")

        except Exception as e:
            print(f"模型验证失败: {e}")
    def run_training(self, features, labels, label_encoder, feature_selector, scaler, selected_features):
        """运行完整的模型训练流程"""
        print("开始模型训练...")

        if features.empty or len(labels) == 0:
            print("错误: 训练数据为空")
            return None, {}, None, None, None, None

        # 划分数据
        X_train, X_test, y_train, y_test = self.split_data(features, labels)

        if X_train.empty or X_test.empty:
            print("错误: 数据划分后训练集或测试集为空")
            return None, {}, None, None, None, None

        # 初始化模型
        self.initialize_models()

        # 训练和评估模型
        results = self.train_models(X_train, y_train, X_test, y_test)

        if not results:
            print("错误: 模型训练失败")
            return None, {}, None, None, None, None

        # 优化最佳模型
        best_model = self.optimize_best_model(results, X_train, y_train)

        if best_model is None:
            print("错误: 模型优化失败")
            return None, results, X_test, y_test, None, None

        # 最终评估
        y_pred_final = best_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred_final)
        print(f"\n最终模型测试集准确率: {final_accuracy:.3f}")

        # 保存模型
        self.save_model(best_model, label_encoder, feature_selector, scaler, selected_features)

        print("模型训练完成!")

        return best_model, self.best_model_name, results, X_test, y_test, y_pred_final