import pandas as pd
import numpy as np
import joblib
import yaml
import os

class EmotionPredictor:
    def __init__(self, model_path=None, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        if model_path is None:
            model_path = self.config['model']['model_file']

        self.load_model(model_path)

    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_selector = model_data['feature_selector']
            self.scaler = model_data['scaler']
            self.selected_features = model_data['selected_features']
            self.model_name = model_data.get('model_name', 'Unknown')

            print(f"模型加载成功: {self.model_name}")

            # 验证加载的组件
            self._validate_loaded_components()

        except Exception as e:
            print(f"模型加载失败: {e}")
            # 提供更详细的错误信息
            if os.path.exists(model_path):
                print(f"模型文件存在，但格式可能不正确: {model_path}")
            else:
                print(f"模型文件不存在: {model_path}")
            raise

    def _validate_loaded_components(self):
        """验证加载的模型组件"""
        components_to_check = [
            ('model', self.model),
            ('label_encoder', self.label_encoder),
            ('feature_selector', self.feature_selector),
            ('scaler', self.scaler),
            ('selected_features', self.selected_features)
        ]

        for name, component in components_to_check:
            if component is None:
                print(f"警告: {name} 为 None")
            else:
                print(f"✓ {name} 加载成功")

        # 特别检查 label_encoder 的 classes_
        if hasattr(self.label_encoder, 'classes_'):
            classes = self.label_encoder.classes_
            print(f"标签类别: {classes} (类型: {type(classes)})")
        else:
            print("警告: label_encoder 没有 classes_ 属性")
    def preprocess_new_data(self, keyboard_data, mouse_data):
        """预处理新数据"""
        # 提取键盘特征
        keyboard_features = self._extract_keyboard_features(keyboard_data)

        # 提取鼠标特征
        mouse_features = self._extract_mouse_features(mouse_data)

        # 合并特征
        features = {**keyboard_features, **mouse_features}
        features_df = pd.DataFrame([features])

        # 确保所有特征都存在
        for feature in self.selected_features:
            if feature not in features_df.columns:
                features_df[feature] = 0

        # 重新排列列顺序以匹配训练数据
        features_df = features_df[self.selected_features]

        # 处理缺失值
        features_df = features_df.fillna(0)

        # 特征缩放
        scaled_features = self.scaler.transform(features_df)

        # 特征选择
        selected_features = self.feature_selector.transform(scaled_features)

        return selected_features

    def _extract_keyboard_features(self, keyboard_data):
        """提取键盘特征（与训练时一致）"""
        return {
            'k_total_keypresses': keyboard_data['total_keypresses'].mean(),
            'k_median_ikd': keyboard_data['median_ikd'].mean(),
            'k_p95_ikd': keyboard_data['p95_ikd'].mean(),
            'k_mad': keyboard_data['mad'].mean(),
            'k_auto_correction_rate': keyboard_data['auto_correction_rate'].mean(),
            'k_space_rate': keyboard_data['space_rate'].mean(),
            'k_backspace_count': keyboard_data['backspace_count'].sum(),
            'k_space_count': keyboard_data['space_count'].sum(),
            'k_duration_sec': keyboard_data['duration_sec'].sum(),
            'k_keypress_per_sec': keyboard_data['total_keypresses'].sum() /
                                  keyboard_data['duration_sec'].sum() if keyboard_data['duration_sec'].sum() > 0 else 0
        }

    def _extract_mouse_features(self, mouse_data):
        """提取鼠标特征（与训练时一致）"""
        return {
            'm_move_entropy': mouse_data['move_entropy'].mean(),
            'm_effective_path_ratio': mouse_data['effective_path_ratio'].mean(),
            'm_avg_speed': mouse_data['avg_speed'].mean(),
            'm_acceleration_variance': np.log1p(mouse_data['acceleration_variance'].mean()),
            'm_total_distance': mouse_data['total_distance'].mean(),
            'm_click_count': mouse_data['click_count'].sum(),
            'm_scroll_count': mouse_data['scroll_count'].sum(),
            'm_left_right_click_ratio': mouse_data['left_right_click_ratio'].mean(),
            'm_double_click_count': mouse_data['double_click_count'].sum(),
            'm_still_time_ratio': mouse_data['still_time_ratio'].mean(),
            'm_click_interval_median': mouse_data['click_interval_median'].mean(),
            'm_duration_sec': mouse_data['duration_sec'].sum(),
            'm_clicks_per_sec': mouse_data['click_count'].sum() /
                                mouse_data['duration_sec'].sum() if mouse_data['duration_sec'].sum() > 0 else 0
        }

    def predict(self, keyboard_data, mouse_data):
        """预测情绪"""
        try:
            # 预处理数据
            processed_features = self.preprocess_new_data(keyboard_data, mouse_data)

            # 预测
            prediction = self.model.predict(processed_features)[0]
            probability = self.model.predict_proba(processed_features)[0]

            # 解码标签
            emotion = self.label_encoder.inverse_transform([prediction])[0]

            # 创建概率字典
            prob_dict = {
                self.label_encoder.inverse_transform([i])[0]: prob
                for i, prob in enumerate(probability)
            }

            return {
                'emotion': emotion,
                'probability': prob_dict,
                'confidence': max(probability),
                'all_probabilities': prob_dict
            }

        except Exception as e:
            print(f"预测失败: {e}")
            return None

    def batch_predict(self, keyboard_data_list, mouse_data_list):
        """批量预测"""
        predictions = []

        for keyboard_data, mouse_data in zip(keyboard_data_list, mouse_data_list):
            prediction = self.predict(keyboard_data, mouse_data)
            if prediction:
                predictions.append(prediction)

        return predictions

    def get_model_info(self):
        """获取模型信息"""
        try:
            # 确保 classes 是列表格式
            if hasattr(self.label_encoder, 'classes_'):
                classes = self.label_encoder.classes_
                # 如果是 numpy 数组，转换为列表；如果已经是列表，直接使用
                if hasattr(classes, 'tolist'):
                    classes_list = classes.tolist()
                else:
                    classes_list = list(classes)
            else:
                classes_list = []

            # 确保 features_used 是列表格式
            if hasattr(self.selected_features, 'tolist'):
                features_used = self.selected_features.tolist()
            else:
                features_used = list(self.selected_features)

            return {
                'model_name': getattr(self, 'model_name', 'Unknown'),
                'model_type': type(self.model).__name__,
                'classes': classes_list,
                'features_used': features_used,
                'num_features': len(features_used)
            }
        except Exception as e:
            print(f"获取模型信息失败: {e}")
            return {
                'model_name': 'Unknown',
                'model_type': 'Unknown',
                'classes': [],
                'features_used': [],
                'num_features': 0
            }

