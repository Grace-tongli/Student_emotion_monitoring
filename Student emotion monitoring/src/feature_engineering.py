import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import yaml


class FeatureEngineer:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.scaler = StandardScaler()
        self.selector = SelectKBest(score_func=f_classif, k=self.config['features']['selected_features'])
        self.label_encoder = LabelEncoder()

        self.features_df = None
        self.labels = None

    def align_behavior_with_emotion(self, emotion_df, keyboard_df, mouse_df):
        """将行为数据与情绪标签对齐（按学生分组）"""
        features = []
        labels = []

        time_window = timedelta(minutes=self.config['features']['time_window_minutes'])

        # 按学生分组处理
        students = emotion_df['student_id'].unique()

        for student_id in students:
            student_emotion = emotion_df[emotion_df['student_id'] == student_id]
            student_keyboard = keyboard_df[keyboard_df['student_id'] == student_id]
            student_mouse = mouse_df[mouse_df['student_id'] == student_id]

            for _, emotion_row in student_emotion.iterrows():
                emotion_time = emotion_row['timestamp']
                emotion_label = emotion_row['emotion']

                # 计算时间窗口
                start_window = emotion_time - time_window
                end_window = emotion_time

                # 匹配键盘行为
                keyboard_mask = (
                        (student_keyboard['start_time'] >= start_window) &
                        (student_keyboard['end_time'] <= end_window)
                )
                keyboard_session = student_keyboard[keyboard_mask]

                # 匹配鼠标行为
                mouse_mask = (
                        (student_mouse['start_time'] >= start_window) &
                        (student_mouse['end_time'] <= end_window)
                )
                mouse_session = student_mouse[mouse_mask]

                if not keyboard_session.empty and not mouse_session.empty:
                    # 提取键盘特征
                    keyboard_features = self._extract_keyboard_features(keyboard_session)

                    # 提取鼠标特征
                    mouse_features = self._extract_mouse_features(mouse_session)

                    # 添加学生ID作为特征（可选）
                    # keyboard_features['student_id_encoded'] = hash(student_id) % 1000  # 简单的编码

                    # 合并特征
                    combined_features = {**keyboard_features, **mouse_features}
                    features.append(combined_features)
                    labels.append(emotion_label)

        if not features:
            print("警告: 没有找到匹配的行为和情绪数据")
            return pd.DataFrame(), []

        return pd.DataFrame(features), labels

    def _extract_keyboard_features(self, keyboard_session):
        """提取键盘特征"""
        return {
            'k_total_keypresses': keyboard_session['total_keypresses'].mean(),
            'k_median_ikd': keyboard_session['median_ikd'].mean(),
            'k_p95_ikd': keyboard_session['p95_ikd'].mean(),
            'k_mad': keyboard_session['mad'].mean(),
            'k_auto_correction_rate': keyboard_session['auto_correction_rate'].mean(),
            'k_space_rate': keyboard_session['space_rate'].mean(),
            'k_backspace_count': keyboard_session['backspace_count'].sum(),
            'k_space_count': keyboard_session['space_count'].sum(),
            'k_duration_sec': keyboard_session['duration_sec'].sum(),
            'k_keypress_per_sec': keyboard_session['total_keypresses'].sum() /
                                  keyboard_session['duration_sec'].sum() if keyboard_session[
                                                                                'duration_sec'].sum() > 0 else 0
        }

    def _extract_mouse_features(self, mouse_session):
        """提取鼠标特征"""
        return {
            'm_move_entropy': mouse_session['move_entropy'].mean(),
            'm_effective_path_ratio': mouse_session['effective_path_ratio'].mean(),
            'm_avg_speed': mouse_session['avg_speed'].mean(),
            'm_acceleration_variance': np.log1p(mouse_session['acceleration_variance'].mean()),
            'm_total_distance': mouse_session['total_distance'].mean(),
            'm_click_count': mouse_session['click_count'].sum(),
            'm_scroll_count': mouse_session['scroll_count'].sum(),
            'm_left_right_click_ratio': mouse_session['left_right_click_ratio'].mean(),
            'm_double_click_count': mouse_session['double_click_count'].sum(),
            'm_still_time_ratio': mouse_session['still_time_ratio'].mean(),
            'm_click_interval_median': mouse_session['click_interval_median'].mean(),
            'm_duration_sec': mouse_session['duration_sec'].sum(),
            'm_clicks_per_sec': mouse_session['click_count'].sum() /
                                mouse_session['duration_sec'].sum() if mouse_session['duration_sec'].sum() > 0 else 0
        }

    def preprocess_features(self, features_df, labels):
        """特征预处理"""
        if features_df.empty:
            print("错误: 特征数据为空")
            return pd.DataFrame(), []

        # 处理缺失值
        features_df = features_df.fillna(features_df.mean())

        # 标签编码
        self.labels = self.label_encoder.fit_transform(labels)

        # 特征标准化
        try:
            self.features_df = pd.DataFrame(
                self.scaler.fit_transform(features_df),
                columns=features_df.columns
            )
        except Exception as e:
            print(f"特征标准化失败: {e}")
            # 如果标准化失败，使用原始特征
            self.features_df = features_df.copy()

        return self.features_df, self.labels

    def select_features(self, features_df, labels):
        """特征选择"""
        if features_df.empty or len(labels) == 0:
            print("错误: 无法进行特征选择，数据为空")
            return pd.DataFrame()

        try:
            selected_features = self.selector.fit_transform(features_df, labels)
            selected_feature_names = features_df.columns[self.selector.get_support()]

            selected_df = pd.DataFrame(selected_features, columns=selected_feature_names)

            print(f"特征选择完成，选择了 {len(selected_feature_names)} 个特征")
            print("选中的特征:", selected_feature_names.tolist())

            return selected_df
        except Exception as e:
            print(f"特征选择失败: {e}")
            # 如果特征选择失败，返回所有特征
            return features_df

    def get_feature_importance(self):
        """获取特征重要性"""
        if hasattr(self.selector, 'scores_') and self.features_df is not None:
            return pd.DataFrame({
                'feature': self.features_df.columns,
                'importance': self.selector.scores_
            }).sort_values('importance', ascending=False)
        return None

    def run_feature_engineering(self, emotion_df, keyboard_df, mouse_df):
        """运行完整的特征工程流程"""
        print("开始特征工程...")

        if emotion_df is None or keyboard_df is None or mouse_df is None:
            print("错误: 输入数据为空")
            return pd.DataFrame(), [], self.label_encoder

        # 对齐数据
        features_df, labels = self.align_behavior_with_emotion(emotion_df, keyboard_df, mouse_df)

        if features_df.empty:
            print("错误: 特征工程后没有有效数据")
            return pd.DataFrame(), [], self.label_encoder

        print(f"生成 {len(features_df)} 个样本")

        # 预处理特征
        features_df, encoded_labels = self.preprocess_features(features_df, labels)

        if features_df.empty:
            print("错误: 特征预处理后没有有效数据")
            return pd.DataFrame(), [], self.label_encoder

        # 特征选择
        selected_features_df = self.select_features(features_df, encoded_labels)

        print("特征工程完成!")

        return selected_features_df, encoded_labels, self.label_encoder