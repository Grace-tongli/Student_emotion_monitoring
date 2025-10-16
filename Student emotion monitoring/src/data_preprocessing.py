import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import os
import glob
import chardet


class DataPreprocessor:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.keyboard_df = None
        self.mouse_df = None
        self.emotion_df = None

    def detect_encoding(self, file_path):
        """检测文件编码"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                print(f"检测到文件 {os.path.basename(file_path)} 编码: {encoding} (置信度: {confidence:.2f})")
                return encoding
        except Exception as e:
            print(f"编码检测失败 {file_path}: {e}")
            return 'utf-8'  # 默认使用utf-8

    def load_csv_with_encoding(self, file_path):
        """使用正确的编码加载CSV文件"""
        encodings_to_try = ['gbk', 'gb2312', 'utf-8', 'latin-1']

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用 {encoding} 编码加载 {os.path.basename(file_path)}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"使用 {encoding} 编码加载失败: {e}")
                continue

        # 如果常用编码都失败，尝试自动检测
        try:
            detected_encoding = self.detect_encoding(file_path)
            if detected_encoding:
                df = pd.read_csv(file_path, encoding=detected_encoding)
                print(f"使用检测到的编码 {detected_encoding} 成功加载 {os.path.basename(file_path)}")
                return df
        except Exception as e:
            print(f"使用检测编码加载失败: {e}")

        print(f"所有编码尝试失败: {file_path}")
        return None

    def load_data(self):
        """加载所有学生的原始数据"""
        try:
            data_dir = self.config['data']['data_directory']

            # 检查目录是否存在
            if not os.path.exists(data_dir):
                print(f"错误: 数据目录不存在: {data_dir}")
                return

            # 加载所有键盘数据文件
            keyboard_files = glob.glob(os.path.join(data_dir, self.config['data']['keyboard_pattern']))
            print(f"找到键盘文件: {keyboard_files}")

            keyboard_dfs = []
            for file in keyboard_files:
                df = self.load_csv_with_encoding(file)
                if df is not None:
                    # 添加学生ID列
                    student_id = os.path.basename(file).split('_')[0]
                    df['student_id'] = student_id
                    keyboard_dfs.append(df)
                else:
                    print(f"跳过无法加载的文件: {file}")

            if keyboard_dfs:
                self.keyboard_df = pd.concat(keyboard_dfs, ignore_index=True)
            else:
                self.keyboard_df = pd.DataFrame()
                print("警告: 没有成功加载任何键盘数据文件")

            # 加载所有鼠标数据文件
            mouse_files = glob.glob(os.path.join(data_dir, self.config['data']['mouse_pattern']))
            print(f"找到鼠标文件: {mouse_files}")

            mouse_dfs = []
            for file in mouse_files:
                df = self.load_csv_with_encoding(file)
                if df is not None:
                    student_id = os.path.basename(file).split('_')[0]
                    df['student_id'] = student_id
                    mouse_dfs.append(df)
                else:
                    print(f"跳过无法加载的文件: {file}")

            if mouse_dfs:
                self.mouse_df = pd.concat(mouse_dfs, ignore_index=True)
            else:
                self.mouse_df = pd.DataFrame()
                print("警告: 没有成功加载任何鼠标数据文件")

            # 加载所有情绪数据文件
            emotion_files = glob.glob(os.path.join(data_dir, self.config['data']['emotion_pattern']))
            print(f"找到情绪文件: {emotion_files}")

            emotion_dfs = []
            for file in emotion_files:
                df = self.load_csv_with_encoding(file)
                if df is not None:
                    student_id = os.path.basename(file).split('_')[0]
                    df['student_id'] = student_id
                    emotion_dfs.append(df)
                else:
                    print(f"跳过无法加载的文件: {file}")

            if emotion_dfs:
                self.emotion_df = pd.concat(emotion_dfs, ignore_index=True)
            else:
                self.emotion_df = pd.DataFrame()
                print("警告: 没有成功加载任何情绪数据文件")

            print(f"数据加载完成:")
            print(f"  键盘数据: {len(self.keyboard_df)} 条记录")
            print(f"  鼠标数据: {len(self.mouse_df)} 条记录")
            print(f"  情绪数据: {len(self.emotion_df)} 条记录")

        except Exception as e:
            print(f"数据加载失败: {e}")
            # 创建空DataFrame避免后续错误
            self.keyboard_df = pd.DataFrame()
            self.mouse_df = pd.DataFrame()
            self.emotion_df = pd.DataFrame()

    def clean_emotion_data(self):
        """清洗情绪数据"""
        if self.emotion_df.empty:
            print("情绪数据为空，跳过清洗")
            return

        # 删除空行
        self.emotion_df = self.emotion_df.dropna(subset=['emotion', 'timestamp'])

        # 转换时间格式 - 处理多种可能的时间格式
        self.emotion_df['timestamp'] = pd.to_datetime(
            self.emotion_df['timestamp'],
            format='%Y/%m/%d %H:%M',
            errors='coerce'
        )

        # 如果上面的格式转换失败，尝试其他格式
        if self.emotion_df['timestamp'].isna().any():
            self.emotion_df['timestamp'] = pd.to_datetime(
                self.emotion_df['timestamp'],
                errors='coerce'
            )

        # 删除转换失败的行
        original_count = len(self.emotion_df)
        self.emotion_df = self.emotion_df.dropna(subset=['timestamp'])
        removed_count = original_count - len(self.emotion_df)
        if removed_count > 0:
            print(f"删除了 {removed_count} 条时间格式无效的情绪记录")

        # 排序
        self.emotion_df = self.emotion_df.sort_values(['student_id', 'timestamp'])

        print(f"情绪数据清洗完成，剩余 {len(self.emotion_df)} 条记录")

    def clean_behavior_data(self):
        """清洗行为数据"""
        if self.keyboard_df.empty or self.mouse_df.empty:
            print("行为数据为空，跳过清洗")
            return

        # 键盘数据清洗
        self.keyboard_df['start_time'] = pd.to_datetime(self.keyboard_df['start_time'], errors='coerce')
        self.keyboard_df['end_time'] = pd.to_datetime(self.keyboard_df['end_time'], errors='coerce')

        # 删除时间转换失败的行
        original_keyboard_count = len(self.keyboard_df)
        self.keyboard_df = self.keyboard_df.dropna(subset=['start_time', 'end_time'])
        removed_keyboard = original_keyboard_count - len(self.keyboard_df)

        # 删除异常记录（持续时间为0或负数的记录）
        self.keyboard_df = self.keyboard_df[self.keyboard_df['duration_sec'] > 0]

        # 鼠标数据清洗
        self.mouse_df['start_time'] = pd.to_datetime(self.mouse_df['start_time'], errors='coerce')
        self.mouse_df['end_time'] = pd.to_datetime(self.mouse_df['end_time'], errors='coerce')

        original_mouse_count = len(self.mouse_df)
        self.mouse_df = self.mouse_df.dropna(subset=['start_time', 'end_time'])
        removed_mouse = original_mouse_count - len(self.mouse_df)

        self.mouse_df = self.mouse_df[self.mouse_df['duration_sec'] > 0]

        print(f"键盘数据清洗完成，删除了 {removed_keyboard} 条无效记录，剩余 {len(self.keyboard_df)} 条记录")
        print(f"鼠标数据清洗完成，删除了 {removed_mouse} 条无效记录，剩余 {len(self.mouse_df)} 条记录")

    def remove_duplicates(self):
        """删除重复数据"""
        if not self.keyboard_df.empty:
            original_count = len(self.keyboard_df)
            self.keyboard_df = self.keyboard_df.drop_duplicates()
            print(f"键盘数据: 删除了 {original_count - len(self.keyboard_df)} 条重复记录")

        if not self.mouse_df.empty:
            original_count = len(self.mouse_df)
            self.mouse_df = self.mouse_df.drop_duplicates()
            print(f"鼠标数据: 删除了 {original_count - len(self.mouse_df)} 条重复记录")

        if not self.emotion_df.empty:
            original_count = len(self.emotion_df)
            self.emotion_df = self.emotion_df.drop_duplicates()
            print(f"情绪数据: 删除了 {original_count - len(self.emotion_df)} 条重复记录")

    def get_clean_data(self):
        """获取清洗后的数据"""
        return self.emotion_df, self.keyboard_df, self.mouse_df

    def run_preprocessing(self):
        """运行完整的数据预处理流程"""
        print("开始数据预处理...")
        self.load_data()

        if self.emotion_df.empty or self.keyboard_df.empty or self.mouse_df.empty:
            print("错误: 缺少必要的数据文件")
            print("请确保在 data/raw/ 目录下有以下格式的文件:")
            print("  [学生编号]_keyboard_performance.csv")
            print("  [学生编号]_mouse_performance.csv")
            print("  [学生编号]_emotion_performance.csv")

            # 检查目录内容
            data_dir = self.config['data']['data_directory']
            if os.path.exists(data_dir):
                print(f"\ndata/raw/ 目录中的文件:")
                for file in os.listdir(data_dir):
                    print(f"  - {file}")
            else:
                print(f"\n目录 {data_dir} 不存在")

            return None, None, None

        self.clean_emotion_data()
        self.clean_behavior_data()
        self.remove_duplicates()

        # 检查清洗后是否还有数据
        if self.emotion_df.empty or self.keyboard_df.empty or self.mouse_df.empty:
            print("错误: 数据清洗后没有有效数据")
            return None, None, None

        print("数据预处理完成!")

        return self.get_clean_data()