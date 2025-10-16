# predict_new_data.py
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from prediction import EmotionPredictor


def predict_student_emotion():
    # 加载模型
    predictor = EmotionPredictor("data/models/emotion_model.pkl")

    # 准备新的学生数据（替换为实际数据）
    new_keyboard_data = pd.DataFrame({
        'total_keypresses': [180],
        'median_ikd': [0.22],
        'p95_ikd': [0.75],
        'mad': [0.18],
        'auto_correction_rate': [0.08],
        'space_rate': [0.11],
        'backspace_count': [25],
        'space_count': [20],
        'duration_sec': [65.0]
    })

    new_mouse_data = pd.DataFrame({
        'move_entropy': [2.9],
        'effective_path_ratio': [0.035],
        'avg_speed': [380.0],
        'acceleration_variance': [95000000000.0],
        'total_distance': [48000.0],
        'click_count': [45],
        'scroll_count': [25],
        'left_right_click_ratio': [45],
        'double_click_count': [30],
        'still_time_ratio': [0.0005],
        'click_interval_median': [1.4],
        'duration_sec': [125.0]
    })

    # 进行预测
    result = predictor.predict(new_keyboard_data, new_mouse_data)

    print("预测结果:")
    print(f"情绪: {result['emotion']}")
    print(f"置信度: {result['confidence']:.3f}")
    print("详细概率:")
    for emotion, prob in result['all_probabilities'].items():
        print(f"  {emotion}: {prob:.3f}")

    return result


if __name__ == "__main__":
    predict_student_emotion()