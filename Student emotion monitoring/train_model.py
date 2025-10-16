import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator


def main():
    print("学生情绪预测模型训练开始...")

    # 1. 数据预处理
    preprocessor = DataPreprocessor()
    emotion_df, keyboard_df, mouse_df = preprocessor.run_preprocessing()

    if emotion_df is None or keyboard_df is None or mouse_df is None:
        print("数据预处理失败，程序退出")
        return

    # 2. 特征工程
    feature_engineer = FeatureEngineer()
    features, labels, label_encoder = feature_engineer.run_feature_engineering(
        emotion_df, keyboard_df, mouse_df
    )

    if features.empty:
        print("特征工程失败，程序退出")
        return

    # 3. 模型训练
    trainer = ModelTrainer()
    best_model, best_model_name, results, X_test, y_test, y_pred = trainer.run_training(
        features, labels, label_encoder,
        feature_engineer.selector, feature_engineer.scaler,
        features.columns.tolist()
    )

    if best_model is None:
        print("模型训练失败，程序退出")
        return

    # 4. 模型评估
    evaluator = ModelEvaluator()
    evaluator.run_evaluation(
        results, best_model, best_model_name, X_test, y_test, y_pred,
        feature_engineer.get_feature_importance(), label_encoder
    )

    print("\n模型训练和评估完成!")


if __name__ == "__main__":
    main()