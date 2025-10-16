import warnings
import sys
import os
import pandas as pd
import numpy as np

# 设置环境变量和警告过滤
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="joblib")

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """主函数 - 学生情绪预测系统"""
    print("学生情绪预测系统")
    print("=" * 30)

    try:
        # 尝试导入预测模块
        from prediction import EmotionPredictor

        # 模型文件路径
        model_path = "data/models/emotion_model.pkl"

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print("💡 请先运行 train_model.py 训练模型")
            print("\n运行训练命令:")
            print("  python train_model.py")
            return

        print(f"✅ 找到模型文件: {model_path}")

        # 加载模型
        print("🔄 正在加载模型...")
        predictor = EmotionPredictor(model_path)

        # 获取模型信息
        model_info = predictor.get_model_info()

        print("\n📊 模型信息:")
        print(f"  • 模型类型: {model_info['model_name']}")
        print(f"  • 算法: {model_info['model_type']}")
        print(f"  • 使用的特征数: {model_info['num_features']}")
        print(f"  • 可预测的情绪类别: {model_info['classes']}")

        # 显示使用的特征
        if model_info['features_used']:
            print(f"  • 使用的特征: {', '.join(model_info['features_used'][:5])}..." if len(
                model_info['features_used']) > 5 else f"  • 使用的特征: {', '.join(model_info['features_used'])}")



        print("\n🎉 系统准备就绪!")
        print("💡 你可以使用 predictor.predict(keyboard_data, mouse_data) 进行预测")

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("💡 请确保 src 目录存在且包含必要的模块")
        print(
            "   需要的模块: prediction.py, data_preprocessing.py, feature_engineering.py, model_training.py, model_evaluation.py")

    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("\n🔧 故障排除步骤:")
        print("  1. 确保已运行: python train_model.py")
        print("  2. 检查 data/models/emotion_model.pkl 文件是否存在")
        print("  3. 如果模型文件损坏，删除并重新训练")
        print("  4. 检查控制台输出是否有其他错误信息")



def check_system_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")

    # 检查必要的目录
    required_dirs = ['data/raw', 'data/processed', 'data/models', 'src', 'config']
    missing_dirs = []

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"❌ 缺少必要的目录: {', '.join(missing_dirs)}")
        print("💡 请运行 create_dirs.py 创建项目目录结构")
        return False

    # 检查必要的Python包
    required_packages = ['pandas', 'numpy', 'sklearn', 'joblib', 'yaml']
    missing_packages = []

    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ 缺少必要的Python包: {', '.join(missing_packages)}")
        print("💡 请运行: pip install -r requirements.txt")
        return False

    print("✅ 系统要求检查通过")
    return True


if __name__ == "__main__":
    # 检查系统要求
    if check_system_requirements():
        # 运行主程序
        main()
    else:
        print("\n🚫 系统要求未满足，程序退出")
        print("💡 请解决上述问题后重新运行程序")