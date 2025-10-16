import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


class ModelEvaluator:
    def __init__(self):
        plt.style.use('seaborn-v0_8')

    def plot_confusion_matrix(self, y_true, y_pred, class_names, title='Confusion Matrix'):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, feature_importance_df, top_n=15):
        """绘制特征重要性"""
        if feature_importance_df is None or feature_importance_df.empty:
            print("没有可用的特征重要性数据")
            return

        top_features = feature_importance_df.head(top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self, results):
        """绘制模型比较图"""
        if not results:
            print("没有可用的模型结果")
            return

        model_names = list(results.keys())
        cv_scores = [results[name]['cv_mean'] for name in model_names]
        test_scores = [results[name]['test_accuracy'] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width / 2, cv_scores, width, label='CV Score', alpha=0.7)
        rects2 = ax.bar(x + width / 2, test_scores, width, label='Test Score', alpha=0.7)

        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Comparison: Cross-validation vs Test Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()

        # 添加数值标签
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, model, X_test, y_test, class_names):
        """绘制ROC曲线（适用于多分类）"""
        if not hasattr(model, 'predict_proba'):
            print("该模型不支持概率预测，无法绘制ROC曲线")
            return

        try:
            y_score = model.predict_proba(X_test)

            # 二值化标签
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

            # 计算每个类的ROC曲线和AUC
            fpr = {}
            tpr = {}
            roc_auc = {}
            n_classes = len(class_names)

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # 绘制所有类的ROC曲线
            plt.figure(figsize=(10, 8))
            colors = ['blue', 'red', 'green', 'orange', 'purple']

            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multi-class ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"绘制ROC曲线失败: {e}")

    def generate_report(self, results, best_model_name, y_test, y_pred, class_names):
        """生成完整的评估报告"""
        print("=" * 50)
        print("模型评估报告")
        print("=" * 50)

        if not results:
            print("没有可用的模型结果")
            return

        # 模型比较
        print("\n1. 模型性能比较:")
        for name, result in results.items():
            mark = "★" if name == best_model_name else " "
            print(f"{mark} {name}: CV={result['cv_mean']:.3f} (±{result['cv_std'] * 2:.3f}), "
                  f"Test={result['test_accuracy']:.3f}")

        # 最佳模型详细报告
        if best_model_name in results:
            best_result = results[best_model_name]
            print(f"\n2. 最佳模型 ({best_model_name}) 详细报告:")
            print(f"   测试集准确率: {best_result['test_accuracy']:.3f}")

            # 分类报告
            print(f"\n3. 分类报告:")
            print(classification_report(y_test, y_pred, target_names=class_names))

            # 混淆矩阵
            print(f"\n4. 混淆矩阵:")
            self.plot_confusion_matrix(y_test, y_pred, class_names)
        else:
            print(f"\n错误: 找不到最佳模型 '{best_model_name}' 的结果")

    def run_evaluation(self, results, best_model, best_model_name, X_test, y_test, y_pred,
                       feature_importance_df, label_encoder):
        """运行完整的模型评估"""
        class_names = label_encoder.classes_

        # 生成报告
        self.generate_report(results, best_model_name, y_test, y_pred, class_names)

        # 模型比较图
        self.plot_model_comparison(results)

        # 特征重要性图
        self.plot_feature_importance(feature_importance_df)

        # ROC曲线（如果模型支持概率预测）
        if best_model is not None:
            self.plot_roc_curve(best_model, X_test, y_test, class_names)