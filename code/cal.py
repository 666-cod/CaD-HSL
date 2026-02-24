import pandas as pd
import numpy as np


def main():
    # 1. 读取 step4_full.py 生成的结果文件
    try:
        df = pd.read_csv("all_tech_metrics.csv")
    except FileNotFoundError:
        print("❌ 错误：找不到 all_tech_metrics.csv 文件。请先运行 step4_full.py。")
        return

    print(f"成功加载 {len(df)} 个技术的测试结果。\n")

    # 2. 计算平均指标
    # 注意：MAPE 是百分比，MAE/MSE 是绝对数值
    avg_metrics = {
        'Base_MAPE': df['Base_MAPE'].mean(),
        'Causal_MAPE': df['Causal_MAPE'].mean(),
        'Base_MAE': df['Base_MAE'].mean(),
        'Causal_MAE': df['Causal_MAE'].mean(),
        'Base_MSE': df['Base_MSE'].mean(),
        'Causal_MSE': df['Causal_MSE'].mean()
    }

    # 3. 计算提升幅度 (Improvement)
    # 提升 = (Base - Causal)
    # 提升比例 = (Base - Causal) / Base
    imp_mape = avg_metrics['Base_MAPE'] - avg_metrics['Causal_MAPE']
    imp_mae = avg_metrics['Base_MAE'] - avg_metrics['Causal_MAE']
    imp_mse = avg_metrics['Base_MSE'] - avg_metrics['Causal_MSE']

    imp_pct_mape = imp_mape / avg_metrics['Base_MAPE'] * 100
    imp_pct_mae = imp_mae / avg_metrics['Base_MAE'] * 100
    imp_pct_mse = imp_mse / avg_metrics['Base_MSE'] * 100

    # 4. 打印报表
    print("=" * 60)
    print("           全量技术回测平均指标 (Average Metrics)")
    print("=" * 60)
    print(f"{'指标 (Metric)':<15} | {'Base (XGB)':<12} | {'Causal (Ours)':<12} | {'提升 (Imp)':<15}")
    print("-" * 60)

    print(
        f"{'MAPE':<15} | {avg_metrics['Base_MAPE']:<12.2f}% | {avg_metrics['Causal_MAPE']:<12.2f}% | +{imp_mape:.2f}% (Opt: {imp_pct_mape:.1f}%)")
    print(
        f"{'MAE':<15} | {avg_metrics['Base_MAE']:<12.0f} | {avg_metrics['Causal_MAE']:<12.0f} | +{imp_mae:.0f}   (Opt: {imp_pct_mae:.1f}%)")
    print(
        f"{'MSE':<15} | {avg_metrics['Base_MSE']:<12.2e} | {avg_metrics['Causal_MSE']:<12.2e} | +{imp_mse:.2e} (Opt: {imp_pct_mse:.1f}%)")

    print("=" * 60)
    print("\n结论分析：")
    if imp_mape > 0:
        print("✅ 你的模型 (CaD-HSL) 在整体上优于基准模型！")
        print(f"   - 平均预测误差 (MAPE) 降低了 {imp_mape:.2f} 个百分点。")
        print(f"   - 在资金绝对值预测上，MAE 平均减少了 {imp_mae:.0f} (Log-Amount 转换后数值)。")
        print(f"   - MSE 的显著降低 ({imp_pct_mse:.1f}%) 说明你的模型能有效避免“离谱”的极端错误预测。")
    else:
        print("⚠️ 模型整体表现未超过基准，请检查数据噪声或尝试 step4_refined.py 去除 Hub 节点。")


if __name__ == "__main__":
    main()