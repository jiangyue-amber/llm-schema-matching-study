import pandas as pd
import os
from pathlib import Path

# 获取results文件夹中的所有CSV文件
results_dir = Path("results1") #replace
csv_files = sorted([f for f in results_dir.glob("*_results*.csv")])

# 存储结果
results_data = []

for csv_file in csv_files:
    print(f"Processing {csv_file.name}...")
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 获取decision和benchmark列
    decisions = df['decision'].str.lower().values
    # 处理benchmark列中可能的NaN值
    benchmarks = df['benchmark'].fillna('').astype(str).values
    
    # 计算unknown的百分比
    unknown_count = sum(1 for d in decisions if d == 'unknown')
    total_count = len(decisions)
    unknown_percentage = (unknown_count / total_count * 100) if total_count > 0 else 0
    
    # 将unknown转换为no
    decisions_converted = ['no' if d == 'unknown' else d for d in decisions]
    
    # 转换benchmark为小写以便比较
    benchmarks_lower = [b.lower() for b in benchmarks]
    
    # 计算TP, FP, TN, FN
    # benchmark: True -> positive, False -> negative
    # decision: yes -> positive, no -> negative
    tp = sum(1 for d, b in zip(decisions_converted, benchmarks_lower) if d == 'yes' and b == 'true')
    fp = sum(1 for d, b in zip(decisions_converted, benchmarks_lower) if d == 'yes' and b == 'false')
    fn = sum(1 for d, b in zip(decisions_converted, benchmarks_lower) if d == 'no' and b == 'true')
    tn = sum(1 for d, b in zip(decisions_converted, benchmarks_lower) if d == 'no' and b == 'false')
    
    # 计算Precision, Recall, F1
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    results_data.append({
        'File': csv_file.name,
        'Unknown %': round(unknown_percentage, 2),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4),
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn
    })

# 创建结果数据框
results_df = pd.DataFrame(results_data)

# 打印简洁的结果表格
print("\n" + "="*80)
print("EVALUATION METRICS SUMMARY")
print("="*80)
summary_df = results_df[['File', 'Unknown %', 'Precision', 'Recall', 'F1']]
print(summary_df.to_string(index=False))

# 计算每个模型的平均值
print("\n" + "="*80)
print("AVERAGE METRICS BY MODEL")
print("="*80)

gemini_df = results_df[results_df['File'].str.contains('gemini')]
qwen_df = results_df[results_df['File'].str.contains('huggingface')]

print(f"\nGemini Model (7 datasets):")
print(f"  Unknown %: {gemini_df['Unknown %'].mean():.2f}%")
print(f"  Precision: {gemini_df['Precision'].mean():.4f}")
print(f"  Recall: {gemini_df['Recall'].mean():.4f}")
print(f"  F1: {gemini_df['F1'].mean():.4f}")

print(f"\nQwen/HuggingFace Model (7 datasets):")
print(f"  Unknown %: {qwen_df['Unknown %'].mean():.2f}%")
print(f"  Precision: {qwen_df['Precision'].mean():.4f}")
print(f"  Recall: {qwen_df['Recall'].mean():.4f}")
print(f"  F1: {qwen_df['F1'].mean():.4f}")

# 保存详细结果到CSV
output_file = "evaluation_metrics_results1.csv"
results_df.to_csv(output_file, index=False)
print(f"\n✓ 详细结果已保存到: {output_file}")
