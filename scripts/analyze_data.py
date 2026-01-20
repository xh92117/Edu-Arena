#!/usr/bin/env python3
"""
Edu-Arena 数据分析脚本

分析模拟日志数据，生成统计报告
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ui.dashboard import load_data, get_simulation_stats, get_model_performance


def analyze_simulation_data():
    """分析模拟数据"""
    print("📊 Edu-Arena 数据分析报告")
    print("=" * 50)

    # 加载数据
    df = load_data()

    if df.empty:
        print("❌ 没有找到模拟数据，请先运行模拟程序")
        print("💡 运行命令: python main.py")
        return

    # 基本统计
    stats = get_simulation_stats(df)
    performance_df = get_model_performance(df)

    print("📈 基本统计")
    print("-" * 20)
    print(f"总环境数: {stats.get('running_environments', 0)}")
    print(f"总周数: {stats.get('total_weeks', 0)}")
    print(f"总记录数: {stats.get('total_records', 0):,}")
    print(".2f"    print(".2f"    print(".2f"    print(".2f"
    print("\n🏆 模型表现排名")
    print("-" * 20)

    if not performance_df.empty:
        # 按综合评分排序
        ranking = performance_df.sort_values('决策次数', ascending=False)
        for idx, row in ranking.iterrows():
            env_id = int(row['环境ID'])
            knowledge = row['知识储备']
            stress = row['压力水平']
            decisions = row['决策次数']

            print(f"环境 {env_id}: 知识{knowledge:.1f}, 压力{stress:.1f}, 决策{decisions}次")

    # 数据质量检查
    print("
🔍 数据质量检查"    print("-" * 20)

    # 检查数据完整性
    total_records = len(df)
    complete_records = len(df.dropna())
    completeness = complete_records / total_records * 100 if total_records > 0 else 0

    print(".2f"
    # 检查数值范围
    invalid_knowledge = len(df[(df['knowledge'] < 0) | (df['knowledge'] > 100)])
    invalid_stress = len(df[(df['stress'] < 0) | (df['stress'] > 100)])

    if invalid_knowledge > 0:
        print(f"⚠️  发现 {invalid_knowledge} 条知识值异常记录")

    if invalid_stress > 0:
        print(f"⚠️  发现 {invalid_stress} 条压力值异常记录")

    # 时间分布分析
    print("
📅 时间分布"    print("-" * 15)

    weekly_stats = df.groupby('week').agg({
        'knowledge': 'mean',
        'stress': 'mean',
        'env_id': 'count'
    }).round(2)

    print("周数 | 平均知识 | 平均压力 | 记录数")
    print("-" * 35)
    for week, row in weekly_stats.iterrows():
        print("3d"
    # 生成摘要报告
    generate_summary_report(df, stats, performance_df)


def generate_summary_report(df, stats, performance_df):
    """生成摘要报告"""
    print("
📄 生成摘要报告"    print("-" * 20)

    report_file = project_root / "logs" / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Edu-Arena 模拟数据分析报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("基本统计:\n")
            f.write(f"- 运行环境数: {stats.get('running_environments', 0)}\n")
            f.write(f"- 总模拟周数: {stats.get('total_weeks', 0)}\n")
            f.write(f"- 数据记录数: {stats.get('total_records', 0)}\n")
            f.write(".2f"            f.write(".2f"            f.write(".2f"            f.write(".2f"
            f.write("\n模型表现详情:\n")
            if not performance_df.empty:
                for _, row in performance_df.iterrows():
                    f.write(f"- 环境 {int(row['环境ID'])}: 知识{row['知识储备']:.1f}, 压力{row['压力水平']:.1f}, 决策{row['决策次数']}次\n")

        print(f"✅ 摘要报告已保存至: {report_file}")

    except Exception as e:
        print(f"❌ 生成报告失败: {e}")


def export_data_summary():
    """导出数据摘要"""
    print("
💾 导出数据摘要"    print("-" * 20)

    df = load_data()
    if df.empty:
        print("❌ 无数据可导出")
        return

    # 导出为CSV
    export_file = project_root / "logs" / f"data_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    try:
        # 选择关键字段导出
        summary_df = df[[
            'timestamp', 'env_id', 'week', 'knowledge', 'stress',
            'physical_health', 'total_relationship', 'savings'
        ]].copy()

        summary_df['timestamp'] = pd.to_datetime(summary_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')

        summary_df.to_csv(export_file, index=False, encoding='utf-8')
        print(f"✅ 数据摘要已导出至: {export_file}")
        print(f"📊 共导出 {len(summary_df)} 条记录")

    except Exception as e:
        print(f"❌ 导出失败: {e}")


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        export_data_summary()
    else:
        analyze_simulation_data()


if __name__ == "__main__":
    main()