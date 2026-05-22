#!/usr/bin/env python3
"""
紧急清理所有索引文件，释放磁盘空间

用途：当磁盘配额超限时，删除所有已生成的索引文件
进度不会丢失（保存在progress.json中），可以安全重新运行
"""

import os
import glob
from pathlib import Path

# 数据集列表
DATASETS = ['yfcc', 'LAION1M', 'tripclick', 'ytb_audio', 'ytb_video', 'arxiv']

def get_dir_size(path):
    """获取目录大小（MB）"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(entry.path)
    except PermissionError:
        pass
    return total / (1024 * 1024)

def cleanup_dataset_indices(dataset):
    """清理指定数据集的所有索引文件"""
    base_path = f"/home/remote/u7905817/benchmarks/discrete/ACORN/data"

    # 查找所有可能的索引目录
    search_patterns = [
        f"{base_path}/param_search_{dataset}/indices/{dataset}/*.json",
        f"{base_path}/quick_test_{dataset}/indices/{dataset}/*.json",
    ]

    total_deleted = 0
    total_size = 0

    for pattern in search_patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                size = os.path.getsize(file) / (1024 * 1024)
                os.remove(file)
                print(f"✅ 删除: {os.path.basename(file)} ({size:.1f}MB)")
                total_deleted += 1
                total_size += size
            except Exception as e:
                print(f"❌ 删除失败 {file}: {e}")

    return total_deleted, total_size

def main():
    print("=" * 70)
    print("开始清理所有索引文件...")
    print("=" * 70)
    print()

    grand_total_files = 0
    grand_total_size = 0

    for dataset in DATASETS:
        print(f"\n📁 处理数据集: {dataset}")
        print("-" * 70)

        deleted, size = cleanup_dataset_indices(dataset)

        if deleted > 0:
            print(f"✅ {dataset}: 删除 {deleted} 个文件，释放 {size:.1f}MB")
        else:
            print(f"⏭️  {dataset}: 无索引文件")

        grand_total_files += deleted
        grand_total_size += size

    print()
    print("=" * 70)
    print(f"清理完成！")
    print(f"总共删除: {grand_total_files} 个文件")
    print(f"释放空间: {grand_total_size:.1f}MB ({grand_total_size/1024:.2f}GB)")
    print("=" * 70)
    print()
    print("注意：")
    print("- progress.json 中的进度记录未被删除")
    print("- summary.csv 中的结果未被删除")
    print("- 可以安全重新运行参数搜索脚本，将从进度记录继续")

if __name__ == '__main__':
    main()
