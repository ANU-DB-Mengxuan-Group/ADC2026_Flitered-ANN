#!/usr/bin/env python3
"""
重新计算quick test的索引大小并打印修正后的summary
"""
import os

# 测试参数
TEST_PARAMS = [
    (32, 48, 4),
    (48, 64, 8),
    (64, 96, 12),
]

DATASETS = ['yfcc', 'LAION1M', 'tripclick', 'ytb_video']

BASE_DIR = "/home/remote/u7905817/benchmarks/discrete/ACORN/data"

def get_index_size(dataset, M, M_beta, gamma):
    """获取索引文件大小（MB）"""
    index_path = f"{BASE_DIR}/quick_test_{dataset}/indices/{dataset}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"
    try:
        if os.path.exists(index_path):
            size_bytes = os.path.getsize(index_path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        else:
            return None
    except Exception as e:
        return None

def main():
    print("=" * 70)
    print("重新计算Quick Test索引大小")
    print("=" * 70)
    print()

    for dataset in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"数据集: {dataset.upper()}")
        print('=' * 70)
        print(f"{'M':>4}  {'Mb':>4}  {'γ':>3}  {'索引大小(MB)':>15}  {'状态':>10}")
        print('-' * 70)

        for M, M_beta, gamma in TEST_PARAMS:
            size = get_index_size(dataset, M, M_beta, gamma)
            if size is not None:
                status = "✅"
                print(f"{M:>4}  {M_beta:>4}  {gamma:>3}  {size:>15.2f}  {status:>10}")
            else:
                status = "❌ 不存在"
                print(f"{M:>4}  {M_beta:>4}  {gamma:>3}  {'N/A':>15}  {status:>10}")

        print()

    print("\n" + "=" * 70)
    print("索引文件路径示例:")
    print("=" * 70)
    dataset = DATASETS[0]
    M, M_beta, gamma = TEST_PARAMS[0]
    print(f"{BASE_DIR}/quick_test_{dataset}/indices/{dataset}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json")

if __name__ == "__main__":
    main()
