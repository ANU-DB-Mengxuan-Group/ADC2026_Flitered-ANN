#!/usr/bin/env python3
"""
检查所有数据集的实际向量数量
"""
import numpy as np
import os

def get_fvecs_count(filename):
    """从 fvecs 文件自动检测向量数量"""
    if not os.path.exists(filename):
        return None, None
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return 0, 0
    dim = fv.view(np.int32)[0]  # 第一个 int32 是维度
    n = fv.size // (dim + 1)    # 总元素数 / (维度+1) = 向量数
    return n, dim

# 数据集配置
datasets = {
    "yfcc": "/home/remote/u7905817/benchmarks/datasets/discrete/yfcc/yfcc_base.fvecs",
    "LAION1M": "/home/remote/u7905817/benchmarks/datasets/discrete/LAION1M/LAION1M_base.fvecs",
    "tripclick": "/home/remote/u7905817/benchmarks/datasets/discrete/tripclick/tripclick_base.fvecs",
    "ytb_audio": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_audio/ytb_audio_base.fvecs",
    "ytb_video": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_video/ytb_video_base.fvecs",
    "arxiv": "/home/remote/u7905817/benchmarks/datasets/discrete/arxiv/arxiv_base.fvecs",
}

print("=" * 80)
print("检查数据集实际大小")
print("=" * 80)
print()

for name, path in datasets.items():
    n, dim = get_fvecs_count(path)
    if n is None:
        print(f"❌ {name:15s} - 文件不存在: {path}")
    else:
        print(f"✅ {name:15s} - N={n:>10,d}  dim={dim:>5d}  ({path})")

print()
print("=" * 80)
