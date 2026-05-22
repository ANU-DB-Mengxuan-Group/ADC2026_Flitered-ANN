# ACORN参数搜索 - 剩余5个数据集

## 📋 数据集列表

基于Arxiv测试结果，为剩余5个数据集优化的参数范围：

| 数据集 | 数据规模 | 场景 | 参数范围 | 预计时间 |
|--------|---------|------|---------|---------|
| **yfcc** | 1,000,000 | equal, or, and | 标准 | ~6-8小时 |
| **LAION1M** | 1,000,448 | and | 标准 | ~2-3小时 |
| **tripclick** | 1,055,976 | equal, and | 标准 | ~4-5小时 |
| **ytb_video** | 5,000,000 | equal, or, and | 缩小 | ~8-10小时 |

**标准参数范围**（基于arxiv优化）：
- Ms = [32, 48, 64]
- M_betas = [48, 64, 96]  # 移除128避免失败
- gammas = [4, 8, 12, 24]

**缩小参数范围**（大数据集）：
- Ms = [32, 48, 64]
- M_betas = [48, 64]      # 减少M_beta
- gammas = [4, 8, 12]     # 减少gamma

---

## 🚀 使用方法

### 方式1：运行所有5个数据集（串行）

```bash
cd /home/remote/u7905817/benchmarks/discrete/ACORN/bash

# 使用screen后台运行（推荐）
screen -S acorn_remaining
python auto_param_search_remaining.py

# Detach: Ctrl+A然后D
# 重新连接: screen -r acorn_remaining
```

**预计总耗时**: 20-25小时（串行）

---

### 方式2：运行单个数据集

```bash
# 只运行yfcc
python auto_param_search_remaining.py yfcc

# 只运行LAION1M
python auto_param_search_remaining.py LAION1M

# 运行多个指定数据集
python auto_param_search_remaining.py yfcc tripclick
```

---

### 方式3：并行运行（推荐！最快）

在不同的screen会话中同时运行：

```bash
# Terminal 1
screen -S acorn_yfcc
cd /home/remote/u7905817/benchmarks/discrete/ACORN/bash
python auto_param_search_remaining.py yfcc
# Ctrl+A然后D

# Terminal 2
screen -S acorn_laion
cd /home/remote/u7905817/benchmarks/discrete/ACORN/bash
python auto_param_search_remaining.py LAION1M
# Ctrl+A然后D

# Terminal 3
screen -S acorn_tripclick
cd /home/remote/u7905817/benchmarks/discrete/ACORN/bash
python auto_param_search_remaining.py tripclick
# Ctrl+A然后D

# Terminal 4（大数据集单独跑）
screen -S acorn_ytb
cd /home/remote/u7905817/benchmarks/discrete/ACORN/bash
python auto_param_search_remaining.py ytb_video
# Ctrl+A然后D
```

**预计总耗时**: 8-10小时（并行，取决于最慢的数据集）

---

## 📊 输出结果

每个数据集都会生成独立的输出目录：

```
/home/remote/u7905817/benchmarks/discrete/ACORN/data/
├── param_search_arxiv/         # ✅ 已完成
├── param_search_yfcc/
│   ├── summary.csv             # 汇总结果
│   ├── progress.json           # 进度追踪
│   ├── indices/yfcc/           # 索引文件
│   └── results/yfcc/           # 详细CSV结果
├── param_search_LAION1M/
├── param_search_tripclick/
└── param_search_ytb_video/
```

---

## 🔍 监控进度

```bash
# 查看某个数据集的进度
cat /home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_yfcc/progress.json

# 查看summary（实时更新）
tail -f /home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_yfcc/summary.csv

# 统计成功/失败的任务
grep "success" /home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_yfcc/summary.csv | wc -l
grep "failed" /home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_yfcc/summary.csv | wc -l

# 重新连接screen查看实时输出
screen -r acorn_yfcc
```

---

## ⚠️ 注意事项

### 1. **参数范围优化**
- 基于arxiv测试结果，移除了M_beta=128（M=32时会失败）
- gamma范围保留24（某些场景可能需要高gamma）
- ytb_video数据集使用更小参数范围以减少构建时间

### 2. **构建中断保护**
脚本内置了中断保护机制：
- 检测不完整的索引文件（<1KB）并自动重建
- 从progress.json恢复进度
- 从旧summary.csv恢复build_time和index_size

### 3. **资源需求**
- CPU: 16核（OpenMP并行）
- 内存: 建议64GB（大数据集如ytb_video需要更多）
- 磁盘: 每个数据集约20-50GB（包括索引和结果）

### 4. **失败处理**
如果某个参数组合失败：
- 脚本会自动跳过并继续下一个
- 失败的组合会在summary.csv中标记为"failed"
- 可以手动检查build.log查看失败原因

---

## 📈 预期结果

基于arxiv的经验：

- **Recall@10范围**: 0.80-0.95（取决于参数）
- **QPS范围**: 500-2000（取决于数据集大小和参数）
- **索引大小**: 200MB-2GB（取决于数据集大小和参数）

---

## 🎯 下一步

所有数据集完成后：

1. **汇总分析**
   ```bash
   # 下载所有summary到本地
   scp weirdo:/home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_*/summary.csv ~/Desktop/
   ```

2. **绘制Recall-QPS曲线**
   - 为每个数据集的每个场景绘制曲线
   - 对比不同数据集的性能

3. **选择代表性参数**
   - 为论文选择最优参数组合
   - 生成对比表格

4. **备份结果**
   - 备份所有summary.csv和代表性的result CSV
   - 保存到论文仓库
