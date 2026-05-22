# ACORN参数搜索脚本 V2 使用指南

## 📋 改进内容

相比V1版本，V2增强了以下功能：

### 1. **完善的断点续传**
- 记录每个任务的构建/搜索状态
- 中断后重启自动跳过已完成任务
- 检测损坏的构建并自动重试

### 2. **自动空间管理**
- 搜索完成后自动删除索引文件
- 只保留CSV结果（节省80%空间）
- 防止磁盘空间不足

### 3. **所有数据集支持3场景**
- 所有数据集都测试 equal, or, and 三个场景
- 基于实际数据集文件验证

### 4. **优化的参数范围**
- 基于quick test结果调整
- 移除会失败的参数组合
- 添加参数约束检查

### 5. **更好的日志和监控**
- 实时进度显示
- 详细的错误信息
- 任务完成统计

---

## 🚀 使用方法

### 推荐：使用screen运行（防止断线）

```bash
# 1. 启动screen会话
screen -S acorn_search

# 2. 进入工作目录
cd ~/benchmarks/discrete/ACORN/bash

# 3. 运行脚本（unbuffered输出）
python -u auto_param_search_v2.py 2>&1 | tee search.log

# 4. 断开screen（不会中断程序）
# 按键: Ctrl+A, 然后按 D

# 5. 重新连接
screen -r acorn_search

# 6. 查看所有screen会话
screen -ls
```

### 运行指定数据集

```bash
# 只运行单个数据集
python -u auto_param_search_v2.py yfcc 2>&1 | tee yfcc_search.log

# 运行多个数据集
python -u auto_param_search_v2.py yfcc LAION1M 2>&1 | tee multi_search.log

# 运行所有数据集（默认）
python -u auto_param_search_v2.py 2>&1 | tee all_search.log
```

---

## 📊 数据集配置

| 数据集 | 向量数 | 维度 | 场景 | 预计时间 |
|--------|--------|------|------|----------|
| arxiv | 132K | 200 | equal, or, and | ~4小时 |
| yfcc | 1M | 192 | equal, or, and | ~8小时 |
| LAION1M | 1M | 200 | equal, or, and | ~8小时 |
| tripclick | 1M | 768 | equal, or, and | ~10小时 |
| ytb_video | 1M | 1024 | equal, or, and | ~10小时 |

**预计总时间：** 约40小时（串行），10小时（并行4数据集）

---

## 📁 输出文件结构

```
data/
├── param_search_yfcc/
│   ├── progress.json                    # 进度追踪
│   ├── results/
│   │   └── yfcc/
│   │       ├── summary.csv              # 汇总结果（重要！）
│   │       ├── equal/
│   │       │   ├── M=32_M_beta=48_gamma=4_result.csv
│   │       │   └── M=32_M_beta=48_gamma=4_search.log
│   │       ├── or/
│   │       └── and/
│   └── indices/                         # 索引文件（搜索后自动删除）
│       └── yfcc/
├── param_search_LAION1M/
├── param_search_tripclick/
└── param_search_ytb_video/
```

**重要文件：**
- `summary.csv`: 所有参数组合的性能汇总（用于绘图）
- `progress.json`: 断点续传进度
- `*_result.csv`: 每个参数组合的详细Recall-QPS曲线

---

## 🔍 监控进度

### 查看日志
```bash
# 实时跟踪
tail -f search.log

# 查看最近的输出
tail -100 search.log

# 搜索特定信息
grep "✅ 搜索成功" search.log
grep "❌" search.log  # 查看错误
```

### 查看进度文件
```bash
# 查看当前进度
cat data/param_search_yfcc/progress.json | python -m json.tool | head -50

# 统计完成情况
cat data/param_search_yfcc/progress.json | grep -o '"build_done": true' | wc -l
cat data/param_search_yfcc/progress.json | grep -o '"search_done": true' | wc -l
```

### 查看结果
```bash
# 查看summary
head -20 data/param_search_yfcc/results/yfcc/summary.csv

# 统计Recall分布
tail -n +2 data/param_search_yfcc/results/yfcc/summary.csv | cut -d',' -f7 | sort -n | tail -10
```

---

## ⚙️ 配置选项

在脚本顶部可以修改：

```python
# 自动清理索引文件（推荐开启）
AUTO_CLEANUP = True

# 最小合法索引大小（用于检测损坏）
MIN_INDEX_SIZE_MB = 50

# 参数范围
DEFAULT_PARAMS = {
    "Ms": [32, 48, 64],
    "M_betas": [48, 64, 96],
    "gammas": [4, 8, 12, 24],
}
```

---

## 🔧 故障排查

### 问题1：进程被杀掉
**原因：** 内存不足或节点资源被抢占

**解决：**
```bash
# 申请更多资源
salloc -p db4ai --qos=db4ai -N 1 -n 1 --cpus-per-task=1 --mem=64G --time=48:00:00
```

### 问题2：某个参数组合一直失败
**原因：** 参数不合法或内存不足

**解决：** 在脚本中排除该参数组合，或增加内存

### 问题3：CSV文件解析失败
**原因：** C++程序崩溃，CSV未生成

**解决：** 检查 `*_search.log` 查看错误，删除对应进度重试

### 问题4：磁盘空间不足
**原因：** 索引文件占用太大

**解决：**
```bash
# 手动删除gamma=1索引（C++自动生成的）
find data -name "*gamma=1.json" -delete

# 清理已完成数据集的索引
rm -rf data/param_search_yfcc/indices/
```

---

## 📈 下一步

完成参数搜索后：

1. **汇总所有summary.csv**
2. **绘制Recall-QPS曲线**
3. **选择代表性参数组合**
4. **备份结果到论文仓库**

详见主README。

---

## 💡 性能建议

### 串行运行（简单但慢）
```bash
python -u auto_param_search_v2.py 2>&1 | tee search.log
```
- 优点：简单，不会资源冲突
- 缺点：需要40小时

### 并行运行（快但复杂）
在不同的screen会话中运行不同数据集：

```bash
# Screen 1
screen -S search_yfcc
python -u auto_param_search_v2.py yfcc 2>&1 | tee yfcc.log
# Ctrl+A, D

# Screen 2
screen -S search_laion
python -u auto_param_search_v2.py LAION1M 2>&1 | tee laion.log
# Ctrl+A, D

# Screen 3
screen -S search_trip
python -u auto_param_search_v2.py tripclick 2>&1 | tee trip.log
# Ctrl+A, D

# Screen 4
screen -S search_ytb
python -u auto_param_search_v2.py ytb_video 2>&1 | tee ytb.log
# Ctrl+A, D
```

- 优点：4个数据集并行，~10小时完成
- 缺点：需要管理多个screen会话

---

## ⚠️ 注意事项

1. **确保使用screen或nohup** - 避免SSH断连中断任务
2. **定期检查磁盘空间** - `df -h ~`
3. **备份progress.json** - 防止意外丢失进度
4. **Arxiv已完成** - 不需要重复运行
5. **测试YTB_VIDEO** - 确认修复后的N=1M配置正常工作

---

## 📞 帮助

遇到问题：
1. 查看日志文件 `*_search.log` 和 `*_build.log`
2. 检查progress.json的状态
3. 确认数据集文件完整性
4. 检查内存和磁盘空间
