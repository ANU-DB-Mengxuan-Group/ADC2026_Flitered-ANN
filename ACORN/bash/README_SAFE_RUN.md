# ACORN参数搜索 - 安全运行指南

## 🎯 目标：防止运行出错需要全部重来

本指南提供**分阶段测试**的方案，确保每一步都验证通过再进行下一步。

---

## 📋 三阶段测试流程

```
阶段1: 超快速测试 (5分钟)  ← 先做这个！
  ├─ 测试1个参数组合
  ├─ 验证数据路径正确
  ├─ 验证构建和搜索都能跑
  └─ ✅ 通过后进入阶段2

阶段2: 小规模测试 (30-60分钟)
  ├─ 测试5-10个参数组合
  ├─ 验证断点续传功能
  ├─ 验证结果记录正确
  └─ ✅ 通过后进入阶段3

阶段3: 完整运行 (6-8小时)
  └─ 运行全部参数组合
```

---

## 🚀 阶段1：超快速测试（必做！）

### 步骤1：上传脚本到云端

```bash
# 在本地
cd ACORN/bash
scp test_auto_search.py your_server:/path/to/ACORN/bash/
scp auto_param_search_arxiv.py your_server:/path/to/ACORN/bash/
```

### 步骤2：修改数据路径

```bash
# 在云端
cd ACORN/bash
vim test_auto_search.py

# 找到这一行，改成实际路径：
DATA_DIR = "../data/arxiv"  # 改成你的实际路径！
```

### 步骤3：运行超快速测试

```bash
cd ACORN/bash
python test_auto_search.py
```

### ✅ 期望输出：

```
[10:30:15] ====================================================================
[10:30:15] ACORN参数搜索 - 超快速测试
[10:30:15] ====================================================================
[10:30:15] 数据集: arxiv
[10:30:15] 测试参数: M=16, M_beta=32, gamma=1
[10:30:15] 测试场景: equal
[10:30:15] ====================================================================

[10:30:15] 【步骤0】检查数据文件...
[10:30:15] ✅ 基础向量: ../data/arxiv/arxiv_base.fvecs
[10:30:15] ✅ 基础标签: ../data/arxiv/label_base.txt
[10:30:15] ✅ equal查询向量: ../data/arxiv/arxiv_query_equal.fvecs
[10:30:15] ✅ equal查询标签: ../data/arxiv/arxiv_query_equal.txt
[10:30:15] ✅ equal_GT: ../data/arxiv/arxiv_gt_equal.txt

[10:30:15] ✅ 所有数据文件检查通过！

[10:30:15] 【步骤1】测试构建索引...
[10:30:15]    执行命令: ./build/demos/build_acorn_index 132687 1 ...
[10:32:30]    ✅ 构建成功！耗时: 135.2秒
[10:32:30]    ✅ 索引文件: .../hybrid_M=16_Mb=32_gamma=1.json
[10:32:30]    索引大小: 145.3 MB

[10:32:30] 【步骤2】测试搜索性能...
[10:32:30]    执行命令: ./build/demos/search_acorn_index 132687 1 ...
[10:33:15]    ✅ 搜索成功！耗时: 45.3秒

[10:33:15]    尝试解析性能指标...
[10:33:15]    📊 Recall@1: 0.8523
[10:33:15]    📊 Recall@10: 0.9234
[10:33:15]    📊 Average query time: 12.5 ms

[10:33:15] ====================================================================
[10:33:15] ✅ 测试完成！整个流程可以正常运行
[10:33:15] ====================================================================
```

### ❌ 如果遇到错误：

#### 错误1：数据文件不存在
```
❌ 基础向量不存在: ../data/arxiv/arxiv_base.fvecs
```
**解决**：修改 `DATA_DIR` 为正确路径

#### 错误2：找不到可执行文件
```
❌ 找不到可执行文件 ./build/demos/build_acorn_index
```
**解决**：先编译项目
```bash
cd ACORN
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
cd ../bash
```

#### 错误3：构建或搜索失败
```
❌ 构建失败！
```
**解决**：查看日志文件
```bash
cat ../data/test_search_arxiv/index/build.log
```

---

## 🔧 阶段2：小规模测试（验证逻辑）

### 修改参数范围（测试5个组合）

```bash
vim auto_param_search_arxiv.py

# 找到参数配置部分，改为：
Ms = [16, 24, 32]          # 3个M值
M_betas = [32, 48, 64]     # 3个M_beta值
gammas = [1]               # 1个gamma值

# 这样有效组合约5个，每个3个场景 = 15个任务
# 预计耗时：15 × 2分钟 = 30分钟
```

### 运行小规模测试

```bash
# 后台运行，记录日志
nohup python auto_param_search_arxiv.py > small_test.log 2>&1 &

# 查看实时日志
tail -f small_test.log

# 查看进度
cat ../data/param_search_arxiv/progress.json
```

### 测试断点续传

```bash
# 运行5分钟后，手动中断（Ctrl+C）

# 重新运行
python auto_param_search_arxiv.py

# 应该看到：
🔄 恢复进度：已完成 3 个任务
⏭️  索引已存在，跳过构建
⏭️  [1/15] 场景 EQUAL 已完成
...
```

### ✅ 通过标准：

1. 所有15个任务都能完成
2. 断点续传正常工作
3. 生成 `summary.csv` 文件
4. 结果看起来合理（Recall@10 > 0.85）

---

## 🎯 阶段3：完整运行（正式参数搜索）

### 修改为完整参数范围

```bash
vim auto_param_search_arxiv.py

# 改为完整范围：
Ms = list(range(12, 49, 4))           # [12, 16, 20, ..., 48]
M_betas = list(range(12, 65, 4))      # [12, 16, 20, ..., 64]
gammas = [1, 2, 4]                    # 3个gamma值

# 约70个组合 × 3场景 = 210个任务
# 预计耗时：6-8小时
```

### 运行完整搜索

```bash
# 后台运行
nohup python auto_param_search_arxiv.py > full_search.log 2>&1 &

# 获取进程ID
echo $! > search.pid

# 查看日志
tail -f full_search.log

# 查看进度（每隔1分钟）
watch -n 60 "cat ../data/param_search_arxiv/progress.json | grep remaining"
```

### 监控运行状态

```bash
# 查看是否还在运行
ps aux | grep auto_param_search

# 查看已完成任务数
wc -l ../data/param_search_arxiv/summary.csv

# 查看最新结果
tail -5 ../data/param_search_arxiv/summary.csv
```

### 如果中断了

```bash
# 检查进度
cat ../data/param_search_arxiv/progress.json

# 继续运行（自动跳过已完成的）
nohup python auto_param_search_arxiv.py >> full_search.log 2>&1 &
```

---

## 🛡️ 保护机制说明

### 1. 断点续传
- 每完成一个任务，立即保存进度到 `progress.json`
- 重新运行会自动跳过已完成的任务
- **即使运行到一半崩溃，也不会从头开始！**

### 2. 超时保护
- 构建索引：最多2小时
- 搜索测试：最多30分钟
- 超时会跳过该任务，继续下一个

### 3. 错误隔离
- 单个参数组合失败，不影响其他组合
- 失败的任务也会记录到 `summary.csv`（status=failed）

### 4. 结果实时保存
- 每完成一个任务，立即写入 `summary.csv`
- 不需要等全部完成才能看结果

### 5. 日志记录
- 每个参数组合都有独立的日志文件
- 方便排查具体哪个组合出错

---

## 📊 结果验证

### 完成后检查

```bash
# 查看汇总结果
head -20 ../data/param_search_arxiv/summary.csv

# 统计成功/失败数量
grep "success" ../data/param_search_arxiv/summary.csv | wc -l
grep "failed" ../data/param_search_arxiv/summary.csv | wc -l

# 查看最优参数（需要pandas）
python -c "
import pandas as pd
df = pd.read_csv('../data/param_search_arxiv/summary.csv')
df_success = df[df['status'] == 'success']
for scenario in ['equal', 'or', 'and']:
    print(f'\n{scenario.upper()}场景最优:')
    best = df_success[df_success['scenario'] == scenario].nlargest(1, 'recall@10')
    print(best[['M', 'M_beta', 'gamma', 'recall@10', 'search_time_ms']])
"
```

### 合理性检查

```bash
# Recall@10 应该在 0.85-0.98 之间
# 查询时间应该在 5-30ms 之间
# 索引大小应该在 100-300MB 之间

# 如果数值异常，检查对应的日志文件
```

---

## 💾 备份结果

```bash
# 完成后立即备份！
cd ../data/param_search_arxiv
tar -czf arxiv_search_results_$(date +%Y%m%d).tar.gz \
    summary.csv progress.json results/

# 下载到本地
scp your_server:/path/to/arxiv_search_results_*.tar.gz ./
```

---

## ❓ 常见问题

### Q1: 运行到一半服务器断网了怎么办？
A: 没关系！重新运行脚本，会自动从断点继续。

### Q2: 发现某个参数组合一直失败怎么办？
A: 检查该组合的日志文件，看具体错误。如果不影响其他组合，可以先跳过。

### Q3: 想中途修改参数范围怎么办？
A: 不建议中途修改。如果必须修改，删除 `progress.json` 重新开始。

### Q4: 如何确认结果是正确的？
A:
1. 检查Recall值是否合理（0.85-0.98）
2. 对比不同参数的趋势（M越大，Recall越高）
3. 随机抽查几个结果的日志文件

### Q5: 6小时太长了，能加速吗？
A: 可以！
- 减少gamma值：只测 gammas=[1,2]
- 减少参数密度：步长改为8而不是4
- 使用多台服务器并行（需要手动分配参数范围）

---

## 📝 检查清单

运行前：
- [ ] 阶段1测试通过
- [ ] 数据路径正确
- [ ] 已编译ACORN
- [ ] 磁盘空间充足（至少20GB）

运行中：
- [ ] 定期查看日志
- [ ] 监控进度
- [ ] 检查磁盘空间

运行后：
- [ ] 检查成功率（>90%）
- [ ] 验证结果合理性
- [ ] 立即备份结果
- [ ] 保存到论文仓库
