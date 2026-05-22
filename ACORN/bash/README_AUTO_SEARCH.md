# ACORN 自动参数搜索 - Arxiv数据集

## 🚨 重要修复（2025-12-06）

**问题发现**：之前运行的程序所有Recall都显示为0.0000，这是因为：
- `search_acorn_index` 程序将结果写入CSV文件（`M={M}_M_beta={M_beta}_gamma={gamma}_result.csv`）
- 但Python脚本试图从日志文件解析Recall，导致始终读取不到数据

**已修复**：
1. 新增 `parse_search_csv()` 函数从CSV文件正确读取Recall/QPS数据
2. 修正 `search_index()` 的output_path参数传递
3. 更新summary CSV列：`search_time_ms` -> `qps`, `qps_no_filter`

## 🔄 如何重新启动搜索

### 在weirdo节点上执行：

```bash
# 1. 停止当前运行（如果还在运行）
# 按 Ctrl+C 停止前台进程

# 2. 拉取最新代码
cd /home/remote/u7905817/benchmarks/discrete/ACORN
git pull origin master

# 3. 进入bash目录
cd bash

# 4. （可选）清理之前的错误数据
rm -f ../data/param_search_arxiv/summary.csv
rm -f ../data/param_search_arxiv/progress.json

# 5. 重新启动参数搜索
python auto_param_search_arxiv.py
```

### 为什么要清理数据？

- `summary.csv`：包含错误的Recall=0数据，需要重新生成
- `progress.json`：可以保留（这样已完成的构建不会重复），但如果想完全重新开始就删除

### 如果想保留已构建的索引，只重新测试：

**推荐做法**：保留旧summary.csv，脚本会自动恢复build_time和index_size
```bash
# 不删除任何文件，直接重新运行
python auto_param_search_arxiv.py

# 脚本会：
# 1. 从旧summary.csv读取之前的构建信息（build_time, index_size）
# 2. 检测索引文件是否完整（>1KB）
# 3. 跳过已完成的构建，只重新运行搜索
# 4. 生成新的summary.csv包含完整metrics
```

### 构建中断保护（新增功能）

脚本现在会自动检测并处理构建中断：
- 索引文件不完整（<1KB）→ 自动重建
- progress.json有记录但文件损坏 → 自动重建
- 保留旧summary中的build_time和index_size → 避免数据丢失

## 📊 预期输出

修复后，你应该看到类似这样的输出：

```
[20:05:37]    ✅ 测试成功:
[20:05:37]       QPS (with filter): 1234.56
[20:05:37]       Recall@10: 0.8250
```

而不是：
```
[20:05:37]       Recall@10: 0.0000  # ❌ 错误！
```

## 🏃 后台运行（推荐）

如果想在后台运行：

```bash
# 使用screen或tmux（推荐）
screen -S acorn_search
cd /home/remote/u7905817/benchmarks/discrete/ACORN/bash
python auto_param_search_arxiv.py

# 按Ctrl+A然后D detach
# 重新连接: screen -r acorn_search
```

或使用nohup：
```bash
nohup python -u auto_param_search_arxiv.py > search.log 2>&1 &
tail -f search.log
```

## 📈 监控进度

```bash
# 查看进度
cat /home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_arxiv/progress.json

# 查看summary（搜索开始后会创建）
tail -20 /home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_arxiv/summary.csv
```

## ⏱️ 预计时间

- 总任务：132个（44个参数组合 × 3个场景）
- 每个构建：2-3分钟
- 每个搜索：<1分钟
- **总耗时：约10-14小时**

## 📝 参数配置

当前搜索的参数范围（已优化为能达到Recall>0.80的范围）：

```python
Ms = [32, 48, 64]                   # 3个值
M_betas = [48, 64, 96, 128]         # 4个值
gammas = [4, 8, 12, 24]             # 4个值
SCENARIOS = ['equal', 'or', 'and']  # 3个场景
```

约束：`M_beta >= M` 且 `M_beta <= 2*M*gamma`

有效组合：44个

## 🎯 下一步

参数搜索完成后：
1. 分析结果，汇总所有CSV数据
2. 绘制Recall-QPS曲线图
3. 选择代表性参数
4. 备份结果到论文仓库
