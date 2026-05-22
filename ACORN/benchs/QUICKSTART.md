# 快速开始指南

## 🚀 三步开始使用

### 1️⃣ 快速测试（推荐首次使用）

```bash
cd /home/remote/u7905817/benchmarks/discrete/ACORN/benchs
./run_workflow.sh quick
```

这会测试单个索引配置，大约需要 **5-10 分钟**。

### 2️⃣ 查看结果

测试完成后，查看结果：

```bash
# 查看汇总报告
cat results/quick_test/summary_*.txt

# 或在 Excel 中打开 CSV 文件
open results/quick_test/results_*.csv
```

### 3️⃣ 运行完整测试

确认一切正常后，运行完整测试：

```bash
./run_workflow.sh full
```

这会测试所有配置，可能需要 **30-60 分钟**。

---

## 📊 结果文件位置

```
results/
├── quick_test/              # 快速测试结果
│   ├── summary_*.txt        # ← 看这个！汇总报告
│   ├── results_*.csv        # ← Excel 表格
│   ├── results_*.json       # JSON 数据
│   └── workflow_*.log       # 详细日志
│
└── auto_workflow/           # 完整测试结果
    └── ...
```

---

## 🎯 核心概念

### 工作流做什么？

1. **建立索引** → 训练和填充向量数据库
2. **自动调参** → 找到最优搜索参数（autotune）
3. **性能测试** → 测试召回率和查询速度
4. **生成报告** → 对比所有配置，推荐最佳选择

### 关键参数说明

- **索引类型**: 如 `IVF4096,PQ64`
  - `IVF4096`: 4096 个聚类中心
  - `PQ64`: 64 字节乘积量化
  
- **搜索参数**: 如 `nprobe=1,ht=2`
  - `nprobe`: 搜索几个聚类（越大越慢越准）
  - `ht`: 汉明阈值（PQ 索引专用）

- **查询类型**: EQUAL, OR, AND
  - 不同的标签查询模式
  - 工作流会自动测试所有三种

---

## 💡 常见用法

### 只测试特定索引

编辑 `quick_test_config.json`：

```json
{
  "index_configs": [
    {"name": "IVF4096_PQ64", "key": "IVF4096,PQ64"},
    {"name": "IVF8192_PQ64", "key": "IVF8192,PQ64"}
  ]
}
```

然后运行：
```bash
./run_workflow.sh quick
```

### 使用自定义配置

```bash
./run_workflow.sh custom my_config.json
```

### 手动运行（不用脚本）

```bash
python auto_workflow_arxiv.py --config quick_test_config.json --output my_results
```

---

## 📈 如何看懂结果

### 汇总报告示例

```
Query Type: EQUAL
================================================================================

Index                Best Params               R@1      R@10     R@100    Time(ms)
--------------------------------------------------------------------------------
IVF4096_PQ64         nprobe=1,ht=2             1.0000   1.0000   1.0000   0.054
```

**解读**：
- ✅ **R@1 = 1.0000**: 100% 召回率，非常好！
- ⚡ **Time = 0.054ms**: 查询很快
- 🎯 **nprobe=1**: 只需搜索 1 个聚类就能达到完美召回

### 推荐配置

报告末尾会显示：

```
Recommendations:
================================================================================

EQUAL:
  Best configuration: IVF4096_PQ64
  Optimal parameters: nprobe=1,ht=2
  R@1: 1.0000, Time: 0.054ms
```

**这就是你应该使用的配置！**

---

## 🔧 下一步

### 在你的代码中使用最优配置

```python
import faiss

# 加载索引
index = faiss.read_index('/tmp/bench_polysemous/arxiv_all_IVF4096,PQ64_populated.index')

# 设置最优参数（从报告中获得）
index.nprobe = 1
ivfpq_index = faiss.downcast_index(index.index)
ivfpq_index.polysemous_ht = 2

# 搜索
D, I = index.search(query_vectors, k=100)
```

### 测试新数据集

1. 在 `bench_arxiv.py` 中添加新数据集
2. 创建配置文件指定数据集名称
3. 运行工作流

### 优化特定查询类型

查看报告中不同查询类型的结果，针对性优化。

---

## ⚠️ 注意事项

- **首次运行慢**: 需要训练索引，后续会快很多
- **磁盘空间**: 索引文件在 `/tmp/bench_polysemous/`
- **内存使用**: 大数据集可能需要较多内存

---

## 🆘 遇到问题？

### 问题：命令找不到
```bash
# 确保在正确目录
cd /home/remote/u7905817/benchmarks/discrete/ACORN/benchs

# 确保脚本有执行权限
chmod +x run_workflow.sh
```

### 问题：Python 模块缺失
```bash
pip install faiss-cpu numpy
```

### 问题：数据集路径错误
检查 `bench_arxiv.py` 中的数据集路径是否正确。

---

## 📚 更多信息

- 详细文档：`README_WORKFLOW.md`
- 配置示例：`index_configs.json`
- 核心脚本：`bench_arxiv.py`

