# 自动化索引测试工作流

## 📋 概述

这个自动化工作流能够：
1. **自动测试多种索引配置**（IVF+PQ 等）
2. **自动调参**（autotune）找到最优搜索参数
3. **详细性能测试**（R@1, R@10, R@100, 查询时间）
4. **生成对比报告**（TXT, JSON, CSV 三种格式）
5. **支持三种查询类型**（EQUAL, OR, AND）一次性测试

## 🚀 快速开始

### 基础用法

```bash
cd /home/remote/u7905817/benchmarks/discrete/ACORN/benchs

# 使用默认配置运行
python auto_workflow_arxiv.py

# 或者指定数据集
python auto_workflow_arxiv.py --dataset arxiv_all
```

### 高级用法

```bash
# 使用配置文件
python auto_workflow_arxiv.py --config index_configs.json

# 指定输出目录
python auto_workflow_arxiv.py --output my_results

# 组合使用
python auto_workflow_arxiv.py --dataset arxiv_all --config index_configs.json --output results/test1
```

## 📁 配置文件

可以通过 JSON 配置文件自定义测试的索引配置。参考 `index_configs.json`：

```json
{
  "dataset": "arxiv_all",
  "output_dir": "results/auto_workflow",
  
  "index_configs": [
    {
      "name": "IVF4096_PQ64",
      "key": "IVF4096,PQ64",
      "description": "4096个聚类中心，64字节量化"
    },
    {
      "name": "IVF8192_PQ64",
      "key": "IVF8192,PQ64",
      "description": "8192个聚类中心，64字节量化"
    }
  ]
}
```

### 索引配置说明

- **IVF**: Inverted File（倒排文件），数字表示聚类中心数量
  - 更多聚类 = 更精确但训练更慢
  - 常用：1024, 2048, 4096, 8192

- **PQ**: Product Quantization（乘积量化），数字表示量化字节数
  - 更多字节 = 更精确但占用更多内存
  - 常用：32, 64

- **Flat**: 精确搜索（无压缩），作为基准对比

## 📊 输出文件

工作流会在输出目录生成以下文件：

```
results/auto_workflow/
├── workflow_20251130_230518.log          # 详细运行日志
├── summary_20251130_230518.txt           # 汇总报告（人类可读）
├── results_20251130_230518.json          # 完整结果（JSON格式）
└── results_20251130_230518.csv           # 结果表格（Excel友好）
```

### 文件说明

1. **日志文件 (.log)**
   - 记录完整的运行过程
   - 包含时间戳和错误信息
   - 用于调试和追溯

2. **汇总报告 (.txt)**
   - 按查询类型分类的性能对比表
   - 最佳配置推荐
   - 可直接查看

3. **JSON 结果 (.json)**
   - 完整的结构化数据
   - 包含所有参数和性能指标
   - 便于程序化处理

4. **CSV 表格 (.csv)**
   - 可在 Excel/Numbers 中打开
   - 便于绘图和进一步分析
   - 格式：Index, Query_Type, Best_Params, R@1, R@10, R@100, Time_ms

## 🔄 工作流程

```
开始
  ↓
对每个索引配置：
  ├─ 1. 训练索引（如果不存在）
  ├─ 2. 填充向量（如果不存在）
  ├─ 3. 运行 autotune 自动调参
  │    ├─ EQUAL 查询类型
  │    ├─ OR 查询类型
  │    └─ AND 查询类型
  ├─ 4. 使用最优参数进行详细测试
  └─ 5. 记录所有结果
  ↓
生成汇总报告
  ├─ 性能对比表
  ├─ 最佳配置推荐
  └─ 导出多种格式
  ↓
完成
```

## 📈 结果解读

### 性能指标

- **R@1, R@10, R@100**: 召回率（Recall）
  - R@1: 在前 1 个结果中找到正确答案的比例
  - R@10: 在前 10 个结果中找到正确答案的比例
  - R@100: 在前 100 个结果中找到正确答案的比例
  - 值越高越好（最高 1.0 = 100%）

- **Time (ms)**: 每个查询的平均时间（毫秒）
  - 越小越好
  - 通常在 0.01ms - 10ms 之间

- **Best Params**: 最优搜索参数
  - `nprobe=N`: 搜索 N 个聚类中心
  - `ht=N`: 汉明距离阈值
  - 参数越大通常越慢但更准确

### 汇总报告示例

```
====================================================================================
Query Type: EQUAL
====================================================================================

Index                Best Params               R@1      R@10     R@100    Time(ms)
------------------------------------------------------------------------------------
IVF4096_PQ64         nprobe=1,ht=2             1.0000   1.0000   1.0000   0.054
IVF8192_PQ64         nprobe=2,ht=2             1.0000   1.0000   1.0000   0.089
Flat                                           1.0000   1.0000   1.0000   1.234

====================================================================================
Recommendations:
====================================================================================

EQUAL:
  Best configuration: IVF4096_PQ64
  Optimal parameters: nprobe=1,ht=2
  R@1: 1.0000, Time: 0.054ms
```

## 🎯 使用场景

### 场景 1: 快速测试单个配置
```bash
# 修改 index_configs.json，只保留一个配置
python auto_workflow_arxiv.py --config index_configs.json
```

### 场景 2: 对比多个配置找最优
```bash
# 使用完整配置列表
python auto_workflow_arxiv.py --config index_configs.json --output results/comparison
```

### 场景 3: 测试新数据集
```bash
# 1. 修改 bench_arxiv.py 添加新数据集支持
# 2. 运行工作流
python auto_workflow_arxiv.py --dataset my_new_dataset
```

## 🔧 自定义和扩展

### 添加新的索引类型

在配置文件中添加：
```json
{
  "name": "HNSW32",
  "key": "HNSW32",
  "description": "HNSW图索引，32个邻居"
}
```

### 修改测试参数

编辑 `bench_arxiv.py` 中的参数范围或添加新的测试模式。

### 自定义评分函数

在 `auto_workflow_arxiv.py` 的 `generate_summary()` 方法中修改评分逻辑：
```python
# 当前：召回率 - 时间/1000
score = best['recall'] - (best['time'] / 1000)

# 可以改为：召回率优先，时间次之
score = best['recall'] * 100 + (1.0 / (best['time'] + 0.001))
```

## ⚠️ 注意事项

1. **存储空间**: 每个索引配置会生成训练和填充后的索引文件（可能几百MB到几GB）
   - 索引文件位置：`/tmp/bench_polysemous/`
   - 可以手动删除以节省空间

2. **运行时间**: 完整工作流可能需要较长时间
   - 首次训练索引：几分钟到几十分钟
   - 后续测试（索引已存在）：几秒到几分钟
   - 建议先用少量配置测试

3. **内存使用**: 大型数据集和索引会占用大量内存
   - 使用 memory-mapped 文件减少内存占用
   - 必要时可以减少训练向量数量

## 📚 相关文件

- `bench_arxiv.py`: 核心基准测试脚本
- `auto_workflow_arxiv.py`: 自动化工作流脚本
- `index_configs.json`: 索引配置文件
- `datasets/`: 数据集目录

## 🆘 故障排查

### 问题：索引训练失败
```
解决：检查数据集路径是否正确，确保有足够的训练数据
```

### 问题：内存不足
```
解决：减少训练向量数量（修改 bench_arxiv.py 中的 xt = xb[:100000]）
```

### 问题：Autotune 没有找到参数
```
解决：检查索引类型是否支持参数搜索（Flat 索引无参数可调）
```

### 问题：结果文件不完整
```
解决：查看日志文件找到具体错误信息
```

## 💡 最佳实践

1. **先小后大**: 先测试少量配置，确认工作正常后再扩大范围
2. **保存配置**: 为不同的实验创建不同的配置文件
3. **记录结果**: 用有意义的输出目录名，如 `results/exp_20251130_test1`
4. **对比分析**: 使用 CSV 文件在 Excel 中绘制性能对比图表
5. **定期清理**: 删除不需要的索引文件释放空间

## 📖 进一步阅读

- Faiss 文档: https://github.com/facebookresearch/faiss/wiki
- IVF 索引原理: https://github.com/facebookresearch/faiss/wiki/Faster-search
- PQ 量化方法: https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint

