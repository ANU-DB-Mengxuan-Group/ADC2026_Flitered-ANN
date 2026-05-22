# StitchedVamana Bug Report: 图损坏导致Overlap场景崩溃

**发现时间**: 2026-02-03
**影响范围**: YFCC、Arxiv、YTB-Audio、YTB-Video 数据集的 StitchedVamana overlap 搜索全部崩溃
**严重程度**: Critical - 导致overlap场景完全不可用

## 问题现象

StitchedVamana在多个数据集的overlap场景下搜索时崩溃，错误信息：

```
Out of range loc found as an edge : [越界ID]
terminate called: diskann::ANNException  Wrong loc[越界ID]
```

### 崩溃案例

| 数据集 | 配置 | 越界ID示例 | 数据集大小 | ID倍数 |
|--------|------|-----------|-----------|--------|
| arxiv | R=32/sR=32 | 3,173,577,062 | 132,687 | 23,918x |
| yfcc | R=128/sR=32 | 1,190,700,160 | 1,000,000 | 1,191x |
| yfcc | R=128/sR=32 | 1,124,270,080 | 1,000,000 | 1,124x |
| yfcc | R=128/sR=32 | 1,121,845,248 | 1,000,000 | 1,122x |

### 场景差异

- **Overlap**: 几乎所有配置崩溃
- **Containment**: 部分配置成功（R=64/sR=32, R=128/sR=32等），部分崩溃（R=32/sR=32）
- **Equality**: 大部分配置成功，少数崩溃

## 根本原因分析

### 越界ID的真实身份

通过bit pattern分析，这些"越界ID"实际上是**float值被误解释为uint32**：

```python
>>> struct.unpack('f', struct.pack('I', 1121845248))[0]
111.0
>>> struct.unpack('f', struct.pack('I', 1124270080))[0]
131.0
>>> struct.unpack('f', struct.pack('I', 1124466688))[0]
134.0
>>> struct.unpack('f', struct.pack('I', 3173577062))[0]
-0.04124584
```

这些值明显是**距离值**而非节点ID。

### 问题追踪

1. **崩溃点** (`src/index.cpp:909-916`):
   ```cpp
   for (auto id : init_ids) {
       if (id >= _max_points + _num_frozen_pts) {
           diskann::cerr << "Out of range loc found as an edge : " << id << std::endl;
           throw diskann::ANNException(...);
       }
   }
   ```

2. **init_ids来源** (`src/index.cpp:2260-2265`):
   ```cpp
   // Overlap查询：为每个标签添加medoid
   for (const auto &label : sorted_filter_labels) {
       if (_label_to_start_id.find(label) != _label_to_start_id.end()) {
           init_ids.emplace_back(_label_to_start_id[label]);
       }
   }
   ```

3. **Medoid生成** (`apps/build_stitched_index.cpp:269-270`):
   ```cpp
   curr_label_entry_point = (uint32_t)random(0, curr_label_index.size());
   label_entry_points[lbl] = label_id_to_orig_id_map[lbl][curr_label_entry_point];
   ```

4. **疑似问题点**：
   - **Per-label图损坏**: `curr_label_index` 加载时可能包含损坏的邻居ID
   - **映射表越界**: `label_id_to_orig_id_map[lbl][node_neighbor]` 访问越界，返回垃圾值
   - **Stitching逻辑**：合并图时未验证邻居ID有效性

### 为什么Overlap更容易崩溃？

- **Overlap查询**传入多个标签，每个标签的medoid都加入`init_ids`
- **Containment/Equality**传入标签较少，碰到损坏medoid的概率低
- 任何一个medoid损坏，整个查询就会崩溃

## 代码问题定位

### 可能的Bug源头

**stitch_label_indices** (`apps/build_stitched_index.cpp:262-278`):

```cpp
for (uint32_t node_point = 0; node_point < curr_label_index.size(); node_point++)
{
    uint32_t original_point_id = label_id_to_orig_id_map[lbl][node_point];
    for (auto &node_neighbor : curr_label_index[node_point])
    {
        uint32_t original_neighbor_id = label_id_to_orig_id_map[lbl][node_neighbor];
        // ☝️ 如果node_neighbor越界，这里会访问无效内存

        std::vector<uint32_t> curr_point_neighbors = stitched_graph[original_point_id];
        if (std::find(...) == curr_point_neighbors.end())
        {
            stitched_graph[original_point_id].push_back(original_neighbor_id);
        }
    }
}
```

**问题**：
- 未检查 `node_neighbor < label_id_to_orig_id_map[lbl].size()`
- Per-label图本身可能包含越界邻居（构建错误或文件损坏）
- 越界访问vector返回未初始化内存，恰好是float值的bit pattern

## 影响评估

### 数据完整性

| 数据集 | Containment | Overlap | Equality | 总体状态 |
|--------|-------------|---------|----------|---------|
| arxiv | 部分可用 | ❌ 全部崩溃 | ✅ 可用 | 不完整 |
| yfcc | ✅ 可用 | ❌ 全部崩溃 | ✅ 可用 | 不完整 |
| LAION1M | ✅ 可用 | ✅ 可用 | ✅ 可用 | 完整 |
| tripclick | ✅ 可用 | ✅ 可用 | ✅ 可用 | 完整 |
| ytb_audio | ？ | ❌ R=64/sR=64崩溃 | ？ | 不完整 |
| ytb_video | ？ | ❌ R=128/sR=32崩溃 | ？ | 不完整 |

### 论文影响

**严重性**: StitchedVamana是DiskANN原生支持的filtered方法，overlap是三大场景之一。

**当前状态**：
- ✅ LAION1M和TripClick数据完整，可用于对比
- ❌ YFCC、Arxiv的overlap数据缺失，无法完整对比三场景
- ⚠️ YTB数据集部分缺失

**可行方案**：
1. **使用现有数据**: 仅报告LAION1M和TripClick的完整对比，其他数据集标注overlap不可用
2. **修复Bug重跑**: 需要修改DiskANN源码，验证修复，重新运行实验（时间成本高）
3. **混合策略**: 主体用完整数据，附录说明StitchedVamana的bug和部分数据缺失

## 临时解决方案

### 检测medoid损坏

在搜索前添加验证：

```cpp
// 验证_label_to_start_id的有效性
for (const auto &[label, medoid_id] : _label_to_start_id) {
    if (medoid_id >= _max_points) {
        diskann::cerr << "Invalid medoid for label " << label
                      << ": " << medoid_id << " >= " << _max_points << std::endl;
    }
}
```

### 修复stitching代码

添加边界检查：

```cpp
for (auto &node_neighbor : curr_label_index[node_point])
{
    if (node_neighbor >= label_id_to_orig_id_map[lbl].size()) {
        diskann::cerr << "Invalid neighbor " << node_neighbor
                      << " for label " << lbl << std::endl;
        continue; // 跳过无效邻居
    }
    uint32_t original_neighbor_id = label_id_to_orig_id_map[lbl][node_neighbor];
    // ...
}
```

## 实验验证（2026-02-03）

### 测试环境
- 数据集：Arxiv (132,687点，4,231个标签)
- 配置：R=32, stitched_R=32 (已知失败配置)
- 构建时间：~60秒（4231个per-label索引）

### 验证结果

✅ **成功重现bug**

检查生成的`index_R32_sR32_labels_to_medoids.txt`文件：

```
总标签数: 4,231
越界medoid: 45个 (1.1%)
```

**越界medoid示例：**

| Label | Medoid ID | 合法范围 | 倍数 | 解释为float |
|-------|-----------|---------|------|------------|
| 1349 | 3,174,016,050 | 0-132,686 | 23,921x | -0.0429 |
| 3844 | 3,271,900,848 | 0-132,686 | 24,659x | -133.24 |
| 2590 | 3,073,500,528 | 0-132,686 | 23,164x | -0.000011 |
| 2558 | 1,056,190,296 | 0-132,686 | 7,961x | 0.477 |
| 1442 | 1,092,054,194 | 0-132,686 | 8,231x | 9.464 |

### 结论

1. ✅ **Bug确认**：StitchedVamana构建的medoid映射中确实包含越界ID
2. ✅ **原因确认**：越界ID是float距离值的bit pattern（非节点ID）
3. ✅ **影响范围**：约1.1%的标签受影响（Arxiv数据集）
4. ⚠️ **稳定性差异**：不同数据集受影响程度不同
   - LAION1M/TripClick：构建成功，medoid可能全部合法
   - Arxiv/YFCC/YTB：部分medoid越界，导致overlap搜索崩溃

### Bug定位

问题出在 `apps/build_stitched_index.cpp:262-278` 的stitching逻辑：

```cpp
for (auto &node_neighbor : curr_label_index[node_point])
{
    uint32_t original_neighbor_id = label_id_to_orig_id_map[lbl][node_neighbor];
    // ☝️ 未检查node_neighbor是否越界
    // 如果node_neighbor >= map.size()，访问越界返回垃圾值
}
```

**根本原因**：Per-label图本身包含越界的邻居ID（可能是距离值被误写入图结构），导致映射时访问越界内存。

## 下一步行动

1. ✅ 已识别bug根源：stitching阶段越界访问
2. ✅ 验证medoid文件包含越界ID
3. ⬜ 决定修复策略：
   - **方案A**：使用现有数据（推荐）
     - LAION1M和TripClick完整可用
     - 其他数据集大部分可用
     - 论文中说明StitchedVamana的不稳定性
   - **方案B**：修复代码 + 重跑实验
     - 需要调试per-label图构建，找出距离值混入的原因
     - 预计1-2周
4. ⬜ 更新论文实验章节，标注数据可用性

## 参考文件

- 崩溃日志: `DiskANN/data_stitched_original/results/{dataset}/stitched_*.log`
- 构建代码: `DiskANN/apps/build_stitched_index.cpp`
- 索引代码: `DiskANN/src/index.cpp`
- Python脚本: `DiskANN/scripts/auto_stitched_diskann_original.py`
