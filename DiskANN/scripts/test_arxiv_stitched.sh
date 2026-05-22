#!/bin/bash
# Arxiv StitchedVamana测试脚本
# 用于验证bug和测试修复

set -e  # 遇到错误立即退出

DISKANN_DIR="/home/remote/u7905817/benchmarks/discrete/DiskANN"
DATA_DIR="/home/remote/u7905817/benchmarks/datasets/discrete"
DATASET="arxiv"

# 测试配置：使用已知失败的配置
R=32
STITCHED_R=32
L_BUILD=100
ALPHA=1.2
T=16

TEST_DIR="${DISKANN_DIR}/test_stitched_arxiv"
mkdir -p ${TEST_DIR}

echo "================================================"
echo "StitchedVamana Arxiv 测试"
echo "配置: R=${R}, stitched_R=${STITCHED_R}"
echo "================================================"

# Step 1: 构建索引
echo ""
echo "[1/3] 构建StitchedVamana索引..."
INDEX_PREFIX="${TEST_DIR}/index_R${R}_sR${STITCHED_R}"

time ${DISKANN_DIR}/build/apps/build_stitched_index \
    --data_type float \
    --data_path ${DATA_DIR}/${DATASET}/${DATASET}_base.bin \
    --index_path_prefix ${INDEX_PREFIX} \
    --label_file ${DATA_DIR}/${DATASET}/label_base.txt \
    -R ${R} \
    -L ${L_BUILD} \
    --alpha ${ALPHA} \
    --stitched_R ${STITCHED_R} \
    -T ${T} \
    2>&1 | tee ${TEST_DIR}/build.log

# Step 2: 检查生成的文件
echo ""
echo "[2/3] 检查生成的索引文件..."
ls -lh ${INDEX_PREFIX}* | head -10

# 检查medoid文件
if [ -f "${INDEX_PREFIX}_labels_to_medoids.txt" ]; then
    echo ""
    echo "Medoid文件前20行:"
    head -20 ${INDEX_PREFIX}_labels_to_medoids.txt

    echo ""
    echo "Medoid统计:"
    echo "  总标签数: $(wc -l < ${INDEX_PREFIX}_labels_to_medoids.txt)"

    # 检查是否有越界的medoid ID（>132687）
    echo "  检查越界medoid..."
    awk -F', ' '$2 > 132687 {count++; if(count<=10) print "    异常: label="$1" medoid="$2} END {if(count>0) print "  发现"count"个越界medoid!"; else print "  ✓ 所有medoid合法"}' \
        ${INDEX_PREFIX}_labels_to_medoids.txt
else
    echo "⚠️  未找到medoid文件: ${INDEX_PREFIX}_labels_to_medoids.txt"
fi

# Step 3: 准备查询标签和GT
echo ""
echo "[3/3] 测试overlap搜索（已知会崩溃的场景）..."

CONVERTED_DIR="${TEST_DIR}/converted_labels"
mkdir -p ${CONVERTED_DIR}/overlap

# 构建临时索引用于标签映射（如果不存在）
TMP_INDEX="${TEST_DIR}/tmp_for_labels"
if [ ! -f "${TMP_INDEX}_labels.txt" ]; then
    echo "  构建临时索引获取标签映射..."
    ${DISKANN_DIR}/build/apps/build_memory_index \
        --data_type float \
        --dist_fn l2 \
        --data_path ${DATA_DIR}/${DATASET}/${DATASET}_base.bin \
        --index_path_prefix ${TMP_INDEX} \
        --label_file ${DATA_DIR}/${DATASET}/label_base.txt \
        -R 32 -L 50 --alpha 1.2 -T ${T} \
        > /dev/null 2>&1

    # 清理临时索引文件，只保留_labels.txt
    find ${TEST_DIR} -name "tmp_for_labels*" ! -name "*_labels.txt" -type f -delete
fi

# 转换查询标签
echo "  转换overlap查询标签..."
python3 << 'PYTHON_SCRIPT'
import sys
from collections import defaultdict

def convert_labels(original_labels_file, index_labels_file, query_labels_file, output_file):
    # 读取原始标签
    original = []
    with open(original_labels_file) as f:
        for line in f:
            line = line.strip()
            if line:
                original.append(set(int(x) for x in line.split(',')))

    # 读取索引标签
    index = []
    with open(index_labels_file) as f:
        for line in f:
            line = line.strip()
            if line:
                index.append(set(int(x) for x in line.split(',')))

    # 建立映射
    orig_label_to_points = defaultdict(set)
    internal_label_to_points = defaultdict(set)

    for i, (orig_set, idx_set) in enumerate(zip(original, index)):
        for label in orig_set:
            orig_label_to_points[label].add(i)
        for label in idx_set:
            internal_label_to_points[label].add(i)

    orig_to_internal = {}
    for orig_label, orig_points in orig_label_to_points.items():
        for internal_label, internal_points in internal_label_to_points.items():
            if orig_points == internal_points:
                if orig_label not in orig_to_internal:
                    orig_to_internal[orig_label] = internal_label
                break

    # 转换查询标签
    converted = []
    with open(query_labels_file) as f:
        for line in f:
            line = line.strip()
            if line:
                q_labels = [int(x) for x in line.split(',')]
                new_labels = [orig_to_internal.get(l, l) for l in q_labels]
                converted.append(','.join(map(str, new_labels)))

    with open(output_file, 'w') as f:
        for line in converted:
            f.write(line + '\n')

    print(f"    转换完成: 映射了{len(orig_to_internal)}个标签")

convert_labels(
    "${DATA_DIR}/${DATASET}/label_base.txt",
    "${TMP_INDEX}_labels.txt",
    "${DATA_DIR}/${DATASET}/${DATASET}_query_or.txt",
    "${CONVERTED_DIR}/overlap/query_labels.txt"
)
PYTHON_SCRIPT

# 转换GT格式
echo "  转换overlap GT格式..."
python3 << 'PYTHON_SCRIPT'
import struct
import numpy as np
import os

def convert_gt(ung_gt_file, output_file, K=10):
    gt_size = os.path.getsize(ung_gt_file)
    num_queries = gt_size // (K * 8)

    with open(ung_gt_file, 'rb') as f:
        gt_data = np.frombuffer(f.read(num_queries * K * 8),
                                dtype=np.dtype([('idx', np.uint32), ('dist', np.float32)]))
    gt_data = gt_data.reshape(num_queries, K)

    ids = np.ascontiguousarray(gt_data['idx']).astype(np.uint32)
    dists = np.ascontiguousarray(gt_data['dist']).astype(np.float32)

    with open(output_file, 'wb') as f:
        f.write(struct.pack('<ii', num_queries, K))
        ids.tofile(f)
        dists.tofile(f)

    print(f"    GT转换完成: {num_queries}个查询")

convert_gt(
    "${DATA_DIR}/${DATASET}/${DATASET}_gt_or.bin",
    "${CONVERTED_DIR}/overlap/gt_diskann.bin"
)
PYTHON_SCRIPT

# 执行overlap搜索
echo ""
echo "  执行overlap搜索 (Ls=100)..."
echo "  如果崩溃，将看到 'Out of range loc' 或 'Wrong loc' 错误"
echo ""

${DISKANN_DIR}/build/apps/search_memory_index \
    --data_type float \
    --dist_fn l2 \
    --index_path_prefix ${INDEX_PREFIX} \
    --query_file ${DATA_DIR}/${DATASET}/${DATASET}_query_or.bin \
    --query_filters_file ${CONVERTED_DIR}/overlap/query_labels.txt \
    --gt_file ${CONVERTED_DIR}/overlap/gt_diskann.bin \
    --label_type uint \
    --recall_at 10 \
    --result_path ${TEST_DIR}/search_result \
    --num_threads ${T} \
    --filter_scenario overlap \
    -L 100 \
    2>&1 | tee ${TEST_DIR}/search_overlap.log

echo ""
echo "================================================"
echo "测试完成！"
echo ""
echo "关键文件:"
echo "  构建日志: ${TEST_DIR}/build.log"
echo "  Medoid文件: ${INDEX_PREFIX}_labels_to_medoids.txt"
echo "  搜索日志: ${TEST_DIR}/search_overlap.log"
echo "================================================"
