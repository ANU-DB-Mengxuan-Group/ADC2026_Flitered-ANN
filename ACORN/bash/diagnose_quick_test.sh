#!/bin/bash
# 诊断quick test的问题

echo "========================================"
echo "1. 检查索引文件是否存在"
echo "========================================"

for dataset in yfcc LAION1M tripclick ytb_video; do
    echo ""
    echo "数据集: $dataset"
    base_dir="/home/remote/u7905817/benchmarks/discrete/ACORN/data/quick_test_${dataset}/indices"

    # 检查索引文件
    if [ -d "$base_dir/$dataset" ]; then
        echo "✅ 目录存在: $base_dir/$dataset"
        echo "索引文件:"
        ls -lh "$base_dir/$dataset"/*.json 2>/dev/null || echo "  ❌ 没有.json文件"
    else
        echo "❌ 目录不存在: $base_dir/$dataset"
    fi
done

echo ""
echo "========================================"
echo "2. 检查数据集文件完整性"
echo "========================================"

data_base="/home/remote/u7905817/benchmarks/datasets/discrete"

for dataset in yfcc LAION1M tripclick ytb_video; do
    echo ""
    echo "数据集: $dataset"

    base_file="$data_base/$dataset/${dataset}_base.fvecs"
    if [ -f "$base_file" ]; then
        size=$(ls -lh "$base_file" | awk '{print $5}')
        echo "  ✅ base文件: $size"
    else
        echo "  ❌ base文件不存在"
    fi

    # 检查query和gt文件（根据scenario）
    case $dataset in
        yfcc|tripclick|ytb_video)
            scenario="equal"
            ;;
        LAION1M)
            scenario="and"
            ;;
    esac

    query_file="$data_base/$dataset/${dataset}_query_${scenario}.fvecs"
    gt_file="$data_base/$dataset/${dataset}_gt_${scenario}.txt"

    if [ -f "$query_file" ]; then
        echo "  ✅ query文件存在 (${scenario})"
    else
        echo "  ❌ query文件不存在: $query_file"
    fi

    if [ -f "$gt_file" ]; then
        echo "  ✅ gt文件存在 (${scenario})"
    else
        echo "  ❌ gt文件不存在: $gt_file"
    fi
done

echo ""
echo "========================================"
echo "3. 检查YTB_VIDEO崩溃原因"
echo "========================================"

ytb_base="/home/remote/u7905817/benchmarks/datasets/discrete/ytb_video/ytb_video_base.fvecs"
if [ -f "$ytb_base" ]; then
    size=$(stat -f%z "$ytb_base" 2>/dev/null || stat -c%s "$ytb_base" 2>/dev/null)
    expected=$((5000000 * 200 * 4 + 4))  # N * dim * sizeof(float) + header

    echo "文件大小: $size 字节"
    echo "期望大小: $expected 字节 (5M向量 × 200维 × 4字节)"

    if [ $size -eq $expected ]; then
        echo "✅ 文件大小正确"
    else
        echo "⚠️  文件大小不匹配"
    fi
else
    echo "❌ YTB_VIDEO base文件不存在"
fi

echo ""
echo "========================================"
echo "4. 检查构建日志中的错误信息"
echo "========================================"

for dataset in yfcc LAION1M tripclick ytb_video; do
    log_file="/home/remote/u7905817/benchmarks/discrete/ACORN/data/quick_test_${dataset}/indices/build.log"
    if [ -f "$log_file" ]; then
        echo ""
        echo "$dataset 构建日志最后20行:"
        tail -20 "$log_file"
    fi
done
