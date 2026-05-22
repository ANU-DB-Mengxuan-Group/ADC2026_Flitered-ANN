#!/bin/bash
# 重新编译 ACORN demos (在修改 BITSET_MAX_SIZE 后需要运行)

set -e  # 遇到错误立即退出

echo "=========================================="
echo "重新编译 ACORN demos"
echo "=========================================="
echo ""

cd ~/benchmarks/discrete/ACORN

# 检查 build 目录是否存在
if [ ! -d "build" ]; then
    echo "❌ build 目录不存在，需要先运行完整编译:"
    echo "   cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build"
    echo "   make -C build -j faiss"
    exit 1
fi

echo "📁 进入 build 目录..."
cd build

echo ""
echo "🔨 重新编译关键可执行文件..."
echo ""

# 重新编译构建索引程序
echo "1️⃣  编译 build_acorn_index..."
make build_acorn_index -j
echo "   ✅ build_acorn_index 编译完成"
echo ""

# 重新编译搜索程序（这是最关键的，包含 BITSET_MAX_SIZE）
echo "2️⃣  编译 search_acorn_index..."
make search_acorn_index -j
echo "   ✅ search_acorn_index 编译完成"
echo ""

echo "3️⃣  编译 search_acorn_index_parse..."
make search_acorn_index_parse -j
echo "   ✅ search_acorn_index_parse 编译完成"
echo ""

echo "=========================================="
echo "✅ 所有程序编译完成！"
echo "=========================================="
echo ""
echo "可执行文件位置:"
echo "  - $(pwd)/demos/build_acorn_index"
echo "  - $(pwd)/demos/search_acorn_index"
echo "  - $(pwd)/demos/search_acorn_index_parse"
echo ""
echo "现在可以运行参数搜索脚本了！"
