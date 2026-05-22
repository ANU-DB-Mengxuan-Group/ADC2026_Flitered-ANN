#!/bin/bash
# 自动化索引测试工作流启动脚本

echo "=========================================="
echo "自动化索引测试工作流"
echo "=========================================="
echo ""

# 检查参数
if [ "$1" == "quick" ]; then
    echo "运行模式: 快速测试（单个索引配置）"
    python auto_workflow_arxiv.py --config quick_test_config.json
elif [ "$1" == "full" ]; then
    echo "运行模式: 完整测试（所有索引配置）"
    python auto_workflow_arxiv.py --config index_configs.json
elif [ "$1" == "custom" ]; then
    if [ -z "$2" ]; then
        echo "错误: 请指定配置文件"
        echo "用法: ./run_workflow.sh custom <config_file>"
        exit 1
    fi
    echo "运行模式: 自定义配置 ($2)"
    python auto_workflow_arxiv.py --config "$2"
else
    echo "用法:"
    echo "  ./run_workflow.sh quick     # 快速测试（推荐首次使用）"
    echo "  ./run_workflow.sh full      # 完整测试（所有配置）"
    echo "  ./run_workflow.sh custom <config.json>  # 使用自定义配置"
    echo ""
    echo "示例:"
    echo "  ./run_workflow.sh quick"
    echo "  ./run_workflow.sh custom my_config.json"
    exit 1
fi

echo ""
echo "=========================================="
echo "测试完成！检查 results/ 目录查看结果"
echo "=========================================="

