#!/bin/bash
# ACORN参数搜索 - 快速启动脚本
# 使用方法：
#   ./quick_start.sh test       # 阶段1: 超快速测试（5分钟）
#   ./quick_start.sh small      # 阶段2: 小规模测试（30分钟）
#   ./quick_start.sh full       # 阶段3: 完整搜索（6-8小时）

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python
check_python() {
    if ! command -v python &> /dev/null; then
        if ! command -v python3 &> /dev/null; then
            log_error "未找到Python！请安装Python 3.6+"
            exit 1
        fi
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
    log_success "Python命令: $PYTHON_CMD"
}

# 检查编译
check_build() {
    if [ ! -f "../build/demos/build_acorn_index" ]; then
        log_error "ACORN未编译！"
        log_info "请先编译："
        log_info "  cd /home/remote/u7905817/benchmarks/discrete/ACORN"
        log_info "  mkdir -p build && cd build"
        log_info "  cmake -DCMAKE_BUILD_TYPE=Release .."
        log_info "  make -j"
        exit 1
    fi
    log_success "ACORN已编译"
}

# 检查数据文件
check_data() {
    DATA_DIR="/home/remote/u7905817/benchmarks/datasets/discrete/arxiv"

    log_info "检查数据文件..."

    missing_files=()

    files=(
        "arxiv_base.fvecs"
        "label_base.txt"
        "arxiv_query_equal.fvecs"
        "arxiv_query_equal.txt"
        "arxiv_gt_equal.txt"
        "arxiv_query_or.fvecs"
        "arxiv_query_or.txt"
        "arxiv_gt_or.txt"
        "arxiv_query_and.fvecs"
        "arxiv_query_and.txt"
        "arxiv_gt_and.txt"
    )

    for file in "${files[@]}"; do
        if [ ! -f "$DATA_DIR/$file" ]; then
            missing_files+=("$file")
        fi
    done

    if [ ${#missing_files[@]} -gt 0 ]; then
        log_error "缺少数据文件："
        for file in "${missing_files[@]}"; do
            echo "  - $DATA_DIR/$file"
        done
        exit 1
    fi

    log_success "所有数据文件就绪"
}

# 阶段1: 超快速测试
run_test() {
    log_info "================================"
    log_info "阶段1: 超快速测试（5分钟）"
    log_info "================================"

    check_python
    check_build
    check_data

    log_info "开始运行测试..."
    $PYTHON_CMD test_auto_search.py

    if [ $? -eq 0 ]; then
        log_success "================================"
        log_success "测试通过！可以进入阶段2"
        log_success "================================"
        log_info "下一步："
        log_info "  ./quick_start.sh small"
    else
        log_error "测试失败！请查看上面的错误信息"
        exit 1
    fi
}

# 阶段2: 小规模测试
run_small() {
    log_info "================================"
    log_info "阶段2: 小规模测试（30分钟）"
    log_info "================================"

    check_python
    check_build
    check_data

    # 备份原始配置
    if [ ! -f "auto_param_search_arxiv.py.backup" ]; then
        cp auto_param_search_arxiv.py auto_param_search_arxiv.py.backup
        log_info "已备份原始配置"
    fi

    # 修改参数范围为小规模
    log_info "修改参数范围为小规模测试..."
    sed -i.tmp "s/Ms = list(range(12, 49, 4))/Ms = [16, 24, 32]/" auto_param_search_arxiv.py
    sed -i.tmp "s/M_betas = list(range(12, 65, 4))/M_betas = [32, 48, 64]/" auto_param_search_arxiv.py
    sed -i.tmp "s/gammas = \[1, 2, 4\]/gammas = [1]/" auto_param_search_arxiv.py
    rm -f auto_param_search_arxiv.py.tmp

    log_info "参数配置："
    log_info "  Ms = [16, 24, 32]"
    log_info "  M_betas = [32, 48, 64]"
    log_info "  gammas = [1]"
    log_info "  预计约15个任务，30分钟完成"

    log_info "开始运行..."
    nohup $PYTHON_CMD auto_param_search_arxiv.py > small_test.log 2>&1 &
    PID=$!
    echo $PID > small_test.pid

    log_success "后台运行已启动！PID: $PID"
    log_info "查看日志: tail -f small_test.log"
    log_info "查看进度: cat /home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_arxiv/progress.json"
    log_info "停止运行: kill \$(cat small_test.pid)"
}

# 阶段3: 完整搜索
run_full() {
    log_info "================================"
    log_info "阶段3: 完整参数搜索（6-8小时）"
    log_info "================================"

    check_python
    check_build
    check_data

    # 恢复完整参数配置
    if [ -f "auto_param_search_arxiv.py.backup" ]; then
        log_warning "检测到备份配置，是否恢复完整参数范围？(y/n)"
        read -p "> " restore
        if [ "$restore" = "y" ]; then
            cp auto_param_search_arxiv.py.backup auto_param_search_arxiv.py
            log_success "已恢复完整参数配置"
        fi
    fi

    log_info "当前参数配置："
    grep "^Ms = " auto_param_search_arxiv.py
    grep "^M_betas = " auto_param_search_arxiv.py
    grep "^gammas = " auto_param_search_arxiv.py

    log_warning "预计运行6-8小时，确认开始？(y/n)"
    read -p "> " confirm

    if [ "$confirm" != "y" ]; then
        log_info "取消运行"
        exit 0
    fi

    log_info "开始完整参数搜索..."
    nohup $PYTHON_CMD auto_param_search_arxiv.py > full_search.log 2>&1 &
    PID=$!
    echo $PID > full_search.pid

    log_success "后台运行已启动！PID: $PID"
    log_info "查看日志: tail -f full_search.log"
    log_info "查看进度: watch -n 60 \"cat /home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_arxiv/progress.json\""
    log_info "停止运行: kill \$(cat full_search.pid)"
}

# 主函数
main() {
    if [ $# -eq 0 ]; then
        echo "ACORN参数搜索 - 快速启动脚本"
        echo ""
        echo "使用方法："
        echo "  ./quick_start.sh test       # 阶段1: 超快速测试（5分钟）"
        echo "  ./quick_start.sh small      # 阶段2: 小规模测试（30分钟）"
        echo "  ./quick_start.sh full       # 阶段3: 完整搜索（6-8小时）"
        echo ""
        echo "推荐流程："
        echo "  1. 先运行 test，确保环境正确"
        echo "  2. 再运行 small，验证逻辑正确"
        echo "  3. 最后运行 full，完整参数搜索"
        exit 0
    fi

    case "$1" in
        test)
            run_test
            ;;
        small)
            run_small
            ;;
        full)
            run_full
            ;;
        *)
            log_error "未知命令: $1"
            log_info "使用 ./quick_start.sh 查看帮助"
            exit 1
            ;;
    esac
}

main "$@"
