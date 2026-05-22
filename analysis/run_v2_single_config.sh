#!/bin/bash
# V2 验证实验 — 使用训练集最优固定配置（不做参数扫描）
#
# 原理：验证路由决策只需要方法间相对排名正确，不需要每个方法的绝对最优 recall。
# 使用训练集最优配置更符合 out-of-sample 精神（不在验证集上调参）。
# 全参数扫描留作后续 Pareto 分析。
#
# 固定配置（来自 6 个训练数据集上的最优参数）：
#   UNG:              max_degree=96, Lbuild=200, Lsearch=500
#   Post-filter:      M=64, efc=400
#   SIEVE:            M=32, ef_construction=40, index_budget=2.0, hist_pct=0.25, ef_search=200
#   ACORN:            M=64, M_beta=96, gamma=8
#   FilteredVamana:   R=128, Lsearch=2000
#   StitchedVamana:   R=128, stitched_R=96, Lsearch=2000
#   CAPS:             nb=1024 (equality only)
#   NHQ:              K=100,L=100,iter=12,S=10,R=300,RANGE=20,PL=350,B=0.4,M=1 (equality only)
#
# 用法:
#   conda activate benchmark
#   cd ~/benchmarks/discrete
#   bash analysis/run_v2_single_config.sh           # 跑 UNG + Post-filter + FV + ACORN
#   bash analysis/run_v2_single_config.sh ung       # 只跑 UNG
#   bash analysis/run_v2_single_config.sh postfilter
#   bash analysis/run_v2_single_config.sh sieve     # 必须在 weirdo 节点
#   bash analysis/run_v2_single_config.sh acorn
#   bash analysis/run_v2_single_config.sh fv        # FilteredVamana
#   bash analysis/run_v2_single_config.sh sv        # StitchedVamana (equality only)
#   bash analysis/run_v2_single_config.sh caps      # CAPS (equality only)
#   bash analysis/run_v2_single_config.sh nhq       # NHQ (equality only)
#
# 注意：
#   SIEVE 必须在 weirdo 节点上跑
#   ACORN 脚本需要从 ACORN/ 目录运行（使用相对路径找 build）

set -e
cd ~/benchmarks/discrete
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

V2_DATASETS="synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k"
METHOD="${1:-all}"

# ==================== UNG ====================
run_ung() {
    echo ""
    echo "============================================================"
    echo " UNG — 固定配置: max_degree=96, Lbuild=200, Lsearch=500"
    echo "============================================================"

    # 临时修改参数网格为单一配置
    UNG_SCRIPT="UNG-dev/bash/auto_ung_original.py"
    cp "$UNG_SCRIPT" "${UNG_SCRIPT}.bak"

    sed -i 's/^MAX_DEGREES = .*/MAX_DEGREES = [96]/' "$UNG_SCRIPT"
    sed -i 's/^L_BUILDS = .*/L_BUILDS = [200]/' "$UNG_SCRIPT"
    sed -i 's/^L_SEARCHES = .*/L_SEARCHES = [500]/' "$UNG_SCRIPT"

    for DS in $V2_DATASETS; do
        echo ""
        echo "--- UNG: $DS ---"
        python "$UNG_SCRIPT" "$DS" || echo "WARNING: UNG $DS failed, continuing..."
    done

    # 恢复原始参数网格
    mv "${UNG_SCRIPT}.bak" "$UNG_SCRIPT"
    echo ""
    echo "UNG V2 single-config complete!"
}

# ==================== Post-filter ====================
run_postfilter() {
    echo ""
    echo "============================================================"
    echo " Post-filter — 固定配置: M=64, efc=400"
    echo "============================================================"

    PF_SCRIPT="faiss/bash/auto_postfilter_hnsw.py"
    cp "$PF_SCRIPT" "${PF_SCRIPT}.bak"

    sed -i 's/^M_VALUES = .*/M_VALUES = [64]/' "$PF_SCRIPT"
    sed -i 's/^EFC_VALUES = .*/EFC_VALUES = [400]/' "$PF_SCRIPT"

    for DS in $V2_DATASETS; do
        echo ""
        echo "--- Post-filter: $DS ---"
        python "$PF_SCRIPT" "$DS" || echo "WARNING: Post-filter $DS failed, continuing..."
    done

    mv "${PF_SCRIPT}.bak" "$PF_SCRIPT"
    echo ""
    echo "Post-filter V2 single-config complete!"
}

# ==================== SIEVE ====================
run_sieve() {
    echo ""
    echo "============================================================"
    echo " SIEVE — 固定配置: M=32, budget=2.0, hist_pct=0.25"
    echo " ⚠️  必须在 weirdo 节点上运行"
    echo "============================================================"

    for DS in $V2_DATASETS; do
        for SIEVE_SC in original_and original_or original_eq; do
            echo ""
            echo "--- SIEVE: $DS / $SIEVE_SC ---"
            python SIEVE/run_sieve.py \
                --dataset "$DS" \
                --scenario "$SIEVE_SC" \
                --M 32 \
                --ef_construction 40 \
                --index_budget 2.0 \
                --hist_pct 0.25 \
                --num_threads 16 \
            || echo "WARNING: SIEVE $DS/$SIEVE_SC failed, continuing..."
        done
    done

    echo ""
    echo "SIEVE V2 single-config complete!"
}

# ==================== ACORN ====================
run_acorn() {
    echo ""
    echo "============================================================"
    echo " ACORN — 固定配置: M=64, M_beta=96, gamma=8"
    echo "============================================================"

    ACORN_SCRIPT="ACORN/bash/auto_param_search_v2.py"
    cp "$ACORN_SCRIPT" "${ACORN_SCRIPT}.bak"

    # 覆盖 DEFAULT_PARAMS 和 YFCC_PARAMS 中的参数列表
    sed -i 's/"Ms": \[.*\]/"Ms": [64]/g' "$ACORN_SCRIPT"
    sed -i 's/"M_betas": \[.*\]/"M_betas": [96]/g' "$ACORN_SCRIPT"
    sed -i 's/"gammas": \[.*\]/"gammas": [8]/g' "$ACORN_SCRIPT"

    cd ACORN
    for DS in $V2_DATASETS; do
        echo ""
        echo "--- ACORN: $DS ---"
        python bash/auto_param_search_v2.py "$DS" || echo "WARNING: ACORN $DS failed, continuing..."
    done
    cd ..

    mv "${ACORN_SCRIPT}.bak" "$ACORN_SCRIPT"
    echo ""
    echo "ACORN V2 single-config complete!"
}

# ==================== FilteredVamana ====================
run_fv() {
    echo ""
    echo "============================================================"
    echo " FilteredVamana — 固定配置: R=128, Lsearch=2000"
    echo "============================================================"

    FV_SCRIPT="DiskANN/scripts/auto_diskann_original.py"
    cp "$FV_SCRIPT" "${FV_SCRIPT}.bak"

    sed -i 's/^R_VALUES = .*/R_VALUES = [128]/' "$FV_SCRIPT"
    sed -i 's/^L_SEARCH_VALUES = .*/L_SEARCH_VALUES = [2000]/' "$FV_SCRIPT"

    for DS in $V2_DATASETS; do
        echo ""
        echo "--- FilteredVamana: $DS ---"
        python "$FV_SCRIPT" "$DS" || echo "WARNING: FV $DS failed, continuing..."
    done

    mv "${FV_SCRIPT}.bak" "$FV_SCRIPT"
    echo ""
    echo "FilteredVamana V2 single-config complete!"
}

# ==================== StitchedVamana ====================
run_sv() {
    echo ""
    echo "============================================================"
    echo " StitchedVamana — 固定配置: R=128, stitched_R=96, Lsearch=2000"
    echo " (Equality only)"
    echo "============================================================"

    SV_SCRIPT="DiskANN/scripts/auto_stitched_diskann.py"
    cp "$SV_SCRIPT" "${SV_SCRIPT}.bak"

    sed -i 's/^R_VALUES = .*/R_VALUES = [128]/' "$SV_SCRIPT"
    sed -i 's/^STITCHED_R_VALUES = .*/STITCHED_R_VALUES = [96]/' "$SV_SCRIPT"
    sed -i 's/^L_SEARCH_VALUES = .*/L_SEARCH_VALUES = [2000]/' "$SV_SCRIPT"

    for DS in $V2_DATASETS; do
        echo ""
        echo "--- StitchedVamana: $DS ---"
        python "$SV_SCRIPT" "$DS" || echo "WARNING: SV $DS failed, continuing..."
    done

    mv "${SV_SCRIPT}.bak" "$SV_SCRIPT"
    echo ""
    echo "StitchedVamana V2 single-config complete!"
}

# ==================== CAPS ====================
run_caps() {
    echo ""
    echo "============================================================"
    echo " CAPS — 固定配置: nb=1024 (Equality only)"
    echo "============================================================"

    CAPS_SCRIPT="CAPS/bash/auto_caps_all.py"
    cp "$CAPS_SCRIPT" "${CAPS_SCRIPT}.bak"

    sed -i 's/^NB_LIST = .*/NB_LIST = [1024]/' "$CAPS_SCRIPT"

    for DS in $V2_DATASETS; do
        echo ""
        echo "--- CAPS: $DS ---"
        python "$CAPS_SCRIPT" "$DS" || echo "WARNING: CAPS $DS failed, continuing..."
    done

    mv "${CAPS_SCRIPT}.bak" "$CAPS_SCRIPT"
    echo ""
    echo "CAPS V2 single-config complete!"
}

# ==================== NHQ ====================
run_nhq() {
    echo ""
    echo "============================================================"
    echo " NHQ — 固定配置: K=100,L=100,R=300,PL=350 (Equality only)"
    echo "============================================================"

    NHQ_SCRIPT="NHQ/auto_nhq_all.py"
    cp "$NHQ_SCRIPT" "${NHQ_SCRIPT}.bak"

    # NHQ 的 PARAM_SETS 是多行列表，用 Python 替换
    python3 -c "
import re
with open('$NHQ_SCRIPT') as f:
    c = f.read()
c = re.sub(
    r'PARAM_SETS = \[.*?\]',
    'PARAM_SETS = [{\"K\": 100, \"L\": 100, \"iter\": 12, \"S\": 10, \"R\": 300, \"RANGE\": 20, \"PL\": 350, \"B\": 0.4, \"M\": 1}]',
    c, flags=re.DOTALL)
with open('$NHQ_SCRIPT', 'w') as f:
    f.write(c)
"

    for DS in $V2_DATASETS; do
        echo ""
        echo "--- NHQ: $DS ---"
        python "$NHQ_SCRIPT" "$DS" || echo "WARNING: NHQ $DS failed, continuing..."
    done

    mv "${NHQ_SCRIPT}.bak" "$NHQ_SCRIPT"
    echo ""
    echo "NHQ V2 single-config complete!"
}

# ==================== 入口 ====================
case "$METHOD" in
    ung)        run_ung ;;
    postfilter) run_postfilter ;;
    sieve)      run_sieve ;;
    acorn)      run_acorn ;;
    fv)         run_fv ;;
    sv)         run_sv ;;
    caps)       run_caps ;;
    nhq)        run_nhq ;;
    all)
        run_ung
        run_postfilter
        run_acorn
        run_fv
        run_sv
        run_caps
        run_nhq
        echo ""
        echo "⚠️  SIEVE 需要在 weirdo 节点上单独运行："
        echo "    bash analysis/run_v2_single_config.sh sieve"
        ;;
    *)
        echo "Usage: $0 [ung|postfilter|sieve|acorn|fv|sv|caps|nhq|all]"
        echo ""
        echo "Methods (all 3 scenarios: AND/OR/Equality):"
        echo "  ung        UNG (max_degree=96, Lbuild=200, Lsearch=500)"
        echo "  postfilter Post-filter HNSW (M=64, efc=400)"
        echo "  sieve      SIEVE (M=32, budget=2.0) — weirdo node only"
        echo "  acorn      ACORN (M=64, M_beta=96, gamma=8)"
        echo "  fv         FilteredVamana (R=128, Lsearch=2000)"
        echo ""
        echo "Methods (Equality only):"
        echo "  sv         StitchedVamana (R=128, stitched_R=96, Lsearch=2000)"
        echo "  caps       CAPS (nb=1024)"
        echo "  nhq        NHQ (K=100, R=300, PL=350)"
        echo ""
        echo "  all        Run everything except SIEVE"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Done! 结果位置："
echo "  UNG:         UNG-dev/results_original/{dataset}/summary.csv"
echo "  Post-filter: faiss/results_postfilter/{dataset}/summary.csv"
echo "  SIEVE:       SIEVE/results/sieve_{dataset}_{scenario}_*.csv"
echo "  ACORN:       ACORN/data/param_search_{dataset}/results/{dataset}/summary.csv"
echo "  FV:          DiskANN/data_original/results/{dataset}/summary.csv"
echo "  SV:          DiskANN/data_stitched_eq/results/{dataset}/summary.csv"
echo "  CAPS:        CAPS/data/results/{dataset}/summary.csv"
echo "  NHQ:         NHQ/data/results/{dataset}/summary.csv"
echo "========================================="
