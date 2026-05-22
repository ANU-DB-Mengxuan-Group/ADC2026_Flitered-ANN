#!/bin/bash
# V2 PF rebuild: synth_192d (~5-8h)
# 单 weirdo 跑. 用法: bash analysis/run_v2_pf_rebuild_synth_192d.sh
set -e
cd ~/benchmarks/discrete
exec bash analysis/run_v2_pf_full_rebuild.sh synth_192d
