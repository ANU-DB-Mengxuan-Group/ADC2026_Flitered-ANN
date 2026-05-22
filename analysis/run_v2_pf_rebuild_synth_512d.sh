#!/bin/bash
# V2 PF rebuild: synth_512d (~13-18h)
# 单 weirdo 跑. 用法: bash analysis/run_v2_pf_rebuild_synth_512d.sh
set -e
cd ~/benchmarks/discrete
exec bash analysis/run_v2_pf_full_rebuild.sh synth_512d
