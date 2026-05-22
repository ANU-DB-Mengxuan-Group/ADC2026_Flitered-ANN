#!/bin/bash
# V2 PF rebuild: dbpedia560k (~18-22h)
# 单 weirdo 跑. 用法: bash analysis/run_v2_pf_rebuild_dbpedia560k.sh
set -e
cd ~/benchmarks/discrete
exec bash analysis/run_v2_pf_full_rebuild.sh dbpedia560k
