#!/bin/bash
# Isolated grouped-MoE (a4w4) tile sweep for DSR1 decode shape.
# Score = gemm1_us + gemm2_us at token=1024 (first init pair).
cd /app/aiter || exit 1
export PYTHONPATH=/app/aiter
TOK=${TOK:-1024}
COMMON="--scenario kernel --data-format a4w4 --experts 65 --topk 9 --model-dim 7168 --inter-dim 2048 --tokens $TOK --act silu --no-check-aot-cache --iters 40"

# tile_m tile_n tile_k num_buffers m_warp n_warp
CANDS=(
"64 256 256 2 1 4"
"64 256 256 3 1 4"
"64 256 256 4 1 4"
"64 128 256 2 1 4"
"64 128 512 2 1 4"
"64 256 512 2 1 4"
"64 256 128 3 1 4"
"64 128 256 4 1 4"
"128 256 256 2 1 4"
"128 256 256 2 2 4"
"128 128 256 2 1 4"
"128 256 128 2 2 4"
"128 256 256 3 2 4"
"256 256 256 2 2 4"
"128 512 256 2 2 8"
"64 512 256 2 1 8"
)

printf "%-28s %10s %10s %10s\n" "tile(m,n,k,nb,mw,nw)" "gemm1_us" "gemm2_us" "sum_us"
for c in "${CANDS[@]}"; do
  read -r TM TN TK NB MW NW <<< "$c"
  out=$(AITER_TDM_TILE_M=$TM AITER_TDM_TILE_N=$TN AITER_TDM_TILE_K=$TK \
        AITER_TDM_NUM_BUFFERS=$NB AITER_TDM_M_WARP=$MW AITER_TDM_N_WARP=$NW \
        python3 op_tests/flydsl_tests/test_flydsl_grouped_gemm.py $COMMON 2>&1)
  g1=$(echo "$out" | grep -m1 "gemm1: us" | sed -E 's/.*us = ([0-9.]+).*/\1/')
  g2=$(echo "$out" | grep -m1 "gemm2: us" | sed -E 's/.*us = ([0-9.]+).*/\1/')
  if [ -z "$g1" ] || [ -z "$g2" ]; then
    err=$(echo "$out" | grep -iE "error|assert|traceback" | head -1)
    printf "%-28s %10s %10s %10s  %s\n" "$c" "FAIL" "FAIL" "-" "${err:0:60}"
  else
    sum=$(python3 -c "print(f'{$g1+$g2:.2f}')")
    printf "%-28s %10s %10s %10s\n" "$c" "$g1" "$g2" "$sum"
  fi
done
