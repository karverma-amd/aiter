#!/bin/bash
# Deliver the tuned grouped-MoE tiles into a running serving container.
#
# The container runs the image's own /app/aiter, so edits to a host clone are
# invisible to it. tuned_grouped_fmoe.csv is the path aiter reads by default,
# so copying it in needs no AITER_CONFIG_GROUPED_FMOE.
#
# Usage: apply_moe_tiles.sh <container>
set -euo pipefail

CTR=${1:?usage: apply_moe_tiles.sh <container>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
CSV=$REPO/aiter/configs/tuned_grouped_fmoe.csv
DST=/app/aiter/aiter/configs/tuned_grouped_fmoe.csv

docker cp "$CSV" "$CTR:$DST"
docker exec "$CTR" python3 - <<'EOF'
import csv
rows = list(csv.DictReader(open("/app/aiter/aiter/configs/tuned_grouped_fmoe.csv")))
hit = [r for r in rows if (r["model_dim"], r["inter_dim"], r["expert"]) == ("7168", "2048", "65")]
print(f"landed: {len(rows)} rows, {len(hit)} for the DSR1 EP4 shape")
assert hit, "DSR1 rows missing -- the copy did not take"
EOF

cat <<'EOF'

Export these on the serve command as a second line of defence; any value set
here wins over the CSV and over the built-in defaults:

  export AITER_TDM_TILE_M=128 AITER_TDM_TILE_N=256 AITER_TDM_TILE_K=256
  export AITER_TDM_NUM_BUFFERS=3 AITER_TDM_M_WARP=2 AITER_TDM_N_WARP=4
  export AITER_GROUPED_DEBUG=1   # logs the row each lookup picks

After the run, gate on the trace, not on the throughput number:

  python3 tuning_scripts/check_moe_tile.py <trace>.pt.trace.json.gz
EOF
