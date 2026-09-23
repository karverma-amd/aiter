import csv, os

SRC_HDR = "/app/aiter/aiter/configs/tuned_grouped_fmoe.csv"
OUT = "/app/aiter/aiter/configs/dsr1_gfx1250_grouped_fmoe.csv"

with open(SRC_HDR, newline="") as f:
    header = next(csv.reader(f))

# bucket -> (tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)  [best from sweep]
WIN = {
    4:    (64, 256, 256, 1, 4, 3),
    16:   (64, 256, 128, 1, 4, 3),
    32:   (64, 128, 256, 1, 4, 2),
    64:   (64, 128, 256, 1, 4, 2),
    128:  (64, 128, 256, 1, 4, 2),
    256:  (64, 128, 256, 1, 4, 2),
    512:  (128, 256, 256, 2, 4, 3),
    1024: (128, 256, 256, 2, 4, 3),
    2048: (128, 256, 256, 2, 4, 3),
}

# Fixed match keys for the DSR1 decode a4w4 grouped MoE (cu_num/ep_fused left
# blank = wildcard so the row serves both EP and non-EP paths / any cu count).
BASE = {
    "gfx": "gfx1250",
    "token": None,
    "model_dim": 7168,
    "inter_dim": 2048,
    "expert": 65,
    "topk": 9,
    "act_type": "ActivationType.Silu",
    "dtype": "torch.bfloat16",
    "q_dtype_a": "torch.float4_e2m1fn_x2",
    "q_dtype_w": "torch.float4_e2m1fn_x2",
    "q_type": "QuantType.per_1x32",
}

rows = []
for tok in sorted(WIN):
    tm, tn, tk, mw, nw, nb = WIN[tok]
    row = {c: "" for c in header}
    row.update({k: ("" if v is None else str(v)) for k, v in BASE.items()})
    row["token"] = str(tok)
    row["tile_m"] = str(tm)
    row["tile_n"] = str(tn)
    row["tile_k"] = str(tk)
    row["m_warp"] = str(mw)
    row["n_warp"] = str(nw)
    row["num_buffers"] = str(nb)
    rows.append(row)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    w.writerows(rows)
print("wrote", OUT, "rows=", len(rows))
