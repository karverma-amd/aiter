"""Report the grouped-MoE tile the GPU actually ran, from a torch trace.

The TDM kernel name encodes its tile (a8w4_tdm_fp4_t<M>x<N>x<K>_w<mw>x<nw>_b<nb>),
so this is the only check that distinguishes a tuned run from one that silently
fell back to the built-in default.

Usage: check_moe_tile.py <trace.pt.trace.json.gz> [...]
"""

import collections
import gzip
import json
import re
import sys

DEFAULT_TILE = "t64x256x256_w1x4_b3"
KERNEL_RE = re.compile(r"a8w4_tdm_fp4_(t\d+x\d+x\d+_w\d+x\d+_b\d+)_K(\d+)")


def main(paths):
    fell_back = False
    for path in paths:
        with gzip.open(path, "rt") as f:
            events = json.load(f).get("traceEvents", [])

        agg = collections.defaultdict(lambda: [0, 0.0])
        for e in events:
            m = KERNEL_RE.match(str(e.get("name", "")))
            if e.get("ph") == "X" and m:
                slot = agg[(m.group(1), m.group(2))]
                slot[0] += 1
                slot[1] += float(e.get("dur", 0) or 0)

        print(f"\n{path}")
        if not agg:
            print("  no grouped-MoE TDM kernels in this trace")
            continue
        for (tile, k), (count, dur) in sorted(agg.items()):
            flag = "  <-- DEFAULT, tuning did not apply" if tile == DEFAULT_TILE else ""
            fell_back |= tile == DEFAULT_TILE
            print(f"  K={k:<5} {tile:<24} {count:6d}x  {dur / 1000:8.2f} ms"
                  f"  {dur / count:6.1f} us/launch{flag}")

    return 1 if fell_back else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
