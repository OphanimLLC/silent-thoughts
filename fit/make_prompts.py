# Build the fit corpus: pretraining-like raw text chunks, one JSON string per
# line. The Jacobian-lens paper fits on 1000 x 128-token web-text sequences and
# notes quality saturates around 100 prompts; wikitext-103 is close enough in
# distribution.
#
#   python make_prompts.py [n] [out_path]

import json
import os
import sys

from datasets import load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = sys.argv[2] if len(sys.argv) > 2 else "out/fit-prompts.jsonl"
MIN_CHARS = 600  # ~128+ tokens of prose; fit() truncates to max_seq_len anyway

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

out = []
buf = ""
for row in ds:
    t = row["text"].strip()
    if not t or t.startswith("="):  # skip headings
        continue
    buf += " " + t
    if len(buf) >= MIN_CHARS:
        out.append(buf.strip())
        buf = ""
    if len(out) >= N:
        break

with open(OUT, "w") as f:
    for p in out:
        f.write(json.dumps(p) + "\n")
print(f"wrote {len(out)} prompts to {OUT}")
