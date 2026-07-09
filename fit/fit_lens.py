# Fit a Jacobian lens for any HuggingFace decoder model.
#
#   python fit_lens.py --model Qwen/Qwen2.5-0.5B-Instruct \
#       --prompts out/fit-prompts.jsonl --out out/lens-qwen0.5b.pt \
#       [--n 200] [--dim-batch 32] [--ckpt out/fit-ckpt.pt]
#
# Resumable: fit() checkpoints a running sum and skips already-processed
# prompts, so an interrupted run just re-runs this script. Requires the jlens
# package (github.com/anthropics/jacobian-lens) plus torch + transformers.
#
# Pin a specific GPU with CUDA_VISIBLE_DEVICES before launching.

import argparse
import json
import logging
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import transformers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("jlens").setLevel(logging.INFO)
log = logging.getLogger("fit")

import jlens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", required=True, help="one JSON string per line")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--dim-batch", type=int, default=32)
ap.add_argument("--ckpt", default=None)
ap.add_argument("--max-seq-len", type=int, default=128)
args = ap.parse_args()

with open(args.prompts) as f:
    prompts = [json.loads(line) for line in f][: args.n]
log.info("fitting %s on %d prompts", args.model, len(prompts))

tok = transformers.AutoTokenizer.from_pretrained(args.model)
try:
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
except ValueError:  # multimodal wrappers register elsewhere
    hf = transformers.AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
model = jlens.from_hf(hf.cuda(), tok)
log.info("loaded %r", model)

lens = jlens.fit(
    model,
    prompts,
    dim_batch=args.dim_batch,
    max_seq_len=args.max_seq_len,
    checkpoint_path=args.ckpt,
    checkpoint_every=25,
)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
lens.save(args.out)
log.info("saved %r -> %s", lens, args.out)
