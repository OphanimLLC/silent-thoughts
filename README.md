# silent-thoughts

**An interactive workbench for the Jacobian lens — watch a language model's thoughts form before it speaks.**

Built on Anthropic's [jlens / jacobian-lens](https://github.com/anthropics/jacobian-lens) reference implementation ([*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)). The lens linearly transports a residual-stream vector at any layer into the final-layer basis and decodes it with the model's own unembedding — so every cell in the grid below is "the word this layer is disposed to say," *before anything is said*.

This repo adds an interactive workbench on top (probe grid, signal-vs-noise scoring, automatic silent-thought mining, concept spotlight, causal steering) and reports what we found running it against a 26B instruction-tuned MoE on a single H100.

## The headline

Ask a 26B chat model *"What currency is used in the country shaped like a boot?"* and probe the prompt's final forward pass — before one token is generated, while its thought channel is empty:

- **` Italy` — the silent middle hop — is rank 1 of ~262k vocabulary for 11 consecutive layers (L18–L28)**, parked on the chat-template scaffolding positions.
- **` eurozone` / ` euro` — the answer — is staged right behind it** (rank 1–6 across L19–L28).

The model resolved *boot → Italy → euro* entirely silently; generation just reads the workspace out. The `Silent thoughts` panel surfaces this automatically — no manual token hunting. (`examples/boot-riddle-chat.json`)

**And the part nobody does with a lens: we close the causal loop.** A readout alone is correlational — so the workbench can *write back*: inject a concept's (centered) output direction into the residual stream at one layer and regenerate. At late layers this flips the answer live (*What is the capital of France?* → **Rome**), while the same injection through the lens transport `Jᵀ` fails on the 26B — reading and writing are not symmetric (finding 3). The lens becomes tweezers, not just a microscope.

## Findings

### 1. Signal vs. noise: how to not fool yourself with a logit-style lens

Mid-layer readouts are full of high-norm "junk magnet" tokens (obscure foreign fragments, code tokens) that surface wherever a cell's activation is underdetermined — they look spooky and mean nothing. Three cheap statistics separate them from real thoughts, all validated against known-answer probes:

- **Concentration** `1 − H/ln|V|` per cell (server-side): catches diffuse mush, but *not* peaked junk — junk magnets can hit p≈0.77.
- **Vertical persistence**: a real thought holds the same top token for ≥3 consecutive layers with a committed peak; junk flickers for 1–2 layers however confident it looks.
- **Late survival**: genuinely staged concepts stay near the top of the readout into the last ~6 layers; persistent mid-stack junk dies by ~2/3 depth. This single gate cleanly separated `Italy/eurozone/assassination` from every junk magnet in our probes.

**Negative result we kept:** unembedding-row norm does *not* discriminate junk (a junk magnet sat at the 72nd percentile while the correct answer token sat at the 83rd). Don't build a norm blocklist.

### 2. The silent-thoughts miner

Combine the three statistics and you get an automatic workspace summary: every token (not from the prompt) that held a top-k readout spot for ≥3 straight layers in a committed cell and survived late. On our probes it returns *only* semantic content:

| probe | silent thoughts found |
|---|---|
| boot riddle (chat) | **Italy · 意大利 · イタリア** · eurozone · shape · país — the hidden hop, held in three scripts |
| "Abraham Lincoln was killed by" (raw) | assassinated · assassination · **John · जॉन · 约翰** · Abram · assailant |
| boot riddle (raw completion) | only junk fragments — **no semantic content survives at all** |

The model stages concepts multilingually (Italy and John each surface in three scripts), and the raw-completion row is a diagnosis at a glance — see finding 4.

### 3. Reading ≠ writing: the average Jacobian is a good reader and a bad writer

The lens readout is correlational; we wanted the causal test. If the workspace reading is real, injecting the concept's direction back into the residual stream should change the output.

- **On the 26B MoE, writing through the lens transport fails outright.** `J_lᵀ · w_token` (centered by mean-row subtraction, or contrastive `w_Rome − w_Paris`) never steered at any layer or strength tested (0.002–0.4, layers 10–27) — it derails into junk-magnet output every time. The transpose of a corpus-averaged Jacobian concentrates its gain in the generic high-norm subspace.
- **Direct injection works.** The centered unembedding direction itself, added at the last position at late layers (strength 0.05–0.15 of the residual norm), reliably flips *Paris → Rome* on both models tested. Injection must be restricted to the last prompt position + generated steps; steering all positions destroys prompt comprehension.
- **The asymmetry is graded, not absolute.** On dense Qwen2.5-0.5B, `Jᵀ` *can* flip the answer at the last fitted layer — but needs ~2× the strength direct injection needs and skips the clean-flip window (direct at L22 ×0.15 yields a tidy `" Rome"`; jacobian needs ×0.3 and yields degenerate repetition). Reading stays cheaper and cleaner than writing everywhere we looked; how much worse writing gets appears to grow with scale/sparsity.

Full transcript in `examples/steering.json`; both modes are in the UI (`direct` as the working tool, `jacobian` as the instructive comparison).

### 4. Chat-tuned models cannot think in raw-completion mode

The paper's `Fact: …` prompt style targets base models. On our chat-tuned 26B, the same fact that the model answers perfectly under the chat template produces, as raw completion:

- an **empty workspace** — the grid contains topic words but no answer, at any layer or position;
- a **greedy repetition loop** (`de facto- facto- facto-…`) — the classic argmax failure mode when there is no confident content to emit.

Even trivial one-hop facts fail raw. If you probe an instruction-tuned model with raw prompts, an empty grid means *the answer circuitry never engaged* — not that the model lacks the fact. Both forward passes are in `examples/` for side-by-side comparison.

## Using it

### Fit a lens

```bash
pip install git+https://github.com/anthropics/jacobian-lens torch transformers datasets
python fit/make_prompts.py 300 out/fit-prompts.jsonl
CUDA_VISIBLE_DEVICES=0 python fit/fit_lens.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts out/fit-prompts.jsonl --out out/lens.pt
```

A 0.5B model fits in ~7 minutes on an H100 (200 prompts); a 26B MoE takes ~2¼ hours. Quality saturates around 100 prompts (paper §9.3).

### Serve and explore

```bash
python server/server.py --model Qwen/Qwen2.5-0.5B-Instruct --lens out/lens.pt
# then open web/index.html (or: python -m http.server -d web)
```

The UI gives you the layer×position grid with signal-vs-noise shading, the silent-thoughts panel, 🔦 track-a-word spotlight heatmaps, rank-vs-layer charts for pinned tokens, and the steering panel.

### Demo Space

A hosted demo (findings gallery + live probing of a small model) lives at **huggingface.co/spaces/MildHotSauce/silent-thoughts**.

## What's in here

```
server/   probe + steer HTTP server (any HF decoder; CPU or GPU)
web/      self-contained workbench UI (no build step)
fit/      corpus builder + resumable lens fitting
examples/ recorded probes behind each finding, with notes
space/    the HuggingFace Space app (gallery + live demo)
```

## Honest limitations

- Findings are from **one** 26B gemma-family chat derivative plus Qwen2.5-0.5B-Instruct; we haven't swept architectures. Late-layer direct steering replicated on both; the total `Jᵀ` failure was 26B-only (weak `Jᵀ` steering exists on the 0.5B), so treat finding 3's strong form as MoE/scale-specific until replicated.
- The silent-thoughts thresholds (run ≥3, conc ≥0.5, late bar = last 6 layers) were tuned on a handful of probes on one model, not a benchmark.
- Steering "works" at late layers in the sense of flipping the argmax; outputs under steering are often degenerate (`Rome Rome Rome`). It is a causal probe, not a control method.

## Author

**Dave Ralston** — [dave@ophanim.ai](mailto:dave@ophanim.ai)

## License

Apache-2.0. Adapts code and method from [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache-2.0, © Anthropic PBC) — see `NOTICE`.
