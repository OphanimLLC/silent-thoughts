# silent-thoughts — HuggingFace Space
#
# Tab 1: findings gallery — recorded probes of a 26B chat model (the Italy
#        catch, raw-mode failure, Lincoln cluster, steering asymmetry).
# Tab 2: live probe — Qwen2.5-0.5B-Instruct + a fitted Jacobian lens, on CPU.
#
# See https://github.com/OphanimLLC/silent-thoughts for the full workbench.

import glob
import json
import math
import os
import threading

import gradio as gr

import render

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LENS_REPO = os.environ.get("ST_LENS_REPO", "MildHotSauce/jlens-qwen2.5-0.5b-instruct")

_STATE = {"ready": False, "error": None}
_MODEL = _TOK = _LENS = None
_LOCK = threading.Lock()


def _load():
    global _MODEL, _TOK, _LENS
    try:
        import torch
        import transformers
        import jlens
        from huggingface_hub import hf_hub_download

        lens_path = hf_hub_download(LENS_REPO, "lens.pt")
        _LENS = jlens.JacobianLens.load(lens_path)
        _TOK = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
        _MODEL = jlens.from_hf(hf, _TOK)
        _STATE["ready"] = True
    except Exception as e:  # surfaced in the UI
        _STATE["error"] = f"{type(e).__name__}: {e}"


threading.Thread(target=_load, daemon=True).start()


def _display_token(t):
    return (t or "").replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\\n").replace("<0x0A>", "\\n")


def _resolve_word(word):
    for cand in (" " + word.strip(), word.strip()):
        ids = _TOK.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = _TOK.encode(" " + word.strip(), add_special_tokens=False)
    return ids[0] if ids else None


def probe(prompt, chat, top_k, gen_tokens, track_words):
    import torch

    if not _STATE["ready"]:
        msg = _STATE["error"] or "model still loading — try again in ~30s"
        return f"<div class='stcard'>{render._esc(msg)}</div>"
    prompt = (prompt or "").strip()
    if not prompt:
        return "<div class='stcard'>enter a prompt</div>"
    top_k = int(top_k)
    gen_tokens = int(gen_tokens)

    pins = []
    for w in (track_words or "").split(","):
        w = w.strip()
        if w:
            tid = _resolve_word(w)
            if tid is not None and tid not in pins:
                pins.append(tid)

    text = prompt
    if chat:
        text = _TOK.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    with _LOCK:
        lens_logits, model_logits, input_ids = _LENS.apply(_MODEL, text, max_seq_len=96)
        ids = input_ids[0].tolist()
        tokens = [_display_token(t) for t in _TOK.convert_ids_to_tokens(ids)]

        def row(logits):
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            ln_vocab = math.log(logits.shape[-1])
            top = probs.topk(top_k, dim=-1)
            cells = [
                [{"t": _display_token(_TOK.convert_ids_to_tokens([i.item()])[0]),
                  "id": i.item(), "p": round(v.item(), 4)}
                 for v, i in zip(vals, idxs)]
                for vals, idxs in zip(top.values, top.indices)
            ]
            entropy = -(probs * log_probs).sum(dim=-1)
            conc = [round(c, 4) for c in (1.0 - entropy / ln_vocab).clamp(0, 1).tolist()]
            return cells, conc

        grid, conc_out = {}, {}
        for layer in sorted(lens_logits):
            grid[str(layer)], conc_out[str(layer)] = row(lens_logits[layer])
        fk = str(_MODEL.n_layers - 1)
        grid[fk], conc_out[fk] = row(model_logits)

        output = None
        if gen_tokens:
            attn = torch.ones_like(input_ids)
            with torch.no_grad():
                gen = _MODEL._hf_model.generate(
                    input_ids, attention_mask=attn, max_new_tokens=gen_tokens,
                    do_sample=False, pad_token_id=_TOK.pad_token_id or _TOK.eos_token_id)
            new_ids = gen[0][input_ids.shape[1]:].tolist()
            output = {"text": _TOK.decode(new_ids, skip_special_tokens=True), "ids": new_ids}

    r = {"tokens": tokens, "token_ids": ids, "n_layers": _MODEL.n_layers,
         "layers": sorted(int(l) for l in grid), "grid": grid, "conc": conc_out,
         "output": output}

    extra = ""
    if pins:
        # rank-across-layers table for tracked words
        rows = []
        with _LOCK:
            for tid in pins:
                tok = _display_token(_TOK.convert_ids_to_tokens([tid])[0])
                cells = []
                for layer in sorted(lens_logits):
                    logits = lens_logits[layer]
                    ref = logits[:, tid : tid + 1]
                    rank = int(((logits > ref).sum(dim=-1) + 1).min().item())
                    cells.append(f"<td>{rank if rank < 1000 else str(round(rank/1000))+'k'}</td>")
                rows.append(f"<tr><th>{render._show_tok(tok)}</th>{''.join(cells)}</tr>")
        head = "".join(f"<th>L{l}</th>" for l in sorted(lens_logits))
        extra = (f"<div class='stcard'><div class='stlabel'>Tracked words — best rank per layer (1 = top of vocabulary)</div>"
                 f"<div class='stscroll'><table class='stgrid'><thead><tr><th></th>{head}</tr></thead>"
                 f"<tbody>{''.join(rows)}</tbody></table></div></div>")

    return render.render_probe(r) + extra


def render_steering_example(path):
    with open(path) as f:
        d = json.load(f)
    m = d.get("meta", {})

    def table(title, runs, baseline):
        rows = "".join(
            f"<tr><td>{r['transport']}</td><td>L{r['layer']}</td><td>{r['strength']}</td>"
            f"<td style='font-family:monospace'>{render._esc(r['steered'])}</td></tr>" for r in runs)
        return (f"<div class='stcard'><div class='stlabel'>{title} — baseline: “{render._esc(baseline)}”</div>"
                f"<div class='stscroll'><table class='stgrid'><thead><tr><th>transport</th><th>layer</th>"
                f"<th>strength</th><th>steered output</th></tr></thead><tbody>{rows}</tbody></table></div></div>")

    out = (f"<div class='stcard'><div class='stlabel'>{render._esc(m.get('title'))}</div>"
           f"<div class='sthint'>{render._esc(m.get('notes'))}</div></div>")
    out += table(f"26B MoE — push “{d['push']}” vs “{d['against']}”", d["runs"], d["baseline"])
    rep = d.get("replication_qwen2.5_0.5b")
    if rep:
        out += table("Replication — Qwen2.5-0.5B (dense)", rep["runs"], rep["baseline"])
    return out


EXAMPLES = {}
for p in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "examples", "*.json"))):
    name = os.path.basename(p)
    if name == "steering.json":
        EXAMPLES["Steering: reading is not writing"] = ("steer", p)
    else:
        with open(p) as f:
            EXAMPLES[json.load(f).get("meta", {}).get("title", name)] = ("probe", p)


def show_example(title):
    kind, path = EXAMPLES[title]
    return render_steering_example(path) if kind == "steer" else render.render_example(path)


with gr.Blocks(title="silent-thoughts") as demo:
    gr.Markdown(
        "# silent-thoughts\n"
        "**Watch a language model's thoughts form before it speaks** — an interactive workbench for "
        "[Anthropic's Jacobian lens](https://github.com/anthropics/jacobian-lens). "
        "Code, method, and findings: [github.com/OphanimLLC/silent-thoughts](https://github.com/OphanimLLC/silent-thoughts).")

    with gr.Tab("Findings gallery (26B model)"):
        gr.Markdown(
            "Recorded probes of a 26B instruction-tuned MoE. The headline: asked a two-hop riddle, the model "
            "held **` Italy` at rank 1 of ~262k for 11 straight layers — silently, with an empty thought channel** — "
            "before saying a word.")
        sel = gr.Dropdown(choices=list(EXAMPLES), value=list(EXAMPLES)[0] if EXAMPLES else None, label="example")
        ex_html = gr.HTML(show_example(list(EXAMPLES)[0]) if EXAMPLES else "")
        sel.change(show_example, sel, ex_html)

    with gr.Tab("Live probe (Qwen2.5-0.5B, CPU)"):
        gr.Markdown(
            "Probe a small model live. A probe takes a few seconds on the free CPU hardware. "
            "Tip: leave the chat template on — chat-tuned models produce junk loops in raw completion mode "
            "(that failure is itself finding 4 in the gallery).")
        prompt = gr.Textbox(label="prompt", value="What is the capital of France? Answer in one word.")
        with gr.Row():
            chat = gr.Checkbox(label="chat template", value=True)
            top_k = gr.Slider(1, 10, value=5, step=1, label="top-k")
            gen_tokens = gr.Slider(0, 48, value=16, step=1, label="say (greedy tokens)")
            track = gr.Textbox(label="track words (comma-separated)", value="Paris")
        btn = gr.Button("Probe →", variant="primary")
        out_html = gr.HTML()
        btn.click(probe, [prompt, chat, top_k, gen_tokens, track], out_html)

    gr.Markdown(
        "Apache-2.0 · adapts [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) "
        "(© Anthropic PBC) · lens fitted on 200 wikitext prompts")

demo.launch(css=render.CSS)
