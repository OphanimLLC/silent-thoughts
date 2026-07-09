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
    # Byte-level BPE tokens are mangled UTF-8 as raw strings — round-trip
    # through the tokenizer for real characters.
    if not t:
        return ""
    if t.startswith("<") and t.endswith(">"):
        return "\\n" if t == "<0x0A>" else t
    s = _TOK.convert_tokens_to_string([t])
    if t.startswith("▁") and not s.startswith(" "):
        s = " " + s
    return s.replace("\n", "\\n")


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
        # Explicit neutral system message — otherwise Qwen's template injects
        # its branded default ("You are Qwen, created by Alibaba Cloud...").
        # Some templates reject system roles; fall back to user-only.
        try:
            text = _TOK.apply_chat_template(
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
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


def steer(prompt, chat, word, vs_word, layer, strength, transport, gen_tokens):
    """Causal test: inject the word's (centered) direction at one layer, at the
    last position only, and compare greedy continuations. transport='direct'
    reliably flips answers at late layers; 'jacobian' writes through J^T — the
    instructive comparison (see finding 3)."""
    import torch

    if not _STATE["ready"]:
        msg = _STATE["error"] or "model still loading — try again in ~30s"
        return f"<div class='stcard'>{render._esc(msg)}</div>"
    word = (word or "").strip()
    prompt = (prompt or "").strip()
    if not word or not prompt:
        return "<div class='stcard'>need a prompt and a word to push toward</div>"
    layer = int(layer)
    strength = float(strength)
    gen_tokens = max(1, min(int(gen_tokens) or 8, 32))

    tid = _resolve_word(word)
    if tid is None:
        return f"<div class='stcard'>couldn't tokenize {render._esc(word)}</div>"
    W = _MODEL._lm_head.weight.detach()
    w_row = W[tid].float()
    vs_label = ""
    if (vs_word or "").strip():
        vs_tid = _resolve_word(vs_word.strip())
        if vs_tid is None:
            return f"<div class='stcard'>couldn't tokenize {render._esc(vs_word)}</div>"
        w_row = w_row - W[vs_tid].float()
        vs_label = f" vs “{render._esc(_display_token(_TOK.convert_ids_to_tokens([vs_tid])[0]))}”"
    else:
        w_row = w_row - W.mean(dim=0, dtype=torch.float32)
    if transport == "jacobian":
        d = _LENS.jacobians[layer].T @ w_row.cpu()
    else:
        d = w_row.cpu()
    d = d / d.norm()

    text = prompt
    if chat:
        try:
            text = _TOK.apply_chat_template(
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            text = _TOK.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    input_ids = _MODEL.encode(text, max_length=256)
    attn = torch.ones_like(input_ids)
    gen_kw = dict(max_new_tokens=gen_tokens, do_sample=False,
                  pad_token_id=_TOK.pad_token_id or _TOK.eos_token_id)

    def generate():
        with torch.no_grad():
            out = _MODEL._hf_model.generate(input_ids, attention_mask=attn, **gen_kw)
        return _TOK.decode(out[0][input_ids.shape[1]:].tolist(), skip_special_tokens=True)

    def steer_hook(module, inputs, output):
        h = output if torch.is_tensor(output) else output[0]
        h = h.clone()
        h[:, -1, :] += strength * h[:, -1, :].norm(dim=-1, keepdim=True) * d.to(h.dtype)
        return h if torch.is_tensor(output) else (h,) + tuple(output[1:])

    with _LOCK:
        baseline = generate()
        handle = _MODEL.layers[layer].register_forward_hook(steer_hook)
        try:
            steered = generate()
        finally:
            handle.remove()

    changed = steered != baseline
    tok_label = render._esc(_display_token(_TOK.convert_ids_to_tokens([tid])[0]))
    verdict = ("✔ the intervention changed the output — the direction is causally live at this layer"
               if changed else "– no change at this strength")
    tone = "#169455" if changed else "#8a8a92"
    return f"""<div class="stcard"><div class="stlabel">Steer — “{tok_label}”{vs_label} @ L{layer} × {strength} ({render._esc(transport)})</div>
      <div class="stcmp">
        <div><div class="sthd">baseline</div><div class="stsay"><span>{render._esc(baseline)}</span></div></div>
        <div><div class="sthd">steered</div><div class="stsay"><span class="stgen">{render._esc(steered)}</span></div></div>
      </div>
      <div style="font-size:12px;margin-top:6px;color:{tone}">{verdict}</div></div>"""


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
        "**Watch a language model's thoughts form before it speaks — then reach in and steer them.** "
        "An interactive workbench for [Anthropic's Jacobian lens](https://github.com/anthropics/jacobian-lens), "
        "plus the part the paper doesn't do: **closing the causal loop** — inject a concept's direction back "
        "into the residual stream and watch the answer flip (Paris → Rome, live, below). "
        "Code, method, and findings: [github.com/OphanimLLC/silent-thoughts](https://github.com/OphanimLLC/silent-thoughts).")

    with gr.Tab("Findings gallery (26B model)"):
        gr.Markdown(
            "Recorded probes of a 26B instruction-tuned MoE. The headline: asked a two-hop riddle, the model "
            "held **` Italy` at rank 1 of ~262k for 11 straight layers — silently, with an empty thought channel** — "
            "before saying a word.")
        sel = gr.Dropdown(choices=list(EXAMPLES), value=list(EXAMPLES)[0] if EXAMPLES else None, label="example")
        ex_html = gr.HTML(show_example(list(EXAMPLES)[0]) if EXAMPLES else "")
        sel.change(show_example, sel, ex_html)

    with gr.Tab("Live probe + steering (Qwen2.5-0.5B, CPU)"):
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
            "### 🎯 Steer — inject a thought and flip the answer\n"
            "**The lens only *reads* the workspace; this closes the loop and *writes* to it** — the causal test "
            "a readout alone can't give you. A word's (centered) output direction is injected into the residual "
            "stream at one layer, last position only, and the greedy answer is regenerated. "
            "**direct** reliably flips answers at late layers (defaults below flip Paris → Rome); "
            "**jacobian** writes through Jᵀ — on the 26B MoE it *never* steers (finding 3), while on this "
            "dense 0.5B it does work at the last fitted layer. Reading is easy; writing is where it gets interesting.")
        with gr.Row():
            st_word = gr.Textbox(label="push toward", value="Rome")
            st_vs = gr.Textbox(label="against (optional)", value="Paris")
            st_layer = gr.Slider(0, 22, value=22, step=1, label="layer")
            st_str = gr.Slider(0.0, 0.6, value=0.2, step=0.05, label="strength")
            st_tr = gr.Dropdown(["direct", "jacobian"], value="direct", label="transport")
        st_btn = gr.Button("Steer →", variant="primary")
        st_html = gr.HTML()
        st_btn.click(steer, [prompt, chat, st_word, st_vs, st_layer, st_str, st_tr, gen_tokens], st_html)

    gr.Markdown(
        "Built by **Dave Ralston** — [dave@ophanim.ai](mailto:dave@ophanim.ai) · Apache-2.0 · "
        "adapts [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) "
        "(© Anthropic PBC) · lens fitted on 200 wikitext prompts")

demo.launch(css=render.CSS)
