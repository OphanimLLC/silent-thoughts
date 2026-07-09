# silent-thoughts probe server — loads a model + fitted Jacobian lens and
# serves layer x position readouts to the web UI. Stdlib HTTP only.
#
#   python server.py --model Qwen/Qwen2.5-0.5B-Instruct --lens out/lens.pt [--port 8791]
#
#   GET  /health -> {status: loading|ready|error, ...}
#   POST /probe  {prompt, top_k?, max_seq_len?, chat?, gen_tokens?,
#                 pins?: [token_id], pin_words?: [str]}
#   POST /steer  {prompt, word|token_id, layer, vs_word?, strength?,
#                 transport?: "direct"|"jacobian", gen_tokens?, chat?}
#
# Binds 127.0.0.1 only. CORS is open so a file:// or localhost web UI can call
# it. Pin a GPU with CUDA_VISIBLE_DEVICES; CPU works for small models.
#
# Adapted from the reference server in anthropics/jacobian-lens usage patterns;
# see NOTICE.

import argparse
import json
import math
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("ST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
ap.add_argument("--lens", default=os.environ.get("ST_LENS", "out/lens.pt"))
ap.add_argument("--port", type=int, default=int(os.environ.get("ST_PORT", "8791")))
ap.add_argument("--device", default=os.environ.get("ST_DEVICE", "auto"))
ARGS = ap.parse_args()

STATE = {"status": "loading", "error": None, "model": ARGS.model, "started": time.time()}
MODEL = None
TOK = None
LENS = None
PROBE_LOCK = threading.Lock()


def load_everything():
    global MODEL, TOK, LENS
    try:
        import torch
        import transformers
        import jlens

        LENS = jlens.JacobianLens.load(ARGS.lens)
        TOK = transformers.AutoTokenizer.from_pretrained(ARGS.model)
        device = ARGS.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        try:
            hf = transformers.AutoModelForCausalLM.from_pretrained(ARGS.model, dtype=dtype)
        except ValueError:  # multimodal wrappers register elsewhere
            hf = transformers.AutoModelForImageTextToText.from_pretrained(ARGS.model, dtype=dtype)
        MODEL = jlens.from_hf(hf.to(device), TOK)
        STATE["status"] = "ready"
    except Exception as e:
        STATE["status"] = "error"
        STATE["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()


def display_token(tok_str: str) -> str:
    # SentencePiece '▁' and BPE 'Ġ' are leading spaces; newlines are byte tokens.
    return (tok_str or "").replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\\n").replace("<0x0A>", "\\n")


def resolve_word(word):
    """Map a typed word to the single token id the lens should track. Prefers
    the leading-space variant (how words appear mid-sentence); multi-token
    words track their first piece."""
    for cand in (" " + word.strip(), word.strip()):
        ids = TOK.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = TOK.encode(" " + word.strip(), add_special_tokens=False)
    return ids[0] if ids else None


def do_probe(body):
    import torch

    prompt = body["prompt"]
    top_k = min(int(body.get("top_k", 5)), 20)
    max_seq_len = min(int(body.get("max_seq_len", 96)), 256)
    gen_tokens = min(max(int(body.get("gen_tokens", 24)), 0), 128)
    pins = [int(t) for t in body.get("pins", [])][:8]

    resolved_words = []
    for w in [str(w) for w in body.get("pin_words", [])][:8]:
        tid = resolve_word(w)
        if tid is not None and tid not in pins:
            pins.append(tid)
            resolved_words.append({
                "word": w, "id": tid,
                "t": display_token(TOK.convert_ids_to_tokens([tid])[0]),
            })
    pins = pins[:8]

    if body.get("chat"):
        prompt = TOK.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        )

    t0 = time.time()
    lens_logits, model_logits, input_ids = LENS.apply(MODEL, prompt, max_seq_len=max_seq_len)
    ids = input_ids[0].tolist()
    tokens = [display_token(t) for t in TOK.convert_ids_to_tokens(ids)]

    def row(logits):  # [n_pos, vocab] -> per-position topk, concentration, pin ranks
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        ln_vocab = math.log(logits.shape[-1])
        top = probs.topk(top_k, dim=-1)
        cells = [
            [
                {"t": display_token(TOK.convert_ids_to_tokens([i.item()])[0]),
                 "id": i.item(), "p": round(v.item(), 4)}
                for v, i in zip(vals, idxs)
            ]
            for vals, idxs in zip(top.values, top.indices)
        ]
        # Concentration = 1 - H/ln(V) in [0,1]: ~1 when the layer commits to one
        # token, ~0 near-uniform (underdetermined cells surface junk-magnet
        # tokens — see README "signal vs noise").
        entropy = -(probs * log_probs).sum(dim=-1)
        conc = (1.0 - entropy / ln_vocab).clamp(0.0, 1.0)
        conc = [round(c, 4) for c in conc.tolist()]
        pin_out = {}
        for pid in pins:
            ref = logits[:, pid : pid + 1]
            rank = (logits > ref).sum(dim=-1) + 1  # 1 = top token
            pin_out[str(pid)] = {
                "rank": rank.tolist(),
                "p": [round(p, 5) for p in probs[:, pid].tolist()],
            }
        return cells, conc, pin_out

    layers_out, conc_out, pins_out = {}, {}, {}
    for layer in sorted(lens_logits):
        cells, conc_row, pin_row = row(lens_logits[layer])
        layers_out[str(layer)] = cells
        conc_out[str(layer)] = conc_row
        if pin_row:
            pins_out[str(layer)] = pin_row
    final_cells, final_conc, final_pins = row(model_logits)
    final_key = str(MODEL.n_layers - 1)
    layers_out[final_key] = final_cells
    conc_out[final_key] = final_conc
    if final_pins:
        pins_out[final_key] = final_pins

    # Greedy continuation from the same ids the grid was read from, so the
    # first generated token matches the output row's last column.
    output = None
    if gen_tokens:
        try:
            import torch as _t
            attn = _t.ones_like(input_ids)
            with _t.no_grad():
                gen = MODEL._hf_model.generate(
                    input_ids, attention_mask=attn, max_new_tokens=gen_tokens,
                    do_sample=False, pad_token_id=TOK.pad_token_id or TOK.eos_token_id,
                )
            new_ids = gen[0][input_ids.shape[1]:].tolist()
            output = {
                "text": TOK.decode(new_ids, skip_special_tokens=True),
                "tokens": [display_token(t) for t in TOK.convert_ids_to_tokens(new_ids)],
                "ids": new_ids,
            }
        except Exception as e:
            output = {"error": f"{type(e).__name__}: {e}"}

    return {
        "tokens": tokens,
        "token_ids": ids,
        "n_layers": MODEL.n_layers,
        "layers": sorted(int(l) for l in layers_out),
        "grid": layers_out,
        "conc": conc_out,
        "pins": pins_out,
        "resolved_words": resolved_words,
        "output": output,
        "seconds": round(time.time() - t0, 2),
    }


def do_steer(body):
    """Causal test of a lens readout: inject a token direction into the
    residual stream at one layer and compare greedy continuations.

    transport="direct" injects the (centered) unembedding direction itself —
    reliably flips answers at late layers. transport="jacobian" writes through
    J_l^T — in our experiments this NEVER steers (it lands in the junk-magnet
    subspace); it is kept as an instructive failure. `strength` is a fraction
    of the position's own residual norm. Only the last prompt position and the
    generated steps are steered; touching earlier positions corrupts prompt
    comprehension."""
    import torch

    prompt = body["prompt"]
    if body.get("chat"):
        prompt = TOK.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        )
    layer = int(body["layer"])
    if layer not in LENS.source_layers:
        raise ValueError(f"layer {layer} not fitted ({LENS.source_layers[0]}..{LENS.source_layers[-1]})")
    strength = max(min(float(body.get("strength", 0.1)), 1.0), -1.0)
    gen_tokens = min(max(int(body.get("gen_tokens", 24)), 1), 64)

    if body.get("word") is not None:
        tid = resolve_word(str(body["word"]))
        if tid is None:
            raise ValueError(f"couldn't tokenize {body['word']!r}")
    else:
        tid = int(body["token_id"])
    tok_str = display_token(TOK.convert_ids_to_tokens([tid])[0])

    W = MODEL._lm_head.weight.detach()
    w_row = W[tid].float().cpu()
    vs_tid = None
    if body.get("vs_word"):
        vs_tid = resolve_word(str(body["vs_word"]))
        if vs_tid is None:
            raise ValueError(f"couldn't tokenize vs_word {body['vs_word']!r}")
        w_row = w_row - W[vs_tid].float().cpu()
    else:
        w_row = w_row - W.mean(dim=0, dtype=torch.float32).cpu()

    transport = body.get("transport", "direct")
    if transport == "jacobian":
        d = LENS.jacobians[layer].T @ w_row
    elif transport == "direct":
        d = w_row
    else:
        raise ValueError("transport must be direct|jacobian")
    d = (d / d.norm()).to(MODEL.input_device)

    t0 = time.time()
    input_ids = MODEL.encode(prompt, max_length=256)
    attn = torch.ones_like(input_ids)
    gen_kw = dict(max_new_tokens=gen_tokens, do_sample=False,
                  pad_token_id=TOK.pad_token_id or TOK.eos_token_id)

    def generate():
        with torch.no_grad():
            out = MODEL._hf_model.generate(input_ids, attention_mask=attn, **gen_kw)
        return TOK.decode(out[0][input_ids.shape[1]:].tolist(), skip_special_tokens=True)

    baseline = generate()

    def steer_hook(module, inputs, output):
        h = output if torch.is_tensor(output) else output[0]
        h = h.clone()
        h[:, -1, :] += strength * h[:, -1, :].norm(dim=-1, keepdim=True) * d.to(h.dtype)
        return h if torch.is_tensor(output) else (h,) + tuple(output[1:])

    handle = MODEL.layers[layer].register_forward_hook(steer_hook)
    try:
        steered = generate()
    finally:
        handle.remove()

    return {
        "token": tok_str, "token_id": tid, "layer": layer, "strength": strength,
        "transport": transport,
        "vs_token": display_token(TOK.convert_ids_to_tokens([vs_tid])[0]) if vs_tid is not None else None,
        "baseline": baseline, "steered": steered,
        "changed": steered != baseline,
        "seconds": round(time.time() - t0, 2),
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path != "/health":
            return self._json(404, {"error": "no such route"})
        payload = dict(STATE)
        if LENS is not None:
            payload["lens"] = {
                "layers": len(LENS.source_layers),
                "n_prompts": LENS.n_prompts,
                "d_model": LENS.d_model,
            }
        self._json(200, payload)

    def do_POST(self):
        if self.path not in ("/probe", "/steer"):
            return self._json(404, {"error": "no such route"})
        if STATE["status"] != "ready":
            return self._json(503, {"error": f"engine {STATE['status']}", "detail": STATE["error"]})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if not (body.get("prompt") or "").strip():
                return self._json(400, {"error": "prompt required"})
            with PROBE_LOCK:
                result = do_steer(body) if self.path == "/steer" else do_probe(body)
            self._json(200, result)
        except (ValueError, KeyError) as e:
            self._json(400, {"error": f"{type(e).__name__}: {e}"})
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    threading.Thread(target=load_everything, daemon=True).start()
    print(f"silent-thoughts server on 127.0.0.1:{ARGS.port} — model {ARGS.model}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()
