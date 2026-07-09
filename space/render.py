# HTML renderers shared by the Space app: probe grid, silent thoughts, output.
# Python port of the web UI's scoring (verdicts, runs, silent-thoughts miner)
# so the Gradio app renders the same story from a probe dict.

import html
import json
import math
import re


def _esc(s):
    return html.escape(str(s or ""), quote=True)


def _show_tok(t):
    if t == " ":
        return "␣"
    if t == "\\n":
        return "⏎"
    return _esc(t)


def _cell_shade(p):
    t = max(0.0, min(1.0, p)) ** 0.45
    lvl = round(240 - t * 220)
    ink = "#fafafa" if t > 0.5 else "#141414"
    return f"rgb({lvl},{lvl},{lvl})", ink


def compute_runs(r):
    """Per cell: consecutive-layer run length of the same top-1 id + peak conc."""
    asc = sorted(r["layers"])
    n_pos = len(r["tokens"])
    runs = {L: [None] * n_pos for L in asc}
    for pos in range(n_pos):
        ids = []
        cc = []
        for L in asc:
            cell = r["grid"].get(str(L), [])
            ids.append(cell[pos][0]["id"] if pos < len(cell) and cell[pos] else -999)
            conc_row = (r.get("conc") or {}).get(str(L), [])
            cc.append(conc_row[pos] if pos < len(conc_row) else 0.0)
        i = 0
        while i < len(asc):
            j = i
            while j + 1 < len(asc) and ids[j + 1] == ids[i] and ids[i] != -999:
                j += 1
            length = j - i + 1
            peak = max(cc[i : j + 1])
            for k in range(i, j + 1):
                runs[asc[k]][pos] = (length, peak)
            i = j + 1
    return runs


def verdict(run, conc):
    conc = conc or 0.0
    length, peak = run if run else (1, conc)
    committed = length >= 3 and peak >= 0.6
    if committed and conc >= 0.5:
        return "stable"
    if committed:
        return "stable"
    if conc >= 0.6:
        return "transient"
    return "noise"


def silent_thoughts(r, cap=8):
    """Tokens holding a top-k spot >=3 consecutive layers in a committed cell,
    not prompt words, surviving into the last 6 layers."""
    asc = sorted(L for L in r["layers"] if L < r["n_layers"] - 1)
    prompt_ids = set(r["token_ids"])
    # also drop case/spacing variants of prompt words (' What' vs 'What')
    prompt_words = {t.strip().lower() for t in r["tokens"]}
    said_ids = set(((r.get("output") or {}).get("ids")) or [])
    best = {}
    # Only the trailing positions: that's where the resolved answer stages
    # (causal attention). Mining the whole prompt surfaces the model chewing on
    # its own system prompt ("knowledgeable", "helpful"...) — honest but noise.
    start_pos = max(0, len(r["tokens"]) - 16)
    for pos in range(start_pos, len(r["tokens"])):
        present = {}
        for li, L in enumerate(asc):
            cell = r["grid"].get(str(L), [])
            conc_row = (r.get("conc") or {}).get(str(L), [])
            conc = conc_row[pos] if pos < len(conc_row) else 0.0
            if pos >= len(cell):
                continue
            for k, c in enumerate(cell[pos]):
                present.setdefault(c["id"], []).append((li, k + 1, conc, c["t"]))
        for tid, occ in present.items():
            if tid in prompt_ids:
                continue
            s = (occ[0][3] or "").strip().replace("\\n", "")
            if len(s) < 2 or "<" in s or ">" in s:
                continue
            if s.lower() in prompt_words:
                continue
            if all(not ch.isalpha() for ch in s):
                continue
            i = 0
            while i < len(occ):
                j = i
                while j + 1 < len(occ) and occ[j + 1][0] == occ[j][0] + 1:
                    j += 1
                run = occ[i : j + 1]
                if len(run) >= 3:
                    peak = max(o[2] for o in run)
                    best_rank = min(o[1] for o in run)
                    if peak >= 0.5:
                        cand = {
                            "id": tid, "t": occ[0][3], "pos": pos, "len": len(run),
                            "bestRank": best_rank, "l0": asc[run[0][0]],
                            "l1": asc[run[-1][0]], "said": tid in said_ids,
                        }
                        prev = best.get(tid)
                        if not prev or cand["len"] > prev["len"] or (
                            cand["len"] == prev["len"] and cand["bestRank"] < prev["bestRank"]
                        ):
                            best[tid] = cand
                i = j + 1
    late_bar = r["n_layers"] - 6
    out = [c for c in best.values() if c["l1"] >= late_bar]
    out.sort(key=lambda c: (-c["l1"], -c["len"], c["bestRank"]))
    return out[:cap]


def render_output(r):
    o = r.get("output")
    if not o or o.get("error"):
        return ""
    prompt_text = "".join(r["tokens"][1:]).replace("\\n", " ")
    prompt_text = re.sub(r"<\|[^>]*\|>", " · ", prompt_text)   # hide template markers
    prompt_text = re.sub(r"(\s*·\s*)+", " · ", prompt_text).strip(" ·")
    if len(prompt_text) > 180:  # chat templates prepend a whole system prompt
        prompt_text = "…" + prompt_text[-180:]
    return f"""<div class="stcard"><div class="stlabel">What the model actually says</div>
      <div class="stsay"><span class="stprompt">{_esc(prompt_text)}</span><span class="stgen">{_esc(o["text"])}</span></div></div>"""


def render_silent(r):
    items = silent_thoughts(r)
    if not items:
        return ""
    has_gen = bool(((r.get("output") or {}).get("ids")) or [])
    chips = []
    for it in items:
        tag = ""
        if has_gen:
            tag = ('<span class="stsaid">said</span>' if it["said"]
                   else '<span class="stunsaid">never said</span>')
        chips.append(
            f'<span class="stchip"><b>{_show_tok(it["t"])}</b> · rank {it["bestRank"]}'
            f' · L{it["l0"]}–L{it["l1"]} {tag}</span>')
    return f"""<div class="stcard"><div class="stlabel">Silent thoughts — in the workspace before a word was said</div>
      <div>{''.join(chips)}</div>
      <div class="sthint">Words (not from the prompt) that held a top-k readout spot for ≥3 straight layers in a committed cell and survived into the late layers.</div></div>"""


def render_grid(r, max_cols=48):
    runs = compute_runs(r)
    layers_desc = sorted(r["layers"], reverse=True)
    final_layer = r["n_layers"] - 1
    n_pos = min(len(r["tokens"]), max_cols)

    head = '<tr><th class="stcorner">layer ＼ pos</th>'
    for i in range(n_pos):
        head += f'<th>{i}<br><span class="stdim">{_show_tok(r["tokens"][i])}</span></th>'
    head += "</tr>"

    rows = []
    for layer in layers_desc:
        cells = r["grid"].get(str(layer), [])
        conc_row = (r.get("conc") or {}).get(str(layer), [])
        is_final = layer == final_layer
        row = f'<tr{" class=stfinal" if is_final else ""}><th>{"output L" if is_final else "L"}{layer}</th>'
        for pos in range(n_pos):
            cell = cells[pos] if pos < len(cells) else []
            top = cell[0] if cell else {"t": "", "p": 0}
            conc = conc_row[pos] if pos < len(conc_row) else 0.0
            v = verdict(runs[layer][pos], conc)
            bg, ink = _cell_shade(top["p"])
            tip = " | ".join(f'{_show_tok(c["t"])} {c["p"]*100:.1f}%' for c in cell[:5])
            row += (f'<td class="stcell st{v}" style="background:{bg};color:{ink}" '
                    f'title="L{layer} pos{pos}: {_esc(tip)}">{_show_tok(top["t"])}</td>')
        rows.append(row + "</tr>")

    truncated = "" if len(r["tokens"]) <= max_cols else f'<div class="sthint">(showing first {max_cols} of {len(r["tokens"])} positions)</div>'
    return f"""<div class="stcard"><div class="stlabel">Readout grid — darker = the layer is more confident; left stripe = a stable thought; faint italics = junk</div>
      <div class="stscroll"><table class="stgrid"><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>{truncated}</div>"""


CSS = """
/* Gradio's light/dark theme CSS fights ours with higher specificity, so every
   element is self-contained with !important — readable on ANY background. */
.stcard { border: 1px solid #d8d8d2 !important; border-radius: 12px; padding: 12px 14px; margin: 10px 0; background: #fff !important; color: #16161a !important; }
.stcard a { color: #7a3ea3 !important; }
.stlabel { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; color: #8a8a92 !important; margin-bottom: 8px; }
.stsay { font-size: 15px; line-height: 1.7; border-left: 3px solid #7a3ea3; padding-left: 12px; word-break: break-word; color: #16161a !important; }
.stbody { color: #16161a !important; }   /* dark body text — needs a class to beat Gradio's theme */
.stprompt { color: #6b6b72 !important; }
.stgen { font-weight: 600; background: #7a3ea3 !important; color: #fff !important; border-radius: 4px; padding: 1px 7px; }
.stchip { display: inline-block; margin: 0 6px 6px 0; padding: 4px 11px; border-radius: 999px; border: 1px solid #d8d8d2; font-size: 13px; font-family: ui-monospace, monospace; background: #f4f4f1 !important; color: #4b4b52 !important; }
.stchip b { color: #16161a !important; }
.stsaid { color: #169455 !important; font-size: 11px; } .stunsaid { color: #b07914 !important; font-size: 11px; font-style: italic; }
.sthint { font-size: 12px; color: #57575e !important; margin-top: 6px; }
.stscroll { overflow: auto; max-height: 640px; border: 1px solid #d8d8d2; border-radius: 10px; }
table.stgrid { border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 11px; }
table.stgrid th { background: #f4f4f1 !important; color: #4b4b52 !important; font-weight: 500; padding: 4px 6px; white-space: nowrap; position: sticky; top: 0; }
table.stgrid td { background: #fff !important; color: #16161a !important; padding: 2px 6px; }
table.stgrid tbody th { position: sticky; left: 0; text-align: right; }
table.stgrid th.stcorner { left: 0; z-index: 2; }
.stdim { color: #9a9aa2; }
.stcell { min-width: 56px; max-width: 86px; height: 26px; padding: 1px 4px; border: 1px solid #e3e3de; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.stcell.ststable { box-shadow: inset 3px 0 0 #7a3ea3; }
.stcell.stnoise { opacity: .38; font-style: italic; }
.stcell.sttransient { opacity: .72; }
tr.stfinal th, tr.stfinal td { border-top: 2px solid #b9b9b2; }
.stcmp { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 8px; }
.sthd { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; color: #8a8a92 !important; margin-bottom: 4px; }
@media (max-width: 800px) { .stcmp { grid-template-columns: 1fr; } }
"""


def render_probe(r):
    return render_output(r) + render_silent(r) + render_grid(r)


def render_example(path):
    with open(path) as f:
        d = json.load(f)
    m = d.get("meta", {})
    head = f"""<div class="stcard"><div class="stlabel">{_esc(m.get("title", path))}</div>
      <div class="stbody" style="font-size:14px"><b>Prompt:</b> {_esc(m.get("prompt", ""))} <span class="stdim">({'chat template' if m.get('chat') else 'raw completion'})</span></div>
      <div class="sthint">{_esc(m.get("notes", ""))}</div>
      <div class="sthint">Model: {_esc(m.get("model", ""))}</div></div>"""
    return head + render_probe(d)
