---
title: Silent Thoughts and Their Hijacking — A Dangerous Game (PoC)
emoji: 🔦
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Read a model's subconscious, then hijack and rewrite it
---

# Silent Thoughts and Their Hijacking — A Dangerous Game (PoC)

An interactive workbench for [Anthropic's Jacobian lens](https://github.com/anthropics/jacobian-lens)
(*Verbalizable Representations Form a Global Workspace in Language Models*).

- **Findings gallery** — recorded probes of a 26B chat model: the model resolving a two-hop
  riddle silently (` Italy` at rank 1 for 11 layers with an empty thought channel), the
  raw-completion failure mode, and the reading≠writing steering asymmetry.
- **Live probe** — Qwen2.5-0.5B-Instruct with a fitted lens, running on CPU: type a prompt,
  see the layer×position grid of what each layer is disposed to say, the silent-thoughts
  summary, and rank-across-layers for any words you track.

- **Steer** — the causal test, live: inject a word's direction into one layer and watch
  the answer flip (or refuse to — the reading≠writing asymmetry, finding 3).

- **Kernel monitor** (in the full workbench) — read the word the model has *committed to
  say next*, before it emits a token, and flag a prompt-injection whose command commits
  there. The same detect→rewrite loop is, mechanically, a silent censorship tool.

⚠️ **This reads and rewrites a model's decision below the visible text — dual-use.**
See **[DANGERS.md](https://github.com/OphanimLLC/silent-thoughts/blob/main/DANGERS.md)**.

Full workbench, method notes, and honest limitations:
**[github.com/OphanimLLC/silent-thoughts](https://github.com/OphanimLLC/silent-thoughts)**

Built by **Dave Ralston** — dave@ophanim.ai
