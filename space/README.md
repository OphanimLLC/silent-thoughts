---
title: silent-thoughts
emoji: 🔦
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Watch a language model's thoughts form before it speaks
---

# silent-thoughts

An interactive workbench for [Anthropic's Jacobian lens](https://github.com/anthropics/jacobian-lens)
(*Verbalizable Representations Form a Global Workspace in Language Models*).

- **Findings gallery** — recorded probes of a 26B chat model: the model resolving a two-hop
  riddle silently (` Italy` at rank 1 for 11 layers with an empty thought channel), the
  raw-completion failure mode, and the reading≠writing steering asymmetry.
- **Live probe** — Qwen2.5-0.5B-Instruct with a fitted lens, running on CPU: type a prompt,
  see the layer×position grid of what each layer is disposed to say, the silent-thoughts
  summary, and rank-across-layers for any words you track.

Full workbench, method notes, and honest limitations:
**[github.com/OphanimLLC/silent-thoughts](https://github.com/OphanimLLC/silent-thoughts)**
