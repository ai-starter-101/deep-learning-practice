# AGENTS.md

## Project Identity

Project path: `<local project path>`

Short description: Deep learning practice workspace for building a practical, code-first learning system toward AI / LLM work.

## Project Goal

Build a repeatable learning workflow where each topic produces runnable code, experiment notes, personal explanations, and a short Notion learning record.

## Current Context

- The user has engineering experience and is transitioning toward AI / large model work.
- Previous deep learning study felt fragmented, so this workspace prioritizes small runnable experiments over broad course consumption.
- Preferred learning route: code first, AI explanation, video only for stuck points, textbooks as reference.
- Initial focus: training loop, softmax/cross entropy, PyTorch basics, MLP, overfitting, embeddings and similarity.

## Working Habits For This Project

- Read this file at the start of each new session.
- Keep each learning unit small and numbered as `L001`, `L002`, etc.
- Every completed learning unit should have runnable code, notes, experiment observations, and a Notion summary.
- Use Notion database `https://app.notion.com/p/fd2bbfdc1ee2401c8b8a1d39fb991164?v=b231218b33a048649277d4f6de98aac8` for daily learning summaries.
- Preserve unrelated user changes.
- Prefer simple PyTorch examples before adding frameworks.

## Setup And Verification

- Install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .`
- Run: `python -m dl_practice.train`
- Test: `pytest`

## Decision Log

- 2026-06-22: Created the project as a code-first deep learning practice workspace.
- 2026-06-22: Chose PyTorch official basics plus AI-assisted explanation as the primary learning route; D2L and videos are references.
- 2026-06-23: Completed conceptual walkthrough for `L001`; remote conda setup and experiment runs remain for the next session.
- 2026-06-24: Ran `L001` learning-rate experiments in a remote Linux environment; observed slow convergence at `lr=0.0005`, stable baseline at `lr=0.05`, and divergence at `lr=1.0`.
- 2026-06-24: Clarified why `MSELoss` fits the L001 regression task and how this differs from classification losses such as cross entropy.
- 2026-06-25: Started `L002: softmax 与交叉熵` with a manual logits experiment.
- 2026-06-25: Verified that target class logit advantage maps to higher softmax probability and lower cross entropy loss.
- 2026-06-30: Verified batch-shape semantics for classification and confirmed `CrossEntropyLoss` should receive logits rather than manually-softmaxed probabilities.
- 2026-06-30: Extended `L002` to a minimal real classifier with `nn.Linear`, one-step gradient update, and a short training loop reaching `100%` accuracy on a tiny toy dataset.
- 2026-06-30: Clarified that manual `softmax -> log` is mathematically equivalent to cross entropy, but `log_softmax + NLLLoss` is more numerically stable in implementation.

## Next Useful Actions

- Read through the core `log_softmax + NLLLoss` implementation path to understand the stability trick more concretely.
- Continue L002 with code-level understanding before moving on to the next topic.
