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

## Next Useful Actions

- Learn the mathematical form of `MSELoss` and why it fits linear regression.
- Improve experiment repeatability by fixing model initialization and DataLoader shuffle seeds.
- Start `L002: softmax 与交叉熵`.
