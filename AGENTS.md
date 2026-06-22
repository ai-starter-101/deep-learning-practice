# AGENTS.md

## Project Identity

Project path: `/Users/ly/Documents/AI/deep-learning-practice`

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

## Next Useful Actions

- Start `L001: 线性回归与训练闭环`.
- Run the minimal training script and write `notes/L001_线性回归与训练闭环.md`.
