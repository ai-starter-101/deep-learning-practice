# Deep Learning Practice

This workspace is for turning fragmented deep learning study into a repeatable practice loop.

The rule is simple: each topic must produce runnable code, a small experiment, a personal explanation, and a short Notion learning record.

## Learning Workflow

1. Choose one small topic, such as `L001: 线性回归与训练闭环`.
2. Create or run the smallest possible PyTorch experiment.
3. Change one or two variables and observe the result.
4. Write local notes in `notes/`.
5. Record experiment details in `experiments/<topic>/`.
6. Summarize the learning in the Notion learning database.

## First Topics

- `L001`: 线性回归与训练闭环
- `L002`: softmax 与交叉熵
- `L003`: PyTorch 训练模板
- `L004`: MLP 分类模型
- `L005`: 过拟合与正则化
- `L006`: embedding 与相似度

## Project Layout

```text
deep-learning-practice/
  AGENTS.md
  README.md
  notebooks/
  src/dl_practice/
  experiments/
  notes/
  prompts/
  templates/
  tests/
  data/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
python -m dl_practice.train
```

## Test

```bash
pytest
```
