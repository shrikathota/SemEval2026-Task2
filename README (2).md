# AFourP @ SemEval-2026 Task 2: Predicting Variation in Emotional Valence and Arousal over Time from Ecological Essays

> 🏆 **4th place** on the Subtask 1 leaderboard (V&A average r_composite: **0.573**)

## Overview

This repository contains the code for team **AFourP**'s submission to [SemEval-2026 Task 2](https://semeval.github.io/SemEval2026/tasks): _Predicting Variation in Emotional Valence and Arousal over Time from Ecological Essays_.

The task involves predicting two continuous affect dimensions — **Valence** (pleasantness, 0–4) and **Arousal** (activation, 0–2) — from longitudinal, self-written ecological essays by U.S. service-industry workers, grounded in the affective circumplex model.

---

## Task Description

### Subtask 1 — Longitudinal Affect Assessment

Given a chronological sequence of essays/feeling words `e₁, e₂, …, eₘ` written by a person, predict a (valence, arousal) score pair for each text entry.

The test set includes two user groups:

- **Seen users** — users present in the training set (evaluated at future timesteps)
- **Unseen users** — users not present during training

### Subtask 2 — Affect Forecasting _(not in this repo)_

Predict the change in affect (Δ) between past and future windows of a user's essay sequence.

---

## Results

### Subtask 1 Leaderboard (top teams)

| Rank  | Team              | Valence (r_composite) | Arousal (r_composite) | V&A Average |
| ----- | ----------------- | --------------------- | --------------------- | ----------- |
| 1     | UKP_Psycontrol    | 0.667                 | **0.554**             | **0.611**   |
| 2     | YNU               | 0.677                 | 0.528                 | 0.603       |
| 3     | cclin             | 0.647                 | 0.527                 | 0.587       |
| **4** | **AFourP (Ours)** | **0.679**             | 0.466                 | 0.573       |

Our approach achieved the **highest Valence r_composite** among the top 4 teams (0.679), placing us 4th overall.

---

## Approach

We fine-tune **RoBERTa-base** as a regression model to jointly predict valence and arousal scores from raw essay text.

### Model Architecture

```
Input Text
    │
    ▼
RoBERTa-base (roberta-base)
    │
    ▼
[CLS] token embedding  (hidden_size = 768)
    │
    ▼
Linear(768 → 2)
    │
    ▼
[pred_valence, pred_arousal]
```

- **Encoder**: `roberta-base` from HuggingFace Transformers
- **Pooling**: `[CLS]` token from the last hidden state
- **Head**: single linear layer mapping to 2 regression outputs
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: AdamW

### Hyperparameters

| Parameter           | Value                        |
| ------------------- | ---------------------------- |
| Model               | `roberta-base`               |
| Max sequence length | 256                          |
| Batch size          | 8                            |
| Epochs              | 3                            |
| Learning rate       | 2e-5                         |
| Hardware            | NVIDIA T4 GPU (Google Colab) |

### Training Loss

| Epoch | Loss   |
| ----- | ------ |
| 1     | 0.7725 |
| 2     | 0.5939 |
| 3     | 0.5013 |

---

## Repository Structure

```
.
├── Task2_first_try.ipynb   # Main training and inference notebook
└── README.md
```

---

## Setup & Usage

### Requirements

```bash
pip install torch transformers pandas scikit-learn tqdm
```

### Data

Download the official SemEval-2026 Task 2 dataset and place the files as follows:

```
SemEval2026/
└── Subtask1/
    ├── train_subtask1.csv    # columns: user_id, text_id, text, valence, arousal
    └── test_subtask1.csv     # columns: user_id, text_id, text
```

Update the paths in the notebook:

```python
TRAIN_CSV = "/path/to/train_subtask1.csv"
TEST_CSV  = "/path/to/test_subtask1.csv"
OUTPUT_CSV = "/path/to/subtask1_submission.csv"
```

### Running the Notebook

Open `Task2.ipynb` in Google Colab (recommended, T4 GPU) or Jupyter and run all cells sequentially. The notebook will:

1. Install dependencies
2. Load and tokenize the training data
3. Fine-tune RoBERTa-base for 3 epochs
4. Run inference on the test set
5. Save predictions to `pred_subtask1.csv`

### Output Format

```
user_id,text_id,pred_valence,pred_arousal
3,256,0.6255,0.5102
3,257,1.0818,0.9948
...
```

---

## Acknowledgements

- Dataset and task organized by the SemEval-2026 Task 2 organizers.
- Ecological essays dataset collected as part of the _Data Science and Alcohol Consumption Study_, ethically approved by an academic IRB.
- Built with [HuggingFace Transformers](https://huggingface.co/transformers/) and [PyTorch](https://pytorch.org/).
