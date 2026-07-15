# Region-Aware Epistemic Uncertainty for Mitigating Hallucinations in Large Vision-Language Models

This repository contains the implementation accompanying my Master's thesis on mitigating hallucinations in Large Vision-Language Models (LVLMs) using **region-aware epistemic uncertainty** and **causal visual-token masking**.

The project evaluates whether suppressing visually uncertain representations reduces object hallucinations while preserving correct visual recognition.

---

# Overview

The proposed framework performs the following steps:

1. Generate semantic region masks
2. Map semantic regions to visual tokens
3. Generate representation-space adversarial attacks
4. Estimate epistemic uncertainty
5. Rank visual tokens by uncertainty
6. Apply causal masking
7. Evaluate hallucination reduction
8. Compare against multiple control experiments

The complete pipeline was evaluated on:

- LLaVA-1.5-7B
- LLaVA-1.5-13B
- Qwen2.5-VL-7B

using the ROHE benchmark.

---

# Repository Structure

```
region-vlm-uncertainty/

├── code/                  # Analysis scripts
├── docs/                  # Notes and experiment logs
├── notebooks/             # Analysis notebooks
├── patches/               # Required source patches
├── scripts/               # HPC scripts
├── qwen_pipeline/
│   ├── code/
│   └── scripts/
│
├── outputs/
│   ├── metrics/
│   └── plots/
│
├── README.md
└── .gitignore
```

Large datasets, model checkpoints, generated attacks, uncertainty tensors, and evaluation outputs are intentionally excluded from version control.

---

# Pipeline

```
ROHE Dataset
      │
      ▼
Semantic Region Masks
      │
      ▼
Visual Token Mapping
      │
      ▼
Representation Attack
      │
      ▼
Epistemic Uncertainty
      │
      ▼
Token Ranking
      │
      ▼
Visual Token Masking

      none
      all
      removed
      context
      background

      │
      ▼
Generation
      │
      ▼
Bootstrap Evaluation
      │
      ▼
Random Control
      │
      ▼
Low-Uncertainty Control
      │
      ▼
Original Image Sanity Check
      │
      ▼
Cross-Model Comparison
```

---

# Models

The repository contains implementations for

| Model |
|--------|
| LLaVA-1.5-7B |
| LLaVA-1.5-13B |
| Qwen2.5-VL-7B |

Although the overall experimental protocol is shared, the intervention differs by architecture.

### LLaVA

Visual tokens are masked inside selected visual-attention layers.

### Qwen2.5-VL

Visual tokens are masked after visual-token merging and before entering the language model.

---

# Experimental Stages

Completed experiments include

- Region-aware visual token mapping
- Representation-space adversarial attacks
- Epistemic uncertainty estimation
- Five-condition causal masking
- Active suppression ablation
- Matched random control
- Matched low-uncertainty control
- Original-image sanity evaluation
- Cross-model comparison

---

# Main Results

## Baseline Hallucination Rate

| Model | Hallucination Rate |
|-------|-------------------:|
| LLaVA-1.5-7B | 79.50% |
| LLaVA-1.5-13B | 88.70% |
| Qwen2.5-VL-7B | 58.24% |

---

## Hallucination Reduction (All Masking)

| Model | Reduction |
|-------|----------:|
| LLaVA-1.5-7B | +3.64 pp |
| LLaVA-1.5-13B | -0.57 pp |
| Qwen2.5-VL-7B | +3.83 pp |

---

## Main Findings

- Global and background masking consistently reduce hallucinations in LLaVA-1.5-7B and Qwen2.5-VL-7B.
- LLaVA-1.5-13B does not reproduce this effect.
- Removed-object and context masking have little or no effect.
- Original-image sanity checks reveal a measurable trade-off between hallucination reduction and correct visual recognition.
- Control experiments show that the observed improvements are not uniquely attributable to uncertainty-guided token selection.

---

# Reproducing the Experiments

The complete pipeline consists of:

1. Region-map generation
2. Adversarial attack generation
3. Epistemic uncertainty computation
4. Five-condition masking evaluation
5. Random-control evaluation
6. Low-uncertainty evaluation
7. Original-image sanity evaluation
8. Cross-model comparison


The Qwen implementation is located under

```
qwen_pipeline/
```

---

# Outputs

The repository includes

```
outputs/metrics/
outputs/plots/final/
```

These contain the final metrics and publication-quality figures used in the thesis.

Large generated tensors, intermediate outputs, and datasets are intentionally omitted.

---

# Requirements

Python 3.9+

Main libraries include

- PyTorch
- Transformers
- NumPy
- Pandas
- Matplotlib
- Pillow
- SciPy

---

# License

This repository is released for academic and research purposes.