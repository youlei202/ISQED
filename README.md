# Quantifying Model Uniqueness in Heterogeneous AI Ecosystems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper Status](https://img.shields.io/badge/Paper-Under%20Review-red)](https://www.nature.com/natmachintell/)

> **Official implementation of the In-Silico Quasi-Experimental Design (ISQED) framework.**

## 📖 Overview

As AI systems evolve from isolated predictors to complex **ecosystems**, a critical governance challenge has emerged: **Redundancy**. How do we determine if a specific target model contributes genuinely unique behavior, or if it is functionally redundant given existing peer models?

This repository implements **ISQED**, a statistical framework that audits model uniqueness by treating models as subjects in a controlled, in-silico quasi-experiment. By applying matched interventions (Type-B treatment) across the ecosystem, we isolate the intrinsic model identity (Type-A treatment) and quantify the **Peer-Inexpressible Residual (PIER)**.

### Key Theoretical Contributions
* **Active Auditing:** We implement an active sampling strategy that achieves the minimax-optimal sample complexity scaling law: $N \sim d \sigma^2 \gamma^{-2}$.
* **Ecosystem Saturation:** We empirically verify the "phase transition" where uniqueness collapses exponentially once the number of peers exceeds the task dimension.
* **Shapley Impossibility:** We demonstrate cases where Shapley Value attribution fails to detect redundancy, whereas PIER succeeds.

---

## 🛠️ System Architecture

This codebase is built on a highly abstract **Operator Interface**, allowing the same auditing logic to be applied seamlessly to **Linear Models**, **LLMs** (e.g., BERT, Llama), and **Generative Models** (e.g., Diffusion).

### Core Abstractions (`isqed/core.py`)
The framework decouples the mathematical auditing logic from the underlying model implementation:

* **`ModelUnit`**: An abstract wrapper for any input-output system (matrix multiplication, neural forward pass, etc.).
* **`Intervention` ($T$):** The operator defining how inputs are perturbed (e.g., noise injection, token masking).
* **`Scalarizer` ($g$):** The functional that maps high-dimensional outputs to a scalar metric for projection.

### Directory Structure
```text
.
ISQED-Prototype/
│
├── data/                    # All data goes here
│   ├── raw/                 # Immutable original data (e.g., SST-2 dataset)
│   └── processed/           # Data after preprocessing (e.g., tokenized text)
│
├── notebooks/               # Jupyter Notebooks for exploration & plotting
│   ├── 01_scaling_law.ipynb       # Generates Figure 2
│   ├── 02_saturation.ipynb        # Generates Figure 4
│   ├── 03_shapley_paradox.ipynb   # Generates Figure 3
│   └── 04_bert_case_study.ipynb   # Real-world analysis
│
├── results/                 # All outputs go here
│   ├── figures/             # Final images for the paper (.pdf, .svg, .png)
│   │   ├── fig2_scaling.pdf
│   │   └── fig4_saturation.pdf
│   ├── tables/              # Generated LaTeX tables or CSVs
│   │   └── metric_summary.csv
│   └── logs/                # Raw experiment logs (json/yaml)
│
├── isqed/
│   ├── core.py          # Abstract Base Classes (ModelUnit, Intervention)
│   ├── geometry.py      # Convex optimization solver (DISCO algorithm)
│   ├── ecosystem.py     # Container for Target and Peer models
│   ├── synthetic.py     # Synthetic environments (Linear Structural Models)
│   ├── real_world.py    # Adapters for HuggingFace Transformers
│   └── auditing.py      # Implementation of Active vs. Passive auditing strategies
|
├── experiments/
│   ├── exp1_scaling_law.py    # Reproduces Figure 2: Active Auditing Efficiency
│   ├── exp2_saturation.py     # Reproduces Figure 4: Ecosystem Saturation
│   ├── exp3_shapley.py        # Reproduces Figure 3: Shapley vs. PIER
│   └── exp4_bert_audit.py     # Real-world case study on BERT ecosystem
|
├── requirements.txt
└── README.md
└── .gitignore               # Ignore data/ and results/ logs
```

## 🚀 Installation
1. **Clone the repository:**
```Bash
git clone [https://github.com/youlei202/ISQED-Prototype.git](https://github.com/youlei202/ISQED-Prototype.git)
cd ISQED-Prototype
```

2. **Install dependencies:**
```Bash
pip install -r requirements.txt
```
Core requirements: numpy, scipy, cvxpy (for convex projection), torch, transformers.


## 🧩 Usage Example (Custom Model)
You can easily audit your own models by subclassing ModelUnit. Here is an example for a custom PyTorch model:
```Python
from isqed.core import ModelUnit, Intervention
from isqed.ecosystem import Ecosystem
from isqed.geometry import DISCOSolver
import torch
import numpy as np

# 1. Define your wrapper
class MyTorchModel(ModelUnit):
    def __init__(self, model):
        self.model = model
    def _forward(self, tensor_data):
        return self.model(tensor_data).detach().numpy()

# 2. Define intervention
class GaussianNoise(Intervention):
    def apply(self, x, theta):
        return x + torch.randn_like(x) * theta

# 3. Setup Ecosystem
# Assume model_A (target) and model_B/C (peers) are pre-loaded PyTorch models
target = MyTorchModel(model_A)
peers = [MyTorchModel(model_B), MyTorchModel(model_C)]
env = Ecosystem(target, peers)

# 4. Audit
X_probe = torch.randn(1, 10) # Your data
y_t, Y_p = env.batched_query(X_probe, theta=0.5, intervention=GaussianNoise())

# 5. Calculate PIER
weights = DISCOSolver.solve_weights(y_t, Y_p)
pier = DISCOSolver.compute_pier(y_t, Y_p, weights)
print(f"Model Uniqueness (PIER Norm): {np.linalg.norm(pier)}")
```


## 📜 Citation
If you find this code or the ISQED framework useful in your research, please cite our paper:
```
To be announced
```