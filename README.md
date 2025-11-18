# 📝 Hackathon Project – Environment Setup & Structure Guide

This README documents every step taken to set up the development environment for the hackathon, including Python installation, virtual environment creation, dependency installation, GPU considerations, and the recommended project folder structure.

---

## 📌 1. Python Installation

The system originally had:
`Python 3.11`

The hackathon required Python 3.12, so we installed it separately from the official Python website.

After installation:
`py -0`

showed:

```
-V:3.12 *   Python 3.12 (64-bit)
-V:3.11     Python 3.11 (64-bit)
```

✔ Python 3.12 is installed
✔ Python 3.11 is still available
✔ Existing projects remain unaffected

---

## 📌 2. Creating a Virtual Environment (venv)

Navigate to the hackathon folder:
`cd "D:\Omnie Solutions\Hackathon"`

Create the venv using Python 3.12:
`py -3.12 -m venv spider`

Activate it:
`spider\Scripts\activate`

Upgrade pip & tools:
`pip install --upgrade pip setuptools wheel`

---

## 📌 3. Installing PyTorch (CPU Version)

We chose to keep CPU-only PyTorch for simplicity and stability in the hackathon.
`pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu`

GPU support was optional and discussed, but we intentionally stayed on CPU.

---

## 📌 4. Installing All Project Dependencies

We installed the full dependency list provided in `requirements.txt`:
`pip install -r requirements.txt`

This installed libraries such as:

- `accelerate`
- `catboost`
- `fastapi`
- `lightgbm`
- `pandas`
- `scikit-learn`
- `scipy`
- `statsmodels`
- `transformers`
- `sentence-transformers`
- `xgboost`
- ...and many more

All packages installed successfully.

---

## 📌 5. GPU Mode (Optional Discussion)

Initially, `nvidia-smi` failed due to GPU mode being off.

After switching to NVIDIA GPU mode, the GPU was detected correctly:

- CUDA Version: 12.5
- GPU: NVIDIA GeForce RTX 2050
- Driver Version: 555.97

We concluded:
✔ Installing GPU PyTorch is safe
✔ But for simplicity we kept CPU-only torch
✔ The environment remains stable and ready for CPU-based ML/NLP work

---

## 📌 6. Folder Structure for the Project

Hackathon-folder structure:

```
Hackathon/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── docs/
│   ├── images/
│   ├── ppt/
│   ├── problem-statement/
│   └── report/
│
├── models/
│   ├── preprocessor.pkl
│   ├── svm_sgd_pipeline.joblib
│   ├── xgb_booster.json
│   
│
├── notebooks/
│   └── 01-Main.ipynb
│
├── submissions/
│   ├── submission.xlsx
│   └── submission_XGB.xlsx
│
├── spider/                # sSpider IDE environment (ignored)
│
├── .gitignore
├── README.md
├── requirements.txt
└── requirements_installed.txt

```


---

## 📌 7. Environment Validation Tests

To verify core libraries:
```python
import torch, sklearn, pandas, numpy, transformers
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("sklearn:", sklearn.__version__)
print("transformers:", transformers.__version__)
```

Expected:
- All imports successful
- `cuda_available = False` (CPU mode)
---
## 📌 8. Freezing the Environment

After all packages were installed:
`pip freeze > requirements_installed.txt`

