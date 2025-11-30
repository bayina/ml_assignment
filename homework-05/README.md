# 📘 Mini Attention & Transformer Encoder Project

This mini-project now contains **two Jupyter Notebooks** that implement:

1. **Scaled Dot-Product Attention (NumPy)**
2. **A Simple Transformer Encoder Block (PyTorch)**

Both notebooks include full explanations, inline comments, and demonstration outputs.

---

## 👩‍🎓 Student Details

| Name                 | Student ID     |
| -------------------- | -------------- |
| **Chandhini Bayina** | **#700756775** |

---

# 📁 Files

### ✅ **1. `scaled_dot_product_attention.ipynb`**

Implements the **Scaled Dot-Product Attention** mechanism using **NumPy**.

### 🔢 Formula

[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
]

### Notebook includes:

* Explanation of Q, K, V and tensor shapes
* Numerically stable softmax implementation
* Full attention function
* Shape tests and printed outputs
* Step-by-step markdown cells so the logic is easy to follow

---

### ✅ **2. `transformer_encoder_block.ipynb`**

Implements a **simplified Transformer Encoder Block** using **PyTorch**.

### Components included:

* Multi-head self-attention (`nn.MultiheadAttention`)
* Feed-forward network (`Linear → ReLU → Linear`)
* Layer Norm + residual connections
* Explanation of how each sub-layer works

### Notebook includes:

* Step-by-step markdown explanation
* Full PyTorch implementation
* Test using:

  * **Batch size:** 32 sentences
  * **Seq length:** 10 tokens
  * **Embedding size:** 128
* Output shape verification

---

# 📦 Requirements

Install dependencies:

```bash
pip install numpy torch
```

(If using PyTorch with NumPy 2.x, you may need `numpy<2` for compatibility.)

---

# ▶️ Running the Notebooks

Open them in **VS Code**, **Jupyter Notebook**, or **JupyterLab**.

Recommended kernel setup:

```bash
pip install ipykernel
python -m ipykernel install --user --name ml_assignment --display-name "Python (ML Assignment)"
```

Then select this kernel when opening the notebooks.

---

# 📄 Output

Each notebook prints:

* Attention weights + context shapes (Q1)
* Encoder block input/output shapes (Q2)

Demonstrating that both implementations work correctly.
