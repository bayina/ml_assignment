# Mini Attention & Transformer Encoder Project

This mini-project contains two small modules that implement:

1. **Scaled Dot-Product Attention (NumPy)**
2. **A Simple Transformer Encoder Block (PyTorch)**

The files are heavily commented and include docstrings so you can understand
what each line is doing.

---

## Student Details

| Name | Student ID            |
|-------------|-----------------|
| Chandhini Bayina         | #700756775   |

---


## Files

- `scaled_dot_product_attention.py`  
  Implements the **scaled dot-product attention** mechanism:

  \[
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
  \]

  - Written with **NumPy**.
  - Provides:
    - `softmax` function.
    - `scaled_dot_product_attention(q, k, v)` function.
  - The `__main__` block contains a small demo showing shapes.

- `transformer_encoder_block.py`  
  Implements a **simplified Transformer encoder block** in **PyTorch**:

  Components:
  - Multi-head self-attention (`nn.MultiheadAttention`)
  - Feed-forward network: `Linear -> ReLU -> Linear`
  - Residual connections (skip connections)
  - Layer normalization (Add & Norm layers)

  Default settings:
  - `d_model = 128`
  - `num_heads = 8`

  The `__main__` block runs a forward pass on a batch of
  **32 sentences × 10 tokens** and prints the resulting shape.

---

## Requirements

Install dependencies (for example with `pip`):

```bash
pip install numpy torch
```
---

## Usage
1. Scaled Dot-Product Attention (NumPy)

```python

attn_weights, context = scaled_dot_product_attention(q, k, v)

print(attn_weights.shape)  # (2, 4, 4)
print(context.shape)       # (2, 4, 8)
```

2. Transformer Encoder Block (PyTorch)

Run the built-in shape test:

```python
batch_size = 32
seq_len = 10
d_model = 128

x = torch.randn(batch_size, seq_len, d_model)
encoder_block = TransformerEncoderBlock(d_model=d_model, num_heads=8)

out = encoder_block(x)
print(out.shape)  # torch.Size([32, 10, 128])
```

## Output