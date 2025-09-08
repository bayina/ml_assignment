# 📓 Assignment-01: Linear Regression with Gradient Descent vs Normal Equation

## Student Details

| Name | Student ID            |
|-------------|-----------------|
| Chandhini Bayina         | #700756775   |

## Notebook Walkthrough

### Cell 1 — Title & Overview (Markdown)
Introduces the assignment: linear regression implemented in two ways — **Normal Equation** and **Gradient Descent**.  
Lists deliverables: scatter plot, fitted lines, loss curve. Only `numpy` and `matplotlib` are used.

---

### Cell 2 — Imports & Install Instructions (Markdown)
Mentions dependencies (`numpy`, `matplotlib`) and how to install them.

---

### Cell 3 — Imports & Reproducibility (Code)
```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
```
Imports libraries and sets seed for reproducible results.

---

### Cell 4 — Dataset Description (Markdown)
Explains how data will be generated:  
- 200 samples  
- \( x \sim U[0,5] \)  
- \( y = 3 + 4x + \epsilon \), with \( \epsilon \sim \mathcal{N}(0,1) \).

---

### Cell 5 — Generate Data (Code)
```python
m = 200
x = np.random.uniform(0.0, 5.0, size=(m, 1))
epsilon = np.random.normal(0.0, 1.0, size=(m, 1))
y = 3.0 + 4.0 * x + epsilon
```
Generates synthetic dataset with noise.

---

### Cell 6 — Add Bias Column Explanation (Markdown)
Explains why we add a column of ones to model intercept \( b \).

---

### Cell 7 — Build Design Matrix (Code)
```python
ones = np.ones((m, 1))
X = np.concatenate([ones, x], axis=1)
```
Creates \( X = [1, x] \) for regression.

---

### Cell 8 — Normal Equation Intro (Markdown)
States formula:  
\(	heta = (X^T X)^{-1} X^T y\).  
Emphasizes using `np.dot`.

---

### Cell 9 — Normal Equation (Code)
```python
XtX = np.dot(X.T, X)
XtX_inv = np.linalg.inv(XtX)
Xt_y = np.dot(X.T, y)
theta_ne = np.dot(XtX_inv, Xt_y)
```
Computes closed-form solution. Prints intercept & slope.

---

### Cell 10 — Loss/Metric Formulas (Markdown)
Explains definitions:  
- MSE: \( rac{1}{m}||X	heta - y||^2 \)  
- Gradient: \( rac{2}{m} X^T (X	heta - y) \)  
- \( R^2 \) score.

---

### Cell 11 — Helper Functions (Code)
Defines reusable functions:  
- `predict()`  
- `mse()`  
- `r2_score()`  
- `grad_mse()`  

Keeps code clean for later.

---

### Cell 12 — Gradient Descent Setup (Markdown)
Describes GD config:  
- Init \(	heta = [0,0]\)  
- Learning rate = 0.05  
- Iterations = 1000  
- Track loss per iteration.

---

### Cell 13 — Gradient Descent Loop (Code)
```python
def gradient_descent(X, y, lr=0.05, num_iters=1000):
    theta = np.zeros((X.shape[1], 1))
    losses = []
    for i in range(num_iters):
        g = grad_mse(X, y, theta)
        theta = theta - lr * g
        losses.append(mse(X, y, theta))
    return theta, np.array(losses)
```
Runs gradient descent, returns final parameters and MSE history. Prints intercept & slope.

---

### Cell 14 — Metrics & Comparison Intro (Markdown)
Explains that we will compute MSE and \(R^2\) for both Normal Equation and Gradient Descent.

---

### Cell 15 — Compute Metrics (Code)
```python
mse_ne = mse(X, y, theta_ne)
mse_gd = mse(X, y, theta_gd)
r2_ne = r2_score(X, y, theta_ne)
r2_gd = r2_score(X, y, theta_gd)
```
Compares both methods quantitatively. Results should be very close.

---

### Cell 16 — Plots Intro (Markdown)
Describes plots to follow:  
1. Data + fitted lines.  
2. Loss curve for GD.

---

### Cell 17 — Plot Data & Fitted Lines (Code)
- Plots raw data scatter.  
- Adds fitted line from Normal Equation.  
- Adds fitted line from Gradient Descent.  
- Both lines overlap closely.

---

### Cell 18 — Plot GD Loss Curve (Code)
Plots MSE vs iteration count. Shows monotonic convergence of gradient descent.

---

### Cell 19 — Final Notes (Markdown)
Summarizes insights:  
- Normal Equation gives exact closed-form solution but can be computationally expensive for large datasets.  
- Gradient Descent is iterative, converges gradually, but scales to large datasets and high dimensions.

---

## 📌 Summary
- Both approaches give nearly identical results.  
- Small differences are due to numerical optimization.  
- Normal Equation is better for small datasets; Gradient Descent is more scalable.  

---
