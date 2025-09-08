# 📘 Machine Learning Assignments


This repository contains a collection of **machine learning assignments** implemented from scratch and documented in Jupyter notebooks.  
The goal is to practice fundamental concepts, implement algorithms without high-level libraries (e.g., scikit-learn), and visualize results.

## Student Details

| Name | Student ID            |
|-------------|-----------------|
| Chandhini Bayina         | #700756775   |

---

## 📂 Repository Structure
```
.
├── Assignment-01/
│   └── GradientDescent_vs_NormalEquation.ipynb
├── Assignment-02/
│   └── <notebooks + supporting files>
├── Assignment-07/
│   └── GradientDescent_vs_NormalEquation.ipynb
└── README.md
└── requirements.txt
```

- Each assignment has its own folder.  
- Solutions are mainly Jupyter notebooks (`.ipynb`).  
- Supporting files (datasets, images, reports) are stored alongside the corresponding assignment.

---

## 📑 Table of Contents

| Assignment | Topic | Notebook |
|------------|-------|----------|
| Assignment-01 | Linear Regression: Normal Equation vs Gradient Descent | [GradientDescent_vs_NormalEquation.ipynb](Assignment-01/GradientDescent_vs_NormalEquation.ipynb) |

*(More assignments will be added here as the repo grows.)*

---

## 📝 Example Assignment: Assignment-01

**Topic:** Linear Regression with Normal Equation vs Gradient Descent  

**Tasks:**
- Generate synthetic dataset: \( y = 3 + 4x + \epsilon \), with \( x \sim U[0,5] \), Gaussian noise.  
- Solve using **Normal Equation** (closed form).  
- Solve using **Gradient Descent** (from scratch).  
- Compare results and visualize fitted lines + convergence curve.  

**Deliverables:**
- Notebook: `GradientDescent_vs_NormalEquation.ipynb`  
- Outputs:  
  - Raw data + fitted lines plot  
  - Loss curve (MSE vs iterations)  
  - Printed intercept & slope from both methods  

---

## ⚙️ Requirements

All notebooks rely on standard Python scientific libraries:

- [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/)  

Install them with:

```bash
pip install numpy matplotlib
```

or

```bash
pip install -r requirements.txt
```
---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/bayina/ml_assignment
   cd ml_assignments
   ```

2. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```

3. Navigate to the desired assignment folder and open the `.ipynb` file.  
   Run all cells to reproduce the results.

---

## 🧾 Guidelines

- Each assignment should include:
  - Well-commented code  
  - Plots/visualizations  
  - Short written explanation of results (2–3 sentences)  
- Keep dependencies minimal and consistent.  
- Use separate folders for clarity.

---

## 🚀 Future Work

- Add more ML algorithms from scratch:
  - Logistic Regression  
  - Decision Trees  
  - k-Means Clustering  
  - SVM, Neural Nets, etc.  
- Provide equivalent implementations using popular libraries for comparison.  
- Maintain the Table of Contents as new assignments are added.

---

## 📌 License

This project is for educational purposes.  
Feel free to use or adapt the code for learning and practice.
