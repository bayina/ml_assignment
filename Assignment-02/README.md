# 📓 Assignment-02: Decision tree, KNN, Performance Evaluation

## Student Details

| Name                     | Student ID      |
|--------------------------|-----------------|
| Chandhini Bayina         | #700756775      |


## Notebook Structure and Detailed Explanation

My work is organized into three major sections, each corresponding to a question in the assignment.  

### **Q7 – Decision Tree**

- **Data Preparation:** Loaded the dataset, handled any preprocessing (feature selection).
- **Model Training:** Implemented a Decision Tree classifier using `sklearn.tree.DecisionTreeClassifier`.
- **Visualization:** Generated a decision tree plot using `sklearn.tree.plot_tree` to make the split criteria interpretable.
- **Results:** Printed classification results, showing accuracy on the training and test sets.
Also,
Discussed signs of underfitting vs overfitting based on the results of training accuracy and test accuracy.

---

### **Q8 – k-Nearest Neighbors**

Focused on the kNN algorithm and decision boundaries:
- **Feature Selection:** Chose two features sepal length & width to simplify visualization in a 2D plane.
- **Model Training:** Used `sklearn.neighbors.KNeighborsClassifier` to train the model for different values of `k`.
- **Visualization:** Plotted the decision boundary and training points, showing how class regions change as `k` varies.
- **Analysis:** Compared results across multiple `k` values k=1, 3, 5, 10, on how the boundaries changes observing underfitting for high `k` and overfitting for very small `k`.

---

### **Q9 – Performance Evaluation**

- **Model Training:** Fixed `k=5` and retrained on all features.
- **Confusion Matrix:** Generated and displayed the confusion matrix with `sklearn.metrics.confusion_matrix` to analyze misclassifications.
- **Metrics:** Computed accuracy, precision, recall, and F1-score to evaluate performance comprehensively.
Also computed ROC curve and AUC.

---

## Key Libraries and Dependencies

- **pandas** – Data handling and preprocessing.
- **numpy** – Numerical operations.
- **scikit-learn** – Model training, evaluation, and visualization utilities.
- **matplotlib** – Plotting decision boundaries, tree diagrams, and metrics.

Dependencies are installed in the first cell via:
```python
%pip install scikit-learn matplotlib pandas
