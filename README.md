# Homework 2 - Part B (Programming)
CS5710 Machine Learning - Fall 2025

# Student Info
- Sujan Akena
- 700770399

This repository contains solutions for **Part B** of Homework 2, using the Iris dataset.

## Q7. Decision Tree Classifier
- Trained `DecisionTreeClassifier` with `max_depth = 1, 2, 3`.
- Reported training and test accuracy for each depth.
- **Findings:**
  - Depth=1 underfits (low accuracy on both train/test).
  - Depth=2 improves both accuracies.
  - Depth=3 gives very high accuracy without strong overfitting.

## Q8. kNN Classification (2 features)
- Used only **sepal length** and **sepal width** features.
- Trained `KNeighborsClassifier` with `k=1, 3, 5, 10`.
- Plotted decision boundaries for each k.
- **Findings:**
  - Small k (e.g., k=1) → irregular, jagged boundaries (sensitive to noise).
  - Larger k → smoother boundaries, better generalization but less detail.

## Q9. kNN (k=5) Performance Evaluation
- Trained kNN with `k=5` on all 4 Iris features.
- Reported:
  - Confusion matrix
  - Accuracy, precision, recall, F1 (classification report)
  - ROC curves (one-vs-rest) and per-class + macro AUC
- **Findings:**
  - Accuracy ≈ 98%
  - All classes have AUC ≈ 0.99–1.00
  - Very strong performance on this dataset.
