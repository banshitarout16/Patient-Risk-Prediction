# Patient Risk Prediction

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0)

![Gradient Boosting](https://img.shields.io/badge/Gradient%20Boosting-Boosting-yellowgreen)
![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-Model-success)
![KNN](https://img.shields.io/badge/KNN-Classifier-blueviolet)
![Random Forest](https://img.shields.io/badge/Random%20Forest-Ensemble-darkgreen)
![Voting Classifier](https://img.shields.io/badge/Voting%20Classifier-Ensemble-purple)
![GridSearchCV](https://img.shields.io/badge/GridSearchCV-Hyperparameter%20Tuning-orange)
![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)

A Machine Learning project that predicts patient health risk using demographic, lifestyle, and medical attributes. The project covers data exploration, preprocessing, model training, and performance evaluation using multiple classification algorithms.

---

## Dataset

- **Records:** 9,549
- **Features:** 23
- **Target Variable:** `Target` (Binary Classification)

---


### Key Findings

- The dataset is well-balanced, with a nearly equal distribution of low-risk and high-risk patients.
- Exploratory analysis identified **BMI**, **Blood Pressure**, **Cholesterol**, and **Age** as the most influential features associated with patient risk.
- Correlation analysis showed that these health indicators have stronger relationships with the target variable compared to other features.
- Pairwise visualizations indicated that no single feature can accurately distinguish patient risk on its own; however, combining multiple clinical and demographic attributes significantly improves predictive performance.
- These findings support the use of machine learning models to capture complex relationships between multiple health factors for accurate patient risk prediction.

---

## Machine Learning Models

Five machine learning classification algorithms were trained and evaluated to identify the most effective model for patient risk prediction.

- **Logistic Regression** – Baseline linear classification model.
- **K-Nearest Neighbors (KNN)** – Instance-based classifier that predicts based on the nearest data points.
- **Random Forest** – Ensemble learning model using multiple decision trees.
- **Gradient Boosting** – Boosting algorithm that builds sequential decision trees to improve prediction performance.
- **Voting Classifier** – Soft voting ensemble combining Logistic Regression, KNN, and Random Forest to leverage the strengths of multiple models.

---

## Results

The trained models were evaluated using multiple classification metrics, including **Accuracy, Precision, Recall, F1-Score, ROC-AUC**, and **Confusion Matrix** analysis. Hyperparameter tuning with **GridSearchCV** was applied to optimize the best-performing model.

- **Random Forest** achieved the highest performance, with **93.7% Accuracy** and **95.8% Recall** after hyperparameter tuning.
- **Gradient Boosting** also delivered strong predictive performance, closely matching the Random Forest model.
- Ensemble-based methods outperformed individual classifiers such as Logistic Regression and K-Nearest Neighbors (KNN).
- Feature Importance analysis identified **BMI, Blood Pressure, Cholesterol, and Age** as the most influential predictors of patient risk.
- ROC Curve and AUC analysis demonstrated the model's strong ability to distinguish between low-risk and high-risk patients, making it suitable for healthcare risk prediction..

---


## Conclusion

This project demonstrates how machine learning can be used to predict patient health risk from demographic, lifestyle, and clinical data. Exploratory analysis identified BMI, Blood Pressure, Cholesterol, and Age as the most informative features, while ensemble models provided the highest prediction accuracy.



















