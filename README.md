# Patient Risk Prediction

A Machine Learning project that predicts patient health risk using demographic, lifestyle, and medical attributes. The project covers data exploration, preprocessing, model training, and performance evaluation using multiple classification algorithms.

---

## Dataset

- **Records:** 9,549
- **Features:** 23
- **Target Variable:** `Target` (Binary Classification)

---

## Exploratory Data Analysis

The dataset was analyzed to understand feature distributions and relationships before model training.

### Key Findings

- The dataset is **well-balanced**, with nearly equal samples in both target classes.
- **BMI** has the strongest positive relationship with the target, indicating it is an important predictor.
- **Blood Pressure** and **Cholesterol** show noticeable differences between the two target classes.
- **Age** also contributes to class separation.
- Pairplot analysis shows that while no single feature perfectly separates the classes, combining multiple health indicators improves prediction performance.
- Correlation analysis suggests that BMI, Blood Pressure, Cholesterol, and Age are among the most influential features.

---

## Machine Learning Models

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Gradient Boosting
- Voting Classifier

---

## Results

The trained models were evaluated using standard classification metrics such as Accuracy, Precision, Recall, and F1-Score.

- Random Forest achieved the best overall performance.
- Gradient Boosting produced results comparable to Random Forest.
- Ensemble learning methods outperformed individual classifiers such as Logistic Regression and KNN.
- The results demonstrate that combining multiple health-related features enables accurate patient risk prediction.

---


## Conclusion

This project demonstrates how machine learning can be used to predict patient health risk from demographic, lifestyle, and clinical data. Exploratory analysis identified BMI, Blood Pressure, Cholesterol, and Age as the most informative features, while ensemble models provided the highest prediction accuracy.