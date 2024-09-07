# ArtificialInteligenceIntroduction
This repository contains a series of notebooks that aim to serve as an introduction to the field of artificial intelligence development, exploring the necessary steps for building machine learning models in different use cases.

## Notebook 1: “Prediction of Heart Attack”

### Use Case:
This notebook addresses the case of a health insurance company aiming to predict whether a client is at risk of a heart attack based solely on the data collected during their initial interview. The objective is to use this data to assess the likelihood of the client experiencing a heart attack, with the target variable being the occurrence of a heart attack.

### Objective:
The goal is to build a machine learning model to predict the probability of heart disease based on various health indicators. The dataset selected for this task is the Heart Disease Health Indicators Dataset, which contains 253,680 subjects and 22 health-related features. These indicators include factors like high blood pressure, cholesterol levels, smoking status, physical activity, and more, all of which are important predictors for heart disease.

### Steps:

#### Dataset Selection:
The dataset used comes from Kaggle: Heart Disease Health Indicators Dataset. It includes information on various health parameters observed in patients to assess their risk of myocardial infarction.

#### Preliminary Analysis:
The dataset contains 22 variables, including HeartDiseaseorAttack as the target, and other features such as high blood pressure, cholesterol, smoking habits, physical activity, and more. The target variable is imbalanced, with 90.6% of the data representing patients without heart disease, and 9.4% representing those who have had a heart attack.

#### Dataset Structure:
A detailed inspection of the dataset reveals no missing values, and all categorical variables are already encoded. A brief analysis was performed to inspect the structure of the dataset, including data types, shape, and basic statistics of the features. The imbalance in the target variable was also confirmed.

#### Dataset Preparation:
The dataset was split into features (predictors) and the target variable (HeartDiseaseorAttack). Given that all categorical variables were already encoded, no further preprocessing was necessary regarding missing data.

#### Cross-Validation:
To ensure robust model evaluation, the dataset was split using Stratified K-Fold Cross-Validation with 5 folds. This approach ensures that each fold maintains the class distribution of the target variable, addressing the imbalance between patients with and without heart disease.

#### Model Training:
Several classification algorithms were tested, including RandomForest, ExtraTrees, DecisionTree, GradientBoosting, and AdaBoost classifiers. However, the Logistic Regression model delivered the best results. This outcome is likely because healthcare-related variables often exhibit linear relationships. For example, smoking and alcohol consumption are linked to a higher risk of heart disease, while a healthy diet is associated with lower risk.

Evaluation:
The Logistic Regression model was evaluated using precision, recall, and F1-score metrics. While the model demonstrated strong performance in predicting non-heart disease cases (precision: 0.92, recall: 0.99), it struggled to predict heart disease accurately due to the imbalanced dataset, yielding a recall of 0.13 for the positive class (heart disease cases).

#### Evaluation Results:
Accuracy: 91%
Precision (for non-heart disease): 92%
Recall (for heart disease): 13%
F1-Score (for heart disease): 0.21

### Conclusion:
The Logistic Regression model effectively predicts heart disease risk for non-affected patients but faces challenges in identifying heart disease cases due to the class imbalance. Future improvements may involve addressing this imbalance through techniques such as oversampling, undersampling, or applying alternative algorithms.


