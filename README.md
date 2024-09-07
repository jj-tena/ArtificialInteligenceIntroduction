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

#### Evaluation:
The Logistic Regression model was evaluated using precision, recall, and F1-score metrics. While the model demonstrated strong performance in predicting non-heart disease cases (precision: 0.92, recall: 0.99), it struggled to predict heart disease accurately due to the imbalanced dataset, yielding a recall of 0.13 for the positive class (heart disease cases).

#### Evaluation Results:
Accuracy: 91%
Precision (for non-heart disease): 92%
Recall (for heart disease): 13%
F1-Score (for heart disease): 0.21

### Conclusion:
The Logistic Regression model effectively predicts heart disease risk for non-affected patients but faces challenges in identifying heart disease cases due to the class imbalance. Future improvements may involve addressing this imbalance through techniques such as oversampling, undersampling, or applying alternative algorithms.

## Notebook 2: Temperature Prediction Model for Greenhouse Climate Control

### Use Case:
This notebook addresses the case of an agricultural company that operates with greenhouse crops, requiring extreme care in temperature control. The goal is to develop an AI model that predicts the temperature on a specific date using various meteorological parameters, allowing the company to adjust the internal temperature of its greenhouses.

### Objective:
The aim is to build a regression model that can predict the temperature based on climatological data from the Jena Climate Dataset. This dataset contains 420,551 timestamps with different meteorological features collected at the Max Planck Institute for Biogeochemistry’s weather station in Jena, Germany.

### Steps:

#### Dataset Selection
The selected dataset is the Jena Climate Dataset: Jena Climate Dataset on Kaggle.
This dataset contains information from 420,551 dates, with a series of meteorological characteristics studied for each date at the Max Planck Institute for Biogeochemistry weather station in Jena, Germany.

#### Preliminary Analysis
The dataset provides a variety of weather-related features such as pressure, temperature, humidity, and wind speed. The goal is to build a model that can predict the temperature based on these meteorological factors.
The problem context is that of a greenhouse agriculture company, which requires precise temperature control to maintain optimal conditions for their crops. Thus, the task is to develop a predictive model using meteorological parameters to forecast the temperature, allowing the company to adjust the internal greenhouse temperature accordingly.

#### Dataset Preparation
There are no categorical variables or missing values in the dataset. The only necessary transformation is converting the "Date Time" column to a more appropriate type for processing.
The dataset is then divided into features and the target variable, where the temperature column serves as the target for prediction.

#### Implementation of Cross-Validation
Cross-validation techniques are used to evaluate the predictive capacity of the model. In this case, TimeSeriesSplit is chosen as the cross-validation method, as it is specifically designed for time series data.
This technique ensures that the test set always follows the training set, preserving the temporal structure of the data and avoiding data leakage by ensuring that future information is not used to predict past outcomes.

#### Model Training
For model selection, Linear Regression is chosen due to the expected linear relationship between the weather variables. For example, colder temperatures may be associated with higher levels of precipitation.
The model is trained in a pipeline that standardizes numerical variables before fitting the regression model.

#### Evaluation
The model is evaluated using mean_absolute_error (MAE), which quantifies the average absolute error between predictions and actual values.
The evaluation reveals that the model consistently achieves a very low MAE across all iterations, indicating that the predictions are highly accurate and the model generalizes well to unseen data.
The use of cross-validation strengthens confidence in the results, as it allows for multiple evaluations of the model's performance, ensuring that the model is robust across different subsets of the dataset. Given the time-series nature of the problem, selecting an inappropriate cross-validation technique could lead to data leakage, where future data could be used to predict past outcomes, but this risk is mitigated by the chosen method.

### Conclusion:
The implementation of TimeSeriesSplit cross-validation provides confidence in the model's results, as it ensures that the evaluations are consistent and that the model does not suffer from data leakage.
Linear Regression is an appropriate choice for this problem due to the inherent linear relationships in climatological data.
The model demonstrated robustness and generalized well to unseen data, as evidenced by the consistently low MAE values across all cross-validation folds.

