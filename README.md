# Customer Churn Prediction

A machine learning project for predicting customer churn using customer demographics, subscription details, engagement behavior, payment activity, and support interactions.

## Project Overview

This project analyzes a customer churn dataset with 3,000 records and 24 features. The workflow covers data understanding, missing-value analysis, outlier detection, feature engineering, statistical analysis, model comparison, and model tuning.

## Objectives

- Explore customer behavior and churn patterns.
- Identify missing values and evaluate whether missingness is related to churn.
- Engineer useful features related to renewal timing, engagement, stability, and support pressure.
- Compare multiple classification models.
- Tune Logistic Regression using threshold and regularization analysis.

## Dataset

The dataset includes customer-level variables such as:

- Age and region
- Subscription type and contract type
- Device type and payment method
- Tenure, monthly spend, sessions, and engagement score
- Support tickets and payment delays
- Target variable: `churn`

Dataset shape: `3000 rows x 24 columns`

## Workflow

1. Data loading and basic inspection
2. Missing-value analysis
3. Missingness vs churn analysis using chi-square testing
4. Distribution and outlier analysis
5. Correlation and covariance analysis
6. Categorical feature association using chi-square testing
7. Feature engineering
8. Model comparison
9. Logistic Regression threshold tuning
10. Regularization sensitivity analysis

## Models Used

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting

## Key Results

Initial model comparison showed close performance across several models. HistGradientBoosting had the highest initial accuracy at approximately `67.17%`, while Logistic Regression performed competitively.

After threshold and regularization tuning, Logistic Regression achieved a validation F1-score of approximately `0.59` using an optimized threshold and C value.

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook

  ## Project Files

- `notebooks/customer_churn_prediction.ipynb` — Main notebook containing data preprocessing, analysis, modeling, and evaluation.
- `data/churn_dataset.csv` — Dataset used for training and testing.
- `requirements.txt` — Python libraries required to run the project.

##Repository Purpose

This repository is intended to demonstrate practical machine learning skills, including exploratory data analysis, feature engineering, model comparison, classification evaluation, and model tuning.

##What I Learned

Through this project, I practiced building a full machine learning workflow from data exploration to model evaluation. I also learned how to compare multiple classifiers, analyze missing values, engineer useful features, and tune Logistic Regression to improve classification performance.
