# Body Fat Percentage Predictor

**Live App:** https://body-fat-predictor-vxt4uvqxblppabud22qmrn.streamlit.app
**Full Technical Analysis:** [View analysis](analysis.md)

A machine learning project that predicts body fat percentage from body measurements.

## Project Overview

This project compares multiple regression and machine learning approaches for predicting body fat percentage using measurements such as age, weight, height, abdomen circumference, chest circumference, and other body measurements.

The project includes:

- Data cleaning and validation
- Exploratory data analysis
- Feature engineering
- Cross-validation
- Hyperparameter tuning
- PCA
- Model comparison
- Residual analysis
- Feature importance analysis
- Interactive Streamlit prediction app

## Dataset

The original dataset contains 252 male observations.

Two questionable observations were excluded during data cleaning, leaving 250 observations for modeling.

Density was excluded as a predictor because body fat percentage is derived from density, which would introduce target leakage.

## Feature Engineering

Two additional predictors were created:

- BMI
- Abdomen-to-height ratio

The abdomen-to-height ratio showed the strongest simple correlation with body fat percentage.

## Models Compared

The following models were evaluated using 5-fold cross-validation:

- Linear Regression
- Ridge Regression
- Lasso Regression
- PCA + Linear Regression
- Random Forest
- Boosting

Boosting achieved the best cross-validated performance.

## Final Model Performance

The selected Boosting model achieved the following performance on the held-out test set:

- MAE: 3.49 percentage points
- RMSE: 4.31 percentage points
- R²: 0.683

## Key Findings

Abdominal measurements contained most of the predictive information.

Permutation importance identified abdomen-to-height ratio and abdomen circumference as the most influential predictors in the final model.

Lasso regression also retained these variables as the strongest predictors, providing consistent evidence across different modeling approaches.

## App

The Streamlit app allows users to enter body measurements and receive an estimated body fat percentage from the trained Gradient Boosting model.

## Limitations

- The dataset is small.
- The observations are male only.
- Predictions should not be interpreted as clinical measurements.
- The model has not been externally validated on an independent population.

## Technologies

- Python
- pandas
- scikit-learn
- matplotlib
- Streamlit