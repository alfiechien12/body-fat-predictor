# Body Fat Prediction Model — Full Analysis

## 1. Project Objective

The goal of this project was to build and evaluate machine-learning models for predicting body fat percentage from easily measured anthropometric variables.

The project focused on:

- Cleaning and validating the dataset
- Exploring relationships between body measurements and body fat
- Engineering potentially useful predictors
- Comparing multiple statistical and machine-learning approaches
- Selecting models using cross-validation
- Evaluating the selected model on a held-out test set
- Interpreting model behavior and limitations

## 2. Dataset and Data Quality

The original dataset contained 252 male observations and 15 variables, including body fat percentage, body density, age, weight, height, and multiple circumference measurements.

The dataset contained:

- 252 observations
- 0 missing values
- 0 duplicate rows
- Primarily continuous numeric variables

Two observations were flagged as questionable during the data-quality review:

- One observation reported a height of 29.5 inches, which was implausible for an adult in this dataset.
- One observation reported body fat of 0.0%, which was treated as an implausible target value.

These two observations were excluded from the modeling dataset, leaving 250 observations.

A weight of 363.15 pounds was also identified as an extreme value. However, the subject's other measurements were correspondingly large, so the observation was retained rather than removed solely for being extreme.

Body density was excluded from the predictor set because body fat percentage is derived from density in the source methodology. Including it would introduce target leakage and would make the prediction problem less practically meaningful.

The final modeling dataset therefore contained 250 observations.
## 3. Feature Engineering

Two additional predictors were created:

- **BMI**
- **Abdomen-to-height ratio**

BMI was calculated from weight in pounds and height in inches.

The abdomen-to-height ratio was created to capture abdominal size relative to stature rather than using abdomen circumference alone.

These engineered features were evaluated alongside the original anthropometric variables rather than assumed to be useful automatically.

## 4. Exploratory Data Analysis

Correlation analysis showed that abdomen-related measurements had the strongest linear relationships with body fat percentage.

The strongest correlations with body fat were:

| Feature | Correlation with Body Fat |
|---|---:|
| Abdomen-to-height ratio | 0.828 |
| Abdomen | 0.809 |
| BMI | 0.720 |
| Chest | 0.696 |
| Hip | 0.613 |
| Weight | 0.603 |
| Thigh | 0.544 |
| Knee | 0.494 |
| Neck | 0.490 |
| Biceps | 0.487 |
| Forearm | 0.351 |
| Wrist | 0.344 |
| Age | 0.293 |
| Ankle | 0.254 |
| Height | -0.032 |

The abdomen-to-height ratio had a slightly stronger simple correlation with body fat than raw abdomen circumference, suggesting that adjusting abdominal size for height may contain useful predictive information.

The scatterplot of abdomen circumference versus body fat showed a clear positive relationship that appeared broadly linear, although individual variation remained substantial.

Many of the anthropometric predictors were also correlated with one another. This suggested potential multicollinearity and motivated later comparisons involving Ridge regression, Lasso regression, and PCA.

These correlations were treated as descriptive relationships rather than evidence of causation.
## 5. Modeling and Validation Strategy

The cleaned dataset was divided into:

- **200 training observations (80%)**
- **50 test observations (20%)**

A fixed random seed (`random_state=42`) was used so the split could be reproduced.

Model selection and hyperparameter tuning were performed using **5-fold cross-validation on the training data**. The held-out test set was not used to tune model hyperparameters.

Root Mean Squared Error (RMSE) was used as the primary model-selection metric because it measures prediction error in body-fat percentage points while placing greater weight on larger errors.

Additional evaluation metrics included:

- Mean Absolute Error (MAE)
- R²

Lower RMSE and MAE indicate better predictive accuracy, while higher R² indicates that more variation in body fat is explained by the model.

## 6. Model Comparison

Six modeling approaches were compared:

| Model | Mean 5-Fold CV RMSE |
|---|---:|
| Gradient Boosting | **4.398** |
| Lasso Regression | 4.458 |
| PCA + Linear Regression | 4.521 |
| Ridge Regression | 4.547 |
| Random Forest | 4.555 |
| Linear Regression | 4.586 |

Gradient Boosting achieved the lowest cross-validated RMSE and was therefore selected as the final predictive model.

The improvement over ordinary linear regression was moderate rather than dramatic, suggesting that much of the predictive relationship could already be captured with relatively simple models, while Gradient Boosting extracted some additional nonlinear predictive structure.
## 7. Model-Specific Findings

### Linear Regression

Ordinary linear regression served as the baseline model.

Its mean 5-fold cross-validation RMSE was:

- **4.586**

On the held-out test set, the baseline linear model achieved:

- MAE: 3.87
- RMSE: 4.75
- R²: 0.616

The training R² was approximately 0.766, compared with a test R² of approximately 0.616. This gap suggested some instability or overfitting, although the cross-validation results showed reasonably consistent generalization across folds.

### Ridge Regression

Ridge regression was tested because many of the anthropometric predictors were correlated with one another.

The best Ridge model used:

- **Alpha = 1**

Its mean cross-validation RMSE was:

- **4.547**

This was only a small improvement over ordinary linear regression, suggesting that coefficient shrinkage alone did not substantially improve predictive performance.

### Lasso Regression

Lasso regression produced a more meaningful improvement.

The best Lasso model used:

- **Alpha = 0.1**

Its mean cross-validation RMSE was:

- **4.458**

Lasso also created a sparser model by shrinking several coefficients to zero.

The largest standardized coefficients were associated with:

- Abdomen
- Abdomen-to-height ratio
- Wrist
- Age

Several variables, including weight, height, hip, thigh, knee, neck, and BMI, were reduced to zero after accounting for the other predictors.

This result showed that strong simple correlation does not necessarily mean a variable adds unique predictive information once correlated predictors are considered together.

### Random Forest

The best Random Forest model used:

- Max depth: 4
- Max features: 1.0
- Minimum samples per leaf: 2
- Number of trees: 200

Its mean cross-validation RMSE was:

- **4.555**

Random Forest did not outperform Lasso or Gradient Boosting. The preference for relatively shallow trees suggested that additional tree complexity mostly increased variance rather than improving generalization.

### Gradient Boosting

Gradient Boosting produced the best cross-validated performance.

The selected model used:

- Learning rate: 0.05
- Max depth: 1
- Minimum samples per leaf: 4
- Number of estimators: 500

Its mean cross-validation RMSE was:

- **4.398**

The best model used shallow decision stumps and a relatively small learning rate, indicating that many small sequential corrections worked better than more complex individual trees.

This model was selected for final evaluation on the held-out test set.
## 8. Principal Component Analysis

PCA was evaluated as a dimensionality-reduction approach because many anthropometric measurements were strongly correlated.

The PCA workflow used:

- Standardization
- PCA
- Linear Regression
- 5-fold cross-validation

The best PCA model used:

- **12 principal components out of 15 predictors**

Its mean cross-validation RMSE was:

- **4.521**

The first 12 principal components explained approximately:

- **99.76% of total predictor variance**

The first principal component alone explained approximately:

- **63.4% of predictor variance**

### Interpretation of Principal Components

#### PC1: General Body Size

PC1 had relatively large positive loadings for:

- Weight
- Hip
- BMI
- Abdomen
- Chest
- Thigh
- Neck
- Knee
- Biceps

This suggests that PC1 primarily represented overall body size.

#### PC2: Stature and Body-Proportion Contrast

PC2 was most strongly associated with:

- Height
- Age
- Abdomen-to-height ratio
- BMI
- Abdomen

This component appeared to reflect differences in stature and body proportions rather than overall size alone.

#### PC3: Age and Frame Contrast

PC3 was most strongly associated with:

- Age
- Wrist
- Height
- Thigh

This component appeared to capture a combination of age and body-frame characteristics.

### PCA Interpretation

Although PCA captured nearly all predictor variance with 12 components, it did not outperform Lasso or Gradient Boosting.

This highlighted an important distinction:

**Explaining variance in the predictors is not the same as maximizing predictive accuracy for body fat percentage.**

PCA does not use the target variable when constructing principal components, so directions that explain substantial variation in body measurements are not necessarily the directions most useful for predicting body fat.
## 9. Final Held-Out Test Performance

After model selection was completed using cross-validation, the selected Gradient Boosting model was evaluated on the untouched 50-observation test set.

The final test results were:

- **MAE: 3.49 percentage points**
- **RMSE: 4.31 percentage points**
- **R²: 0.683**

This means the model's predictions differed from the actual body-fat values by about 3.5 percentage points on average.

The test RMSE was slightly lower than the model's mean cross-validation RMSE of 4.398, suggesting that final test performance was consistent with the model-selection results rather than showing a major deterioration on unseen data.

Compared with the baseline linear model, Gradient Boosting reduced both MAE and RMSE and increased R².

## 10. Residual Analysis

Residuals were defined as:

**Actual Body Fat − Predicted Body Fat**

The residual summary on the held-out test set was approximately:

- Mean residual: **-1.26**
- Median residual: **-1.03**
- Standard deviation: **4.17**
- Minimum residual: **-12.27**
- Maximum residual: **6.38**

The negative mean residual indicates that the model slightly overpredicted body fat on average in the held-out test sample.

The residual plot did not show a strong curved pattern or obvious funnel shape, which suggests that no major systematic nonlinear pattern remained unexplained.

However, several larger individual errors remained, showing that the model should not be interpreted as highly precise at the individual level.

## 11. Permutation Importance

Permutation importance was used to evaluate how much model performance deteriorated when individual predictors were randomly shuffled.

The most important predictors were:

| Feature | Mean Increase in RMSE |
|---|---:|
| Abdomen-to-height ratio | 2.701 |
| Abdomen | 1.190 |
| Wrist | 0.337 |
| Neck | 0.216 |
| Age | 0.069 |
| Hip | 0.057 |

Most remaining variables had permutation importance values near zero.

These results reinforced the earlier EDA and Lasso findings that abdomen-related measurements contained most of the useful predictive information.

A key limitation is that abdomen-to-height ratio is mathematically derived from abdomen and height. Because these predictors are correlated, their importance values should not be interpreted as independent causal effects.

Permutation importance therefore supports the conclusion that the final model relied strongly on abdominal measurements, but it does not establish that any individual feature causes changes in body fat.
## 12. Limitations

Several limitations should be considered when interpreting the results.

### Small Sample Size

The final modeling dataset contained only 250 observations. This limits the stability of model estimates and increases the risk that results may vary across different samples.

### Male-Only Dataset

The dataset contains male observations only. The model should therefore not be assumed to generalize to women or to populations with substantially different characteristics.

### No External Validation

The model was evaluated using a held-out subset from the same dataset. It has not been tested on a completely independent external dataset.

### Measurement Error

Anthropometric measurements such as abdomen, chest, thigh, and wrist circumference can vary depending on how and where they are measured.

Prediction accuracy therefore depends partly on measurement consistency.

### Correlated Predictors

Several predictors are highly correlated with one another.

This complicates interpretation because different variables may contain overlapping information.

For example, abdomen circumference and abdomen-to-height ratio are mathematically related, so their individual feature-importance values should not be interpreted as independent effects.

### Not a Clinical Tool

The model is a statistical prediction tool created for educational and portfolio purposes.

It should not be used for diagnosis, medical decision-making, or clinical assessment of body composition.

## 13. Deployment

After model selection was completed, the chosen Gradient Boosting specification was refit using all cleaned observations for deployment.

The trained model was saved and connected to a Streamlit application.

The deployed application:

- Accepts anthropometric measurements from the user
- Recreates the engineered BMI and abdomen-to-height-ratio features
- Sends the full feature set to the trained Gradient Boosting model
- Returns an estimated body-fat percentage

The deployment model uses the same selected hyperparameters as the model evaluated during the validation process.

The held-out test results reported earlier remain the performance estimate for the project. The full-data deployment refit was not used to generate a new performance estimate.

## 14. Overall Conclusion

This project showed that body-fat percentage could be predicted reasonably well from anthropometric measurements using a combination of statistical modeling and machine-learning methods.

Across exploratory analysis, Lasso regression, Gradient Boosting feature importance, and permutation importance, abdomen-related measurements consistently emerged as the strongest predictors.

Gradient Boosting achieved the best cross-validated performance and produced a final held-out test RMSE of approximately 4.31 percentage points, MAE of 3.49 percentage points, and R² of 0.683.

The project also showed that more complex methods were not always better. Random Forest and PCA-based regression did not outperform simpler regularized models, while Gradient Boosting achieved the best results using shallow trees and gradual sequential updates.

Overall, the project demonstrates an end-to-end machine-learning workflow including data validation, feature engineering, model comparison, cross-validation, hyperparameter tuning, dimensionality reduction, interpretation, diagnostics, and deployment.