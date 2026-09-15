import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("bodyfat.csv")


# ============================================================
# 2. BASIC DATA AUDIT
# ============================================================

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMNS")
print(df.columns.tolist())

print("\nDATA TYPES / MISSING VALUES")
df.info()

print("\nSUMMARY STATISTICS")
print(df.describe())

print("\nMISSING VALUES")
print(df.isna().sum())

print("\nDUPLICATES")
print(df.duplicated().sum())


# ============================================================
# 3. INVESTIGATE SUSPICIOUS OBSERVATIONS
# ============================================================

print("\nHEIGHT BELOW 50 INCHES")
print(df[df["Height"] < 50])

print("\nBODY FAT <= 0")
print(df[df["BodyFat"] <= 0])

print("\nWEIGHT ABOVE 300 LB")
print(df[df["Weight"] > 300])


# ============================================================
# 4. CLEAN DATA
# ============================================================

# Remove the implausible height observation
# and the 0% body-fat observation.

df_clean = df[
    (df["Height"] >= 50)
    & (df["BodyFat"] > 0)
].copy()

print("\nCLEAN DATA SHAPE")
print(df_clean.shape)


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

# Dataset uses:
# Weight = pounds
# Height = inches

df_clean["BMI"] = (
    df_clean["Weight"] * 703
    / (df_clean["Height"] ** 2)
)

df_clean["AbdomenHeightRatio"] = (
    df_clean["Abdomen"]
    / df_clean["Height"]
)


# ============================================================
# 6. EXPLORATORY DATA ANALYSIS
# ============================================================

# Density is excluded because BodyFat is derived from density.
# Including Density as a predictor would create target leakage.

correlations = (
    df_clean
    .drop(columns=["Density"])
    .corr()["BodyFat"]
    .sort_values(ascending=False)
)

print("\nCORRELATION WITH BODY FAT")
print(correlations)

print("\nENGINEERED FEATURE CORRELATIONS")
print(
    df_clean[
        ["BMI", "AbdomenHeightRatio", "BodyFat"]
    ].corr()["BodyFat"]
)


# Abdomen vs BodyFat scatterplot

plt.scatter(
    df_clean["Abdomen"],
    df_clean["BodyFat"]
)

plt.xlabel("Abdomen Circumference")
plt.ylabel("Body Fat %")
plt.title("Abdomen vs Body Fat")
plt.show()


# ============================================================
# 7. DEFINE PREDICTORS AND TARGET
# ============================================================

X = df_clean.drop(
    columns=[
        "BodyFat",
        "Density"
    ]
)

y = df_clean["BodyFat"]


print("\nX SHAPE")
print(X.shape)

print("\nY SHAPE")
print(y.shape)

print("\nPREDICTORS")
print(X.columns.tolist())


# IMPORTANT:
# X should now have 15 predictors.

# Expected:
# (250, 15)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAINING OBSERVATIONS")
print(X_train.shape)

print("\nTEST OBSERVATIONS")
print(X_test.shape)


# ============================================================
# 9. BASELINE LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)


# ============================================================
# 10. BASELINE TRAINING / TEST PERFORMANCE
# ============================================================

linear_train_predictions = linear_model.predict(
    X_train
)

linear_test_predictions = linear_model.predict(
    X_test
)


train_r2 = r2_score(
    y_train,
    linear_train_predictions
)

test_mae = mean_absolute_error(
    y_test,
    linear_test_predictions
)

test_mse = mean_squared_error(
    y_test,
    linear_test_predictions
)

test_rmse = test_mse ** 0.5

test_r2 = r2_score(
    y_test,
    linear_test_predictions
)


print("\nLINEAR REGRESSION BASELINE")

print("Training R2:", train_r2)

print("Test MAE:", test_mae)

print("Test RMSE:", test_rmse)

print("Test R2:", test_r2)


# ============================================================
# 11. 5-FOLD CROSS-VALIDATION:
#     LINEAR REGRESSION
# ============================================================

linear_cv_rmse = -cross_val_score(
    linear_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

print("\nLINEAR REGRESSION - 5 FOLD CV")

print(
    "CV RMSE Scores:",
    linear_cv_rmse
)

print(
    "Mean CV RMSE:",
    linear_cv_rmse.mean()
)

print(
    "CV RMSE Standard Deviation:",
    linear_cv_rmse.std()
)


# ============================================================
# 12. RIDGE REGRESSION
# ============================================================

# StandardScaler and Ridge are placed inside a Pipeline.
#
# This ensures scaling happens independently inside each
# cross-validation fold and prevents data leakage.

ridge_pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "ridge",
        Ridge()
    )
])


ridge_parameters = {
    "ridge__alpha": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}


ridge_search = GridSearchCV(
    ridge_pipeline,
    ridge_parameters,
    cv=5,
    scoring="neg_root_mean_squared_error"
)


ridge_search.fit(
    X_train,
    y_train
)


ridge_cv_rmse = (
    -ridge_search.best_score_
)


print("\nRIDGE REGRESSION")

print(
    "Best Alpha:",
    ridge_search.best_params_[
        "ridge__alpha"
    ]
)

print(
    "Best CV RMSE:",
    ridge_cv_rmse
)


# ============================================================
# 13. LASSO REGRESSION
# ============================================================

lasso_pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "lasso",
        Lasso(
            max_iter=10000
        )
    )
])


lasso_parameters = {
    "lasso__alpha": [
        0.001,
        0.01,
        0.1,
        1,
        10
    ]
}


lasso_search = GridSearchCV(
    lasso_pipeline,
    lasso_parameters,
    cv=5,
    scoring="neg_root_mean_squared_error"
)


lasso_search.fit(
    X_train,
    y_train
)


lasso_cv_rmse = (
    -lasso_search.best_score_
)


print("\nLASSO REGRESSION")

print(
    "Best Alpha:",
    lasso_search.best_params_[
        "lasso__alpha"
    ]
)

print(
    "Best CV RMSE:",
    lasso_cv_rmse
)


# ============================================================
# 14. LASSO COEFFICIENTS
# ============================================================

best_lasso_model = (
    lasso_search.best_estimator_
)

lasso_coefficients = (
    best_lasso_model
    .named_steps["lasso"]
    .coef_
)


coefficient_table = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lasso_coefficients
})


coefficient_table[
    "AbsoluteCoefficient"
] = (
    coefficient_table[
        "Coefficient"
    ].abs()
)


coefficient_table = (
    coefficient_table
    .sort_values(
        "AbsoluteCoefficient",
        ascending=False
    )
)


print("\nLASSO COEFFICIENTS")

print(
    coefficient_table.to_string(
        index=False
    )
)


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

model_comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    ],

    "CV RMSE": [
        linear_cv_rmse.mean(),
        ridge_cv_rmse,
        lasso_cv_rmse
    ]
})


model_comparison = (
    model_comparison
    .sort_values(
        "CV RMSE"
    )
)


print("\nMODEL COMPARISON")

print(
    model_comparison.to_string(
        index=False
    )
)
from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(
    random_state=42
)

rf_parameters = {
    "n_estimators": [200, 500],
    "max_depth": [None, 4, 8],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [0.5, 1.0]
}

rf_search = GridSearchCV(
    random_forest,
    rf_parameters,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

rf_search.fit(
    X_train,
    y_train
)

rf_cv_rmse = -rf_search.best_score_

print("\nRANDOM FOREST")

print(
    "Best Parameters:",
    rf_search.best_params_
)

print(
    "Best CV RMSE:",
    rf_cv_rmse
)
from sklearn.ensemble import GradientBoostingRegressor

gradient_boosting = GradientBoostingRegressor(
    random_state=42
)

gb_parameters = {
    "n_estimators": [100, 200, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [1, 2, 3],
    "min_samples_leaf": [1, 2, 4]
}

gb_search = GridSearchCV(
    gradient_boosting,
    gb_parameters,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

gb_search.fit(
    X_train,
    y_train
)

gb_cv_rmse = -gb_search.best_score_

print("\nGRADIENT BOOSTING")

print(
    "Best Parameters:",
    gb_search.best_params_
)

print(
    "Best CV RMSE:",
    gb_cv_rmse
)
from sklearn.decomposition import PCA

# ============================================================
# PCA + LINEAR REGRESSION
# ============================================================

pca_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("linear", LinearRegression())
])

pca_parameters = {
    "pca__n_components": [
        2,
        3,
        5,
        7,
        10,
        12,
        15
    ]
}

pca_search = GridSearchCV(
    pca_pipeline,
    pca_parameters,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

pca_search.fit(
    X_train,
    y_train
)

pca_cv_rmse = -pca_search.best_score_

best_components = (
    pca_search.best_params_[
        "pca__n_components"
    ]
)

print("\nPCA + LINEAR REGRESSION")

print(
    "Best Number of Components:",
    best_components
)

print(
    "Best CV RMSE:",
    pca_cv_rmse
)


# ============================================================
# EXPLAINED VARIANCE
# ============================================================

best_pca = (
    pca_search
    .best_estimator_
    .named_steps["pca"]
)

explained_variance = (
    best_pca
    .explained_variance_ratio_
)

print(
    "Explained Variance by Component:",
    explained_variance
)

print(
    "Total Explained Variance:",
    explained_variance.sum()
)
pca_loadings = pd.DataFrame(
    best_pca.components_,
    columns=X.columns,
    index=[
        f"PC{i + 1}"
        for i in range(best_components)
    ]
)

print("\nPC1 LOADINGS")
print(
    pca_loadings.loc["PC1"]
    .sort_values(
        key=abs,
        ascending=False
    )
)

print("\nPC2 LOADINGS")
print(
    pca_loadings.loc["PC2"]
    .sort_values(
        key=abs,
        ascending=False
    )
)

print("\nPC3 LOADINGS")
print(
    pca_loadings.loc["PC3"]
    .sort_values(
        key=abs,
        ascending=False
    )
)
best_gb = gb_search.best_estimator_

gb_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_gb.feature_importances_
})

gb_importance = gb_importance.sort_values(
    "Importance",
    ascending=False
)

print("\nGRADIENT BOOSTING FEATURE IMPORTANCE")
print(
    gb_importance.to_string(
        index=False
    )
)
final_model = gb_search.best_estimator_

final_predictions = final_model.predict(X_test)

final_mae = mean_absolute_error(
    y_test,
    final_predictions
)

final_rmse = (
    mean_squared_error(
        y_test,
        final_predictions
    ) ** 0.5
)

final_r2 = r2_score(
    y_test,
    final_predictions
)

print("\nFINAL GRADIENT BOOSTING TEST RESULTS")

print("MAE:", final_mae)
print("RMSE:", final_rmse)
print("R2:", final_r2)
residuals = y_test - final_predictions

print("\nRESIDUAL SUMMARY")
print(pd.Series(residuals).describe())


# Predicted vs residuals
plt.scatter(
    final_predictions,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Body Fat %")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()


# Actual vs predicted
plt.scatter(
    y_test,
    final_predictions
)

minimum = min(
    y_test.min(),
    final_predictions.min()
)

maximum = max(
    y_test.max(),
    final_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Body Fat %")
plt.ylabel("Predicted Body Fat %")
plt.title("Actual vs Predicted Body Fat")
plt.show()


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

from sklearn.inspection import permutation_importance

permutation_results = permutation_importance(
    final_model,
    X_test,
    y_test,
    scoring="neg_root_mean_squared_error",
    n_repeats=30,
    random_state=42,
    n_jobs=-1
)

permutation_table = pd.DataFrame({
    "Feature": X.columns,
    "Mean RMSE Increase":
        permutation_results.importances_mean,

    "Standard Deviation":
        permutation_results.importances_std
})

permutation_table = permutation_table.sort_values(
    "Mean RMSE Increase",
    ascending=False
)

print("\nPERMUTATION IMPORTANCE")
print(
    permutation_table.to_string(
        index=False
    )
)

import joblib

deployment_model = GradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=1,
    min_samples_leaf=4,
    n_estimators=500,
    random_state=42
)

# Train using all cleaned observations
deployment_model.fit(
    X,
    y
)

# Save model + feature names
model_package = {
    "model": deployment_model,
    "features": X.columns.tolist()
}

joblib.dump(
    model_package,
    "bodyfat_model.pkl"
)

print("\nSaved final model as bodyfat_model.pkl")