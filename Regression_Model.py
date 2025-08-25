#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap

# =====================================================
# 1. Load & Inspect Data
# =====================================================
df = pd.read_csv("insurance.csv")
print("Initial Shape:", df.shape)
print(df.head())

# =====================================================
# 2. Data Cleaning
# =====================================================
if df['charges'].dtype == object:
    df['charges'] = df['charges'].astype(str).str.replace('$', '', regex=False).astype(float)

df['sex'] = df['sex'].replace({'F': 'female', 'M': 'male', 'woman': 'female', 'man': 'male'}).str.lower()
df['region'] = df['region'].str.lower()
df['smoker'] = df['smoker'].str.lower().map({'yes': 1, 'no': 0})

df = df.dropna()
df = df[(df['age'] > 0) & (df['bmi'] > 0) & (df['charges'] > 0)]

# =====================================================
# 3. Exploratory Data Analysis (EDA)
# =====================================================
plt.figure(figsize=(10, 6))
sns.histplot(df['charges'], bins=50, kde=True)
plt.title("Distribution of Insurance Charges")
plt.savefig("eda_charges_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title("Charges by Smoker Status")
plt.savefig("eda_charges_by_smoker.png")
plt.close()

sns.pairplot(df[['age','bmi','children','charges']], diag_kind='kde')
plt.savefig("eda_pairplot.png")
plt.close()

# =====================================================
# 4. Feature Engineering
# =====================================================
def preprocess_df(df):
    df = df.copy()
    df['is_male'] = (df['sex'] == 'male').astype(int)
    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    df['bmi_smoker'] = df['bmi'] * df['smoker']
    df['age_smoker'] = df['age'] * df['smoker']
    df['age_bmi'] = df['age'] * df['bmi']
    df = df.drop(columns=['sex'])
    return df

df_processed = preprocess_df(df)
X = df_processed.drop(columns=['charges'])
y = df_processed['charges']

# =====================================================
# 5. Train-Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================
# 6. Baseline Random Forest
# =====================================================
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("Random Forest R²:", r2_score(y_test, y_pred))

# =====================================================
# 7. Benchmark Models
# =====================================================
models = {
    "Random Forest": rf_model,
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(random_state=42, n_jobs=-1),
    "CatBoost": cb.CatBoostRegressor(verbose=0, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = r2_score(y_test, preds)

print("\nModel Benchmark Results:")
for name, score in results.items():
    print(f"{name}: R² = {score:.4f}")

# =====================================================
# 8. Hyperparameter Tuning
# =====================================================
param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Best RF Params:", grid_search.best_params_)

tuned_rf = grid_search.best_estimator_
tuned_preds = tuned_rf.predict(X_test)
print("Tuned RF R²:", r2_score(y_test, tuned_preds))

# =====================================================
# 9. Learning Curve
# =====================================================
train_sizes, train_scores, test_scores = learning_curve(
    tuned_rf, X, y, cv=5, scoring='r2', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='CV')
plt.xlabel("Training examples")
plt.ylabel("R² score")
plt.legend()
plt.title("Learning Curve")
plt.savefig("learning_curve.png")
plt.close()

# =====================================================
# 10. Feature Importance & Explainability
# =====================================================
explainer = shap.TreeExplainer(tuned_rf)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("shap_summary_plot.png")
plt.close()

# =====================================================
# 11. Save Model
# =====================================================
joblib.dump(tuned_rf, "insurance_model.pkl")

# =====================================================
# 12. Validation Predictions
# =====================================================
val_df = pd.read_csv("validation_dataset.csv")
val_df['sex'] = val_df['sex'].replace({'F': 'female', 'M': 'male', 'woman': 'female', 'man': 'male'}).str.lower()
val_df['region'] = val_df['region'].str.lower()
val_df['smoker'] = val_df['smoker'].str.lower().map({'yes': 1, 'no': 0})
val_df = val_df.dropna()

val_processed = preprocess_df(val_df)
for col in X.columns:
    if col not in val_processed.columns:
        val_processed[col] = 0
val_processed = val_processed[X.columns]

val_df['predicted_charges'] = tuned_rf.predict(val_processed)
val_df.loc[val_df['predicted_charges'] < 1000, 'predicted_charges'] = 1000

print(val_df[['age','sex','bmi','children','smoker','region','predicted_charges']].head())



