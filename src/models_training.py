#!/usr/bin/env python
# coding: utf-8

import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')


print(os.path.exists("../data/preprocessed_data.csv"))


ds = pd.read_csv("../data/preprocessed_data.csv")

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

X = ds.drop(columns=['price'])
y = ds['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Elastic Net': ElasticNet(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
    'Support Vector Regressor': SVR(),
    'XGBoost Regressor': XGBRegressor()
}

import pandas as pd

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'R^2 Score': r2})

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='R^2 Score', ascending=False)
print(results_df)

importances = models['XGBoost Regressor'].feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(12, 10)) 
plt.title('Feature Importances', fontsize=12)
plt.barh(range(len(indices)), importances[indices], color='#9370DB', edgecolor='black', align='center')  
plt.yticks(range(len(indices)), [X.columns[i] for i in indices], fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel('Relative Importance', fontsize=8)
plt.show()


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Створення моделі
xgb = XGBRegressor()

# Параметри для перебору
param_grid = {
    'n_estimators': [100, 200, 300, 400, 450, 500],
    'learning_rate': [0.05, 0.1, 0.25, 0,35, 0.5],
    'max_depth': [2, 3, 4, 5, 6, 7]
}

# Пошук оптимальних гіперпараметрів
grid_search = GridSearchCV(xgb, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Виведення кращих параметрів та результатів
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Створення моделі з кращими параметрами
best_gbr = XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=500)
best_gbr.fit(X_train, y_train)

# Прогнозування на тестовому наборі
y_pred = best_gbr.predict(X_test)

# Виведення метрик регресії
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print('Root Mean Squared Error:', rmse_val)
print("R^2 Score:", r2)



