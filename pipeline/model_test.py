import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def test_model(file_name):
    if not os.path.isfile('../data/' + file_name):
        print(f"File '{file_name}' not found.")
        return

    data = pd.read_csv('../data/' + file_name)

    X = data.drop(columns=['price'])
    Y = data['price']

    with open(f'../model/XGBRegressor.pkl', 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X)

    pd.DataFrame(predictions).to_csv('../data/predictions.csv', index=False)

    # Виведення метрик регресії
    mse = mean_squared_error(Y, predictions)
    mae = mean_absolute_error(Y, predictions)
    rmse_val = np.sqrt(mean_squared_error(Y, predictions))
    r2 = r2_score(Y, predictions)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print('Root Mean Squared Error:', rmse_val)
    print("R^2 Score:", r2)


test_model('test.csv')
