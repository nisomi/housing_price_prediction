import pandas as pd
import pickle
from xgboost import XGBRegressor
import os

def train_model(file_name):
    if not os.path.isfile('../data/' + file_name):
        print(f"File '{file_name}' not found.")
        return

    data = pd.read_csv('../data/' + file_name)

    X = data.drop(columns=['price'])
    Y = data['price']

    best_gbr = XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=500)
    best_gbr.fit(X, Y)

    with open(f'../model/XGBRegressor.pkl', 'wb') as f:
        pickle.dump(best_gbr, f)


train_model('train.csv')
