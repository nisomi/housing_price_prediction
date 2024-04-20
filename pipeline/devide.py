import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/preprocessed_data.csv')

train, test = train_test_split(df, train_size=0.7, test_size=0.3, random_state=22)

train.to_csv('../data/train.csv', index=False)
test.to_csv('../data/test.csv', index=False)