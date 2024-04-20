#!/usr/bin/env python
# coding: utf-8


import os.path

import pandas as pd
import numpy as np
from scipy import stats

import warnings

warnings.simplefilter('ignore')

print(os.path.exists("data.csv"))

ds = pd.read_csv("../data/data.csv")

# Видалення дублікатів
ds = ds.drop_duplicates()

# Видалення рядків з відсутніми значеннями
ds = ds.dropna(how='all')

# Видалення непотрібних ознак
columns_to_drop = ['country']
ds = ds.drop(columns=columns_to_drop)

# Перевірка розміру датасету після видалення
print("Розмір датасету після видалення:", ds.shape)

import scipy.stats as stats


def remove_outliers(df, column_name, z_threshold=3):
    initial_rows = df.shape[0]

    outliers = df[np.abs(stats.zscore(df[column_name])) >= z_threshold]
    df = df[(np.abs(stats.zscore(df[column_name])) < z_threshold)]

    print("\nПочаткова кількість рядків:", initial_rows)
    print("Кількість видалених рядків:", initial_rows - df.shape[0])

    print("Видалені значення:")
    print(outliers[column_name])

    return df


ds = remove_outliers(ds, 'price', z_threshold=1.5)


categorical_columns = ds.select_dtypes(include=['object']).columns
print(categorical_columns)


from sklearn.preprocessing import LabelEncoder

# Створення екземпляра LabelEncoder
label_encoders = {}

# Кодування категоріальних стовпців
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    ds[column] = label_encoders[column].fit_transform(ds[column])

floatColumns = ds.select_dtypes(include=['float64']).columns
ds[floatColumns] = ds[floatColumns].astype(int)
ds.info()

# Виведення кількості нульових значень
print("Number of zeros in 'price' column:", len(ds[ds['price'] == 0]))

# Зберігання рядків з ціною 0
zero_price_rows_before = ds[ds['price'] == 0]

# Виведення рядків до заповнення
print("Before filling zeros:")
print(zero_price_rows_before.head(1))

# Виведення середнього значення перед заміною
print("\n median price before filling zeros:", ds['price'].median())

# Заміна нульових значень на середнє значення
median_price = ds['price'].median()
ds['price'] = ds['price'].replace(0, median_price)

# Виведення результату після заміни
print("\nAfter filling zeros:")
print(ds[ds.index.isin(zero_price_rows_before.index)].head(1))

# Перевірка середнього значення після заміни
print("\n median price after filling zeros:", ds['price'].median())

# Збереження датасету у файл
ds.to_csv('../data/preprocessed_data.csv', index=False)
