#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[41]:


import warnings
warnings.simplefilter('ignore')


# In[42]:


print(os.path.exists("../data/data.csv"))


# In[43]:


ds = pd.read_csv("../data/data.csv")


# In[44]:


print('columns count - ',len(ds.columns), '\n')
print('columns: ',list(ds.columns))


# In[45]:


print('Samples count: ',ds.shape[0])




# In[47]:


print("Any missing sample in training set:",ds.isnull().values.any())


# In[48]:


ds.nunique()


# In[49]:


ds.describe()


# In[50]:


ds.info()


# In[97]:


import matplotlib.pyplot as plt
import seaborn as sns

# Виберемо лише числові дані з DataFrame
numerical_columns = ds.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(15, 12))
correlation_matrix = numerical_columns.corr()
sns.heatmap(
    correlation_matrix,
    vmax=1,
    square=True,
    annot=True,
    fmt='.2f',
    cmap='GnBu',
    cbar_kws={"shrink": .5},
    robust=True)
plt.title('Correlation Matrix of features', fontsize=8)
plt.show()


# In[73]:


plt.figure(figsize=(10, 6))
plt.hist(ds['price'], bins=30, range=(0, 27000000), color='skyblue', edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Price (0 - 27 000 000)')
plt.grid(axis='y', alpha=0.75)
plt.show()

min_price = ds['price'].min()
max_price = ds['price'].max()

print(f"Мінімальне значення ціни: ${min_price:.2f}")
print(f"Максимальне значення ціни: ${max_price:.2f}")


# In[159]:


plt.figure(figsize=(10, 6))
plt.hist(ds['price'], bins=6, range=(0, 1000000), color='skyblue', edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Price (0 - 1 000 000)')
plt.grid(axis='y', alpha=0.75)
plt.show()

min_price = ds['price'].min()
max_price = ds['price'].max()


# In[80]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(ds['bedrooms'], bins=range(1, int(ds['bedrooms'].max()) + 1), color='skyblue', edgecolor='black')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Frequency')
plt.title('Histogram of Number of Bedrooms')
plt.grid(axis='y', alpha=0.75)
plt.show()


# In[88]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(ds['bathrooms'], bins=range(1, int(ds['bathrooms'].max()) + 1), color='skyblue', edgecolor='black')
plt.xlabel('Number of bathrooms')
plt.ylabel('Frequency')
plt.title('Histogram of Number of bathrooms')
plt.grid(axis='y', alpha=0.75)
plt.show()


# In[96]:


slices = [ds[ds['waterfront'] == 1].shape[0], ds[ds['waterfront'] == 0].shape[0]]
labels = ['1', '0']
explode = [0, 0.05]
colors = ['#9195F6', '#F28585']

plt.rcParams["figure.figsize"] = (7, 5)
plt.pie(slices, labels=labels, explode=explode, colors=colors, wedgeprops={'edgecolor': 'black'}, shadow=True, autopct='%1.1f%%')
plt.title('Waterfront Property Distribution')
plt.tight_layout()
plt.show()


# In[134]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))

# Графік для підрахунку кількості будинків за кожним оглядом
plt.subplot(1, 2, 1)
sns.countplot(x='view', data=ds, palette='pastel', edgecolor='black')
plt.title('Distribution of House Views')
plt.xlabel('View')
plt.ylabel('Count')

# Графік для відображення середньої ціни за кожним оглядом
plt.subplot(1, 2, 2)
sns.barplot(x='view', y='price', data=ds, palette='pastel',edgecolor='black')
plt.title('Average Price by House View')
plt.xlabel('View')
plt.ylabel('Average Price')

plt.tight_layout()
plt.show()


# In[167]:


plt.figure(figsize=(12, 6))
plt.hist(ds['yr_built'], bins=6, color='#F28585', edgecolor='black')
plt.xlabel('Year Built')
plt.ylabel('Frequency')
plt.title('Histogram of Year Built')
plt.grid(axis='y', alpha=0.75)
plt.show()


# In[135]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))

# Графік для підрахунку кількості будинків за кожною умовою
plt.subplot(1, 2, 1)
sns.countplot(x='condition', data=ds, palette='pastel', edgecolor='black')
plt.title('Distribution of House Condition')
plt.xlabel('Condition')
plt.ylabel('Count')

# Графік для відображення середньої ціни за кожною умовою
plt.subplot(1, 2, 2)
sns.barplot(x='condition', y='price', data=ds, palette='pastel',edgecolor='black')
plt.title('Average Price by House Condition')
plt.xlabel('Condition')
plt.ylabel('Average Price')

plt.tight_layout()
plt.show()


# In[124]:


# Обчислення середньої ціни для кожного міста
mean_price_by_city = ds.groupby('city')['price'].mean().sort_values()

# Побудова графіка для перших 5 міст
plt.figure(figsize=(12, 6))
mean_price_by_city.head().plot(kind='bar', color='skyblue',  edgecolor='black')
plt.xlabel('City')
plt.ylabel('Average Price')
plt.title('Average Price Comparison for Bottom 5 Cities')
plt.xticks(rotation=45)
plt.show()

# Побудова графіка для останніх 5 міст
plt.figure(figsize=(12, 6))
mean_price_by_city.tail().plot(kind='bar', color='salmon',  edgecolor='black')
plt.xlabel('City')
plt.ylabel('Average Price')
plt.title('Average Price Comparison for Top 5 Cities')
plt.xticks(rotation=45)
plt.show()


# In[130]:


import matplotlib.pyplot as plt

# Підрахунок кількості будинків з ремонтом та без нього
renovated_counts = ds[ds['yr_renovated'] > 0].shape[0]
not_renovated_counts = ds[ds['yr_renovated'] == 0].shape[0]

# Створення колової діаграми
labels = 'Renovated', 'Not Renovated'
explode = [0, 0.05]
sizes = [renovated_counts, not_renovated_counts]
colors = ['#66c2a5', '#fc8d62']

plt.figure(figsize=(7, 5))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=explode,  wedgeprops={'edgecolor': 'black'}, shadow=True)
plt.axis('equal')  # Забезпечення круглої форми діаграми
plt.title('Distribution of Houses by Renovation Status')
plt.show()


# In[154]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='price', data=ds, color='skyblue')
plt.xlabel('Price')
plt.title('Boxplot of Prices')
plt.show()


# In[177]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Обмежимо ціну до 20 млн
ds_filtered = ds[ds['price'] <= 1000000]

# Розділимо роки будівництва на 6 інтервалів
ds_filtered['yr_built_intervals'] = pd.cut(ds_filtered['yr_built'], bins=50)

plt.figure(figsize=(12, 6))
average_price_by_year_interval = ds_filtered.groupby('yr_built_intervals')['price'].mean().reset_index()
sns.barplot(x='yr_built_intervals', y='price', data=average_price_by_year_interval, palette='viridis')
plt.xlabel('Year Built Intervals')
plt.ylabel('Average Price')
plt.title('Average House Price by Year Built Intervals (Price <= $20M)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Видалемо стовпець для подальшого аналізу
ds_filtered.drop(columns=['yr_built_intervals'], inplace=True)


# In[ ]:




