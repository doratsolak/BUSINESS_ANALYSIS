#!/usr/bin/env python
# coding: utf-8

# 2. Exploratory Data Analysis (EDA)

# 2.1 Descriptive Statistics:

# In[6]:


df.describe()


# 2.2 Data Visualization:

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
plt.figure(figsize=(10, 6))
df['price'].hist(bins=30)
plt.title('Histogram of price')
plt.xlabel('price')
plt.ylabel('area')
plt.show()

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='price', data=df)
plt.title('Box plot of price')
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['bedrooms'], df['bathrooms'])
plt.title('Scatter Plot between bedrooms and bathrooms')
plt.xlabel('bedrooms')
plt.ylabel('bathrooms')
plt.show()

# Heatmap for correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:




