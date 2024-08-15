#!/usr/bin/env python
# coding: utf-8

# ### 1. Data Collection and Preparation

# 1.1 Importing Data:

# In[1]:


import pandas as pd

# Load the dataset
file_path = 'C:/Users/theod/Downloads/Housing.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())


# 1.2 Data Cleaning:

# In[2]:


# Check for missing values
print(df.isnull().sum())

# Handle missing values, e.g., fill with median or drop rows/columns
df.fillna(df.median(), inplace=True)  # Example for numerical columns

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert categorical variables to numerical ones (example using one-hot encoding)
df = pd.get_dummies(df, drop_first=True)

# Standardize data formats if necessary
# For example, convert date columns to datetime
# df['date_column'] = pd.to_datetime(df['date_column'])


# 1.3 Data Transformation:

# In[3]:


from sklearn.preprocessing import StandardScaler

# Feature scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)


# In[4]:


df_scaled


# In[ ]:




