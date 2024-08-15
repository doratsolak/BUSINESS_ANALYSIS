#!/usr/bin/env python
# coding: utf-8

# # Predictive Analysis using Python

# In[1]:


# installing dependencies
import pandas as pd
import seaborn as sns 
import numpy as np


# In[2]:


#read the data
data = pd.read_csv("C:/Users/theod/Downloads/Housing.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


#names of the columns
data.columns


# In[6]:


data.shape


# In[7]:


data.describe()


# In[8]:


#missing values check

missing_values = data.isnull().sum()


# In[9]:


columns_with_missing = missing_values[missing_values > 0].index


# In[10]:


# Fill missing values
for column in columns_with_missing:
    if pd.api.types.is_numeric_dtype(data[column]):
        # Fill numeric columns with the mean value
        mean_value = data[column].mean()
        data[column].fillna(mean_value, inplace=True)
    else:
        # Fill non-numeric columns with the label 'Unknown'
        data[column].fillna('Unknown', inplace=True)
        print(f"The column {column} is not numeric and has been filled with 'Unknown'")


# In[11]:


## Confirm that there are no more missing values
data.isnull().sum()


# In[12]:


#visualization

#Full Bathroom per Sales Prices
sns.relplot(x='price' , y='bathrooms', data=data)


# In[13]:


#Bedrooms per Sales Prices
sns.relplot(x='price' , y='bedrooms', data=data)


# In[14]:


#Above-ground living area in square feet per Sales Prices
sns.relplot(x='price' , y='area', hue='basement', data=data)


# In[15]:


numeric_data = data.select_dtypes(include=['number'])


# In[16]:


data = numeric_data


# In[17]:


data.head()


# In[18]:


#model
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[19]:


# Drop non-numeric columns and target variable
X = data.drop(['price'], axis=1)
y = data['price']


# In[20]:


# Handle categorical data
X = pd.get_dummies(X, drop_first=True)


# In[21]:


# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)


# In[22]:


# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[23]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[25]:


# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVM": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}


# In[26]:


# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        "Mean Squared Error": mse,
        "R-squared": r2
    }
    print(f"{name}:")
    print(f"  Mean Squared Error: {mse}")
    print(f"  R-squared: {r2}\n")


# In[27]:


import matplotlib.pyplot as plt

# Create a DataFrame to compare the models
comparison = pd.DataFrame(results).T

# Plot the comparison
comparison.plot(kind='bar', figsize=(14, 7), subplots=True, layout=(1, 2), sharey=False, title='Model Comparison')
plt.tight_layout()
plt.show()


# In[ ]:


# New Models


# In[31]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[32]:


data = pd.read_csv("C:/Users/theod/Downloads/Housing.csv")


# In[33]:


X = data.drop("price", axis=1)
y = data["price"]


# In[34]:


X_encoded = pd.get_dummies(X, drop_first=True)


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[36]:


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Support Vector Machine": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}


# In[37]:


results = []


# In[38]:


for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": model_name,
        "Mean Squared Error": mse,
        "R-squared": r2
    })


# In[39]:


results_df = pd.DataFrame(results)


# In[40]:


results_df


# In[43]:


import matplotlib.pyplot as plt

# Create a DataFrame to compare the models
comparison_df = results_df.set_index('Model')

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

#Mean Squared Error
comparison_df['Mean Squared Error'].plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Comparison of Mean Squared Error')
axes[0].set_ylabel('Mean Squared Error')

#R-squared
comparison_df['R-squared'].plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Comparison of R-squared')
axes[1].set_ylabel('R-squared')

plt.tight_layout()
plt.show()


# In[ ]:




