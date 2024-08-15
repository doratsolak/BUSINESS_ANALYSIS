### Predictive Analysis of Housing Prices
Overview

This project is focused on predicting housing prices using various regression models in Python. The goal is to explore the performance of different machine learning models, evaluate their accuracy, and identify the best model for predicting housing prices based on the dataset provided.
Methodology

The project follows these key steps:
1. Data Preprocessing

    Loading the Dataset: The housing dataset is loaded using pandas.
    Handling Missing Values: Missing values are identified and filled. Numerical columns are filled with the mean value, and non-numerical columns are filled with the label "Unknown".
    Encoding Categorical Variables: Categorical features are converted to numerical representations using One-Hot Encoding.
    Feature Scaling: Numerical features are scaled using StandardScaler to ensure all features contribute equally to the model's training.

2. Data Visualization

    Relationships between different features (e.g., bathrooms, bedrooms, and area) and the target variable (price) are visualized using scatter plots to understand the distribution and potential correlations.

3. Modeling

    The dataset is split into training and testing sets.
    Six different regression models are trained and evaluated:
        Linear Regression
        Decision Tree Regressor
        Random Forest Regressor
        Support Vector Machine (SVR)
        K-Nearest Neighbors
        Gradient Boosting Regressor
    Each model's performance is measured using:
        Mean Squared Error (MSE): The average squared difference between the actual and predicted values.
        R-squared (R²): The proportion of variance in the dependent variable that is predictable from the independent variables.

4. Model Evaluation

    A comparison of models is visualized using bar plots to assess which model performs the best based on MSE and R².

5. Comparison of Results

    The performance of models before and after feature scaling and encoding is compared to understand the impact of preprocessing on model accuracy.

Results

The results show varying levels of accuracy across the models, with Gradient Boosting and Linear Regression showing the best performance in terms of both MSE and R².
