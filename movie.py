import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv('movies.csv')  # Assume a CSV file with movie data

# Data Preprocessing
data = data.dropna()  # Dropping missing values for simplicity

#Feature engineering
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(data[['genre', 'director', 'actors']])

# Combine encoded features with numerical features
features = pd.concat([data[['year']], pd.DataFrame(encoded_features.toarray())], axis=1)
ratings = data['rating']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'R²: {r2_score(y_test, y_pred)}')

# Save the model
joblib.dump(model, 'movie_rating_predictor.pkl')
