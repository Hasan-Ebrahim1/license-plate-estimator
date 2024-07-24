import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from itertools import groupby

# Load your dataset
df = pd.read_csv('license_plate_prices.csv')

# Debug: Print the first few rows of the dataset
print("Dataset head:\n", df.head())

# Feature extraction function
def is_palindrome(s):
    return s == s[::-1]

def is_serial(s):
    return all(int(s[i]) == int(s[i-1]) + 1 for i in range(1, len(s))) or all(int(s[i]) == int(s[i-1]) - 1 for i in range(1, len(s)))

def is_double_sequential(s):
    return all(s[i] == s[i + 1] and (int(s[i]) == int(s[i + 2]) + 1 or int(s[i]) == int(s[i + 2]) - 1) for i in range(0, len(s) - 2, 2))

def is_duplicated_and_serial(s):
    return (len(s) % 2 == 0 and 
            all(s[i] == s[i + len(s)//2] for i in range(len(s)//2)) and 
            (is_serial(s[:len(s)//2]) or is_serial(s[len(s)//2:])))

def count_repeats(s):
    return len(s) - len(set(s))

def longest_consecutive_repeats(s):
    return max([len(list(g)) for k, g in groupby(s)])

def lower_number_value(s):
    return sum(10 - int(digit) for digit in s)

def zero_between_features(s):
    return int('0' in s)

def serial_length_value(s):
    max_length = 0
    current_length = 1
    for i in range(1, len(s)):
        if int(s[i]) == int(s[i-1]) + 1 or int(s[i]) == int(s[i-1]) - 1:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 1
    return max_length

def extract_features(plate_number):
    features = {}
    length = len(plate_number)
    # Adjust weights based on the length of the plate number
    if length == 4:
        weight_factor = 6
    elif length == 5:
        weight_factor = 3
    else:
        weight_factor = 1
    
    features['num_digits'] = length
    features['num_zeros'] = plate_number.count('0') * weight_factor
    features['num_repeats'] = count_repeats(plate_number) * weight_factor
    features['longest_consecutive_repeats'] = longest_consecutive_repeats(plate_number) * weight_factor
    features['palindrome'] = is_palindrome(plate_number) * weight_factor
    if length == 6 and is_palindrome(plate_number):
        features['palindrome'] *= (1/3)  # Decrease the influence by 1/3 for 6-digit palindromes
    features['serial'] = is_serial(plate_number) * weight_factor
    features['double_sequential'] = is_double_sequential(plate_number) * weight_factor
    features['duplicated_and_serial'] = is_duplicated_and_serial(plate_number) * weight_factor
    # Interaction features
    features['repeats_and_serial'] = count_repeats(plate_number) * is_serial(plate_number) * weight_factor
    features['zeros_and_repeats'] = plate_number.count('0') * count_repeats(plate_number) * weight_factor
    features['double_and_repeats'] = is_double_sequential(plate_number) * count_repeats(plate_number) * weight_factor
    features['zeros_and_serial'] = plate_number.count('0') * is_serial(plate_number) * weight_factor
    features['palindrome_and_repeats'] = is_palindrome(plate_number) * count_repeats(plate_number) * weight_factor
    
    # Additional features
    features['lower_number_value'] = lower_number_value(plate_number) * weight_factor
    features['zero_between_features'] = zero_between_features(plate_number) * weight_factor
    features['serial_length_value'] = serial_length_value(plate_number) * weight_factor
    
    # Additional interaction features for better accuracy
    features['num_repeats_and_length'] = count_repeats(plate_number) * length * weight_factor
    features['serial_and_length'] = is_serial(plate_number) * length * weight_factor
    return features

# Apply feature extraction to each plate number
df_features = df['Plate no'].apply(lambda x: pd.Series(extract_features(str(x))))
df = pd.concat([df, df_features], axis=1)

# Debug: Print the first few rows of the dataset with features
print("Dataset with features:\n", df.head())

# Check feature correlation with price
print("Feature correlation with price:\n", df.corr()['Price'])

# Ensure no missing values
df.fillna(0, inplace=True)

# Split data into features (X) and target (y)
X = df[['num_digits', 'num_zeros', 'num_repeats', 'longest_consecutive_repeats', 'palindrome', 'serial', 'double_sequential', 'duplicated_and_serial', 'repeats_and_serial', 'zeros_and_repeats', 'double_and_repeats', 'zeros_and_serial', 'palindrome_and_repeats', 'lower_number_value', 'zero_between_features', 'serial_length_value', 'num_repeats_and_length', 'serial_and_length']]
y = df['Price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Verify the split
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train the model using GridSearchCV to find the best parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Train the model using the best parameters
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Check feature importances
feature_importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse}")
print(f"R^2 Score on test set: {r2}")

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {np.mean(cv_scores)}")

# Function to predict the price of a new plate number
def predict_price(plate_number):
    features = pd.Series(extract_features(plate_number)).values.reshape(1, -1)
    features_scaled = scaler.transform(features)
    predicted_price = model.predict(features_scaled)[0]
    return predicted_price

# Ask the user for a plate number and predict the price
user_plate_number = input("Enter the plate number: ")
predicted_price = predict_price(user_plate_number)
print(f'The estimated price for the plate number {user_plate_number} is {predicted_price:.2f}')

# Debug: Compare predicted and actual prices for all records
df['Predicted Price'] = model.predict(scaler.transform(X))
