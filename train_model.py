import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from license_for_flask import extract_features  # Import your feature extraction function
import joblib

# Load your dataset
df = pd.read_csv('license_plate_prices.csv')

# Apply feature extraction to each plate number
features = df['Plate no'].apply(lambda x: pd.Series(extract_features(str(x))))
features.fillna(0, inplace=True)  # Ensure no missing values

# Split data into features (X) and target (y)
X = features
y = df['Price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train your RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Optionally, evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model and scaler to a file
joblib.dump(model, 'trained_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")
