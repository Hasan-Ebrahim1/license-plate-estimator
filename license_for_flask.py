import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from feature_extraction import extract_features

def main():
    # Load your dataset
    df = pd.read_csv('license_plate_prices.csv')

    # Apply feature extraction to each plate number
    df_features = df['Plate no'].apply(lambda x: pd.Series(extract_features(str(x))))
    df = pd.concat([df, df_features], axis=1)

    # Ensure no missing values
    df.fillna(0, inplace=True)

    # Split data into features (X) and target (y)
    X = df.drop(columns=['Plate no', 'Price'])
    y = df['Price']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the model using GridSearchCV to find the best parameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2'],
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

    # Save the model and scaler
    joblib.dump(model, 'trained_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

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

if __name__ == "__main__":
    main()
