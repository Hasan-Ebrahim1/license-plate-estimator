import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from feature_extraction import extract_features  # Import from the correct module
import joblib

def train_and_save_model():
    print("Loading dataset...")
    df = pd.read_csv('license_plate_prices.csv')

    print("Extracting features...")
    features = df['Plate no'].apply(lambda x: pd.Series(extract_features(str(x))))
    features.fillna(0, inplace=True)  # Ensure no missing values

    print("Preparing data for training...")
    X = features
    y = df['Price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("Training the RandomForest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    print("Saving model and scaler...")
    joblib.dump(model, 'trained_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("Model and scaler saved successfully.")


if __name__ == "__main__":
    train_and_save_model()
