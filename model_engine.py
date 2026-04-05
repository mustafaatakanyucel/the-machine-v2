import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import time
import importlib.util
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_feature_columns():
    schema_path = BASE_DIR / 'feature_schema.py'
    spec = importlib.util.spec_from_file_location('feature_schema', schema_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.FEATURE_COLUMNS


FEATURE_COLUMNS = _load_feature_columns()

class ModelEngine:
    """
    An enterprise-level model engine for The Machine v2.0.
    This version uses Scikit-learn's MLPClassifier to avoid hardware-specific
    (AVX) compatibility issues with TensorFlow.
    """
    def __init__(self, data_path='processed_crime_data.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        print("ModelEngine (Scikit-learn MLP) for The Machine v2.0 initialized.")

    def _load_and_prep_data(self):
        """Loads and prepares the data for model training."""
        print("Loading processed data...")
        df = pd.read_csv(self.data_path)

        # --- Define Target Variable (y) ---
        high_risk_crimes = ['BATTERY', 'ROBBERY', 'ASSAULT', 'HOMICIDE']
        df['HighRisk'] = df['primary_type'].isin(high_risk_crimes).astype(int)
        
        # Convert 'Date' to datetime and extract temporal features
        df['Date'] = pd.to_datetime(df['Date'])
        
        if 'datetime' in df.columns:
             df['Hour'] = pd.to_datetime(df['datetime']).dt.hour
        elif 'date' in df.columns and hasattr(pd.to_datetime(df['date']), 'dt'):
             try:
                df['Hour'] = pd.to_datetime(df['date']).dt.hour
             except AttributeError:
                df['Hour'] = 0
        else:
             df['Hour'] = 0

        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        
        # Define features (X) and target (y)
        features = FEATURE_COLUMNS
        target = 'HighRisk'

        for col in features:
            if col not in df.columns:
                raise ValueError(f"Missing required feature column in dataset: {col}")

        X = df[features]
        y = df[target]
        
        print("Features and target variable defined.")
        return X, y

    def run(self):
        """
        Executes the full model training pipeline.
        """
        start_time = time.time()

        # 1. Load data
        X, y = self._load_and_prep_data()
        
        # 2. Chronological train-test split
        split_index = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        print(f"Data split chronologically: {len(X_train)} training samples, {len(X_test)} testing samples.")

        # 3. Scale features
        print("Scaling features with StandardScaler...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 4. Initialize and train MLPClassifier
        print("\n--- Starting Model Training with MLPClassifier ---")
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            verbose=True,
            early_stopping=True
        )
        self.model.fit(X_train_scaled, y_train)
        print("--- Model Training Complete ---")

        # 5. Evaluate model
        print("\n--- Evaluating Model Performance ---")
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)
        
        print(f"Test ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # 6. Save model and scaler
        print("\nSaving model and scaler...")
        joblib.dump(self.model, 'the_machine_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        print("Model saved to 'the_machine_model.pkl'")
        print("Scaler saved to 'scaler.pkl'")
        
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    engine = ModelEngine(data_path='processed_crime_data.csv')
    engine.run()