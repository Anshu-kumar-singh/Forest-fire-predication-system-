"""
Forest Fire Risk Prediction Model Training
============================================
This script trains a Random Forest classifier on historical forest fire data
and saves the model for deployment.

Features used:
- Temperature (°C)
- RH (Relative Humidity %)
- Ws (Wind Speed km/h)
- Rain (mm)
- FFMC (Fine Fuel Moisture Code)
- DMC (Duff Moisture Code)
- DC (Drought Code)
- ISI (Initial Spread Index)
- BUI (Build-Up Index)
- FWI (Fire Weather Index)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import joblib
import os
import json

# Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'forest_fires.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fire_risk_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
METADATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_metadata.json')

# Feature columns for prediction
FEATURE_COLUMNS = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

def load_and_preprocess_data():
    """Load and preprocess the forest fire dataset."""
    print("=" * 60)
    print("STEP 1: Loading and Preprocessing Data")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {len(df)} records from dataset")
    print(f"  Columns: {list(df.columns)}")
    
    # Convert target variable to binary (0 = not fire, 1 = fire)
    df['fire'] = df['Classes'].apply(lambda x: 1 if 'fire' in x.lower() and 'not' not in x.lower() else 0)
    
    # Check class distribution
    fire_count = df['fire'].sum()
    no_fire_count = len(df) - fire_count
    print(f"\n  Class Distribution:")
    print(f"  - Fire events: {fire_count} ({fire_count/len(df)*100:.1f}%)")
    print(f"  - No fire events: {no_fire_count} ({no_fire_count/len(df)*100:.1f}%)")
    
    # Select features
    X = df[FEATURE_COLUMNS].copy()
    y = df['fire'].copy()
    
    # Handle any missing values with mean imputation
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].mean(), inplace=True)
            print(f"  ⚠ Filled missing values in {col}")
    
    print(f"\n✓ Features selected: {list(FEATURE_COLUMNS)}")
    return X, y, df

def train_model(X, y):
    """Train Random Forest classifier and evaluate performance."""
    print("\n" + "=" * 60)
    print("STEP 2: Training Random Forest Model")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Data split: {len(X_train)} training, {len(X_test)} testing")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features standardized using StandardScaler")
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print("✓ Random Forest trained (100 trees, max_depth=10)")
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    return model, scaler, X_test_scaled, y_test, y_pred, y_pred_proba

def evaluate_model(y_test, y_pred, y_pred_proba, model, feature_names):
    """Evaluate model performance and display metrics."""
    print("\n" + "=" * 60)
    print("STEP 3: Model Evaluation")
    print("=" * 60)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n  📊 PERFORMANCE METRICS:")
    print(f"  ┌─────────────────────────────────┐")
    print(f"  │ Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)     │")
    print(f"  │ Precision: {precision:.4f} ({precision*100:.1f}%)     │")
    print(f"  │ Recall:    {recall:.4f} ({recall*100:.1f}%)     │")
    print(f"  │ F1 Score:  {f1:.4f} ({f1*100:.1f}%)     │")
    print(f"  └─────────────────────────────────┘")
    
    print(f"\n  📋 CONFUSION MATRIX:")
    print(f"                 Predicted")
    print(f"                 No Fire  Fire")
    print(f"  Actual No Fire   {conf_matrix[0][0]:3d}     {conf_matrix[0][1]:3d}")
    print(f"  Actual Fire      {conf_matrix[1][0]:3d}     {conf_matrix[1][1]:3d}")
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = dict(zip(feature_names, importance))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  🔥 FEATURE IMPORTANCE (Top Contributing Factors):")
    for i, (feat, imp) in enumerate(sorted_features):
        bar = "█" * int(imp * 50)
        print(f"  {i+1}. {feat:12s} {imp:.4f} {bar}")
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': conf_matrix.tolist(),
        'feature_importance': {k: float(v) for k, v in feature_importance.items()}
    }
    
    return metrics

def save_model(model, scaler, metrics, feature_names):
    """Save trained model and metadata to disk."""
    print("\n" + "=" * 60)
    print("STEP 4: Saving Model and Metadata")
    print("=" * 60)
    
    # Create models directory
    models_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✓ Model saved to: {MODEL_PATH}")
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"✓ Scaler saved to: {SCALER_PATH}")
    
    # Save metadata
    metadata = {
        'feature_columns': feature_names,
        'metrics': metrics,
        'risk_thresholds': {
            'low': {'min': 0, 'max': 33},
            'medium': {'min': 34, 'max': 66},
            'high': {'min': 67, 'max': 100}
        },
        'model_type': 'RandomForestClassifier',
        'model_params': {
            'n_estimators': 100,
            'max_depth': 10
        }
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {METADATA_PATH}")

def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("🔥 FOREST FIRE RISK PREDICTION MODEL TRAINING 🔥")
    print("=" * 60)
    
    # Load data
    X, y, df = load_and_preprocess_data()
    
    # Train model
    model, scaler, X_test, y_test, y_pred, y_pred_proba = train_model(X, y)
    
    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, model, FEATURE_COLUMNS)
    
    # Save
    save_model(model, scaler, metrics, FEATURE_COLUMNS)
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\n  The model is ready for deployment.")
    print("  Run the FastAPI backend to start making predictions.\n")

if __name__ == "__main__":
    main()
