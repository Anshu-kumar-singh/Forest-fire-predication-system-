"""
ML Model Loading and Prediction
================================
Loads trained Random Forest model and makes predictions for grid cells.
Provides risk scores, categories, and feature importance explanations.
"""

import os
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import joblib

# Paths to model files
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'fire_risk_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')

# Feature columns in order (must match training)
FEATURE_COLUMNS = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

# Weather data key mapping to feature names
WEATHER_TO_FEATURE = {
    'temperature': 'Temperature',
    'humidity': 'RH',
    'wind_speed': 'Ws',
    'rainfall': 'Rain',
    'ffmc': 'FFMC',
    'dmc': 'DMC',
    'dc': 'DC',
    'isi': 'ISI',
    'bui': 'BUI',
    'fwi': 'FWI'
}

# Risk thresholds
RISK_THRESHOLDS = {
    'low': (0, 33),
    'medium': (34, 66),
    'high': (67, 100)
}

class FireRiskModel:
    """Wrapper for the trained fire risk prediction model."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.is_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load model files from disk."""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                
                if os.path.exists(METADATA_PATH):
                    with open(METADATA_PATH, 'r') as f:
                        self.metadata = json.load(f)
                
                self.is_loaded = True
                print("✓ Fire risk model loaded successfully")
            else:
                print("⚠ Model files not found. Using fallback prediction.")
                self.is_loaded = False
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            self.is_loaded = False
    
    def _weather_to_features(self, weather_data: Dict[str, Any]) -> np.ndarray:
        """Convert weather data to feature array for prediction."""
        features = []
        for feature_name in FEATURE_COLUMNS:
            # Find corresponding key in weather data
            weather_key = None
            for wk, fn in WEATHER_TO_FEATURE.items():
                if fn == feature_name:
                    weather_key = wk
                    break
            
            if weather_key and weather_key in weather_data:
                features.append(weather_data[weather_key])
            else:
                # Use default values if missing
                defaults = {
                    'Temperature': 25, 'RH': 50, 'Ws': 10, 'Rain': 0,
                    'FFMC': 70, 'DMC': 10, 'DC': 100, 'ISI': 5, 'BUI': 15, 'FWI': 10
                }
                features.append(defaults.get(feature_name, 0))
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_fallback_risk(self, weather_data: Dict[str, Any]) -> float:
        """Calculate risk score using simple rules when model isn't available."""
        temp = weather_data.get('temperature', 25)
        humidity = weather_data.get('humidity', 50)
        wind = weather_data.get('wind_speed', 10)
        rain = weather_data.get('rainfall', 0)
        fwi = weather_data.get('fwi', 10)
        
        # Simple weighted formula
        risk = (
            (temp - 20) * 2 +           # Higher temp = higher risk
            (80 - humidity) * 0.5 +      # Lower humidity = higher risk
            wind * 1.5 +                 # Higher wind = higher risk
            fwi * 3 -                    # FWI is a strong indicator
            rain * 10                    # Rain reduces risk
        )
        
        # Normalize to 0-100
        risk = max(0, min(100, risk))
        return risk
    
    def predict(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fire risk for given weather conditions.
        
        Returns:
            - risk_score: 0-100 continuous score
            - risk_category: Low, Medium, or High
            - probability: Model confidence (if available)
            - feature_importance: Per-feature contribution
        """
        if self.is_loaded and self.model is not None:
            # Use trained model
            features = self._weather_to_features(weather_data)
            features_scaled = self.scaler.transform(features)
            
            # Get probability of fire
            proba = self.model.predict_proba(features_scaled)[0]
            fire_probability = proba[1] if len(proba) > 1 else proba[0]
            
            # Convert to 0-100 risk score
            risk_score = fire_probability * 100
            
            # Get feature importances
            importance = self.model.feature_importances_
            feature_importance = dict(zip(FEATURE_COLUMNS, importance))
        else:
            # Use fallback calculation
            risk_score = self._calculate_fallback_risk(weather_data)
            fire_probability = risk_score / 100
            
            # Approximate feature importance
            feature_importance = {
                'Temperature': 0.18, 'RH': 0.15, 'Ws': 0.12, 'Rain': 0.10,
                'FFMC': 0.15, 'DMC': 0.08, 'DC': 0.07, 'ISI': 0.06, 'BUI': 0.05, 'FWI': 0.04
            }
        
        # Determine risk category
        risk_score = round(risk_score, 1)
        if risk_score <= RISK_THRESHOLDS['low'][1]:
            category = 'Low'
        elif risk_score <= RISK_THRESHOLDS['medium'][1]:
            category = 'Medium'
        else:
            category = 'High'
        
        return {
            'risk_score': risk_score,
            'risk_category': category,
            'probability': round(fire_probability, 4),
            'feature_importance': {k: round(v, 4) for k, v in feature_importance.items()},
            'model_type': 'Random Forest' if self.is_loaded else 'Rule-based Fallback'
        }
    
    def explain_prediction(self, weather_data: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate human-readable explanation for a prediction.
        
        Returns contributing factors and their impact on the risk.
        """
        factors = []
        importance = prediction['feature_importance']
        
        # Temperature
        temp = weather_data.get('temperature', 25)
        if temp > 35:
            factors.append({
                'factor': 'High Temperature',
                'value': f'{temp}°C',
                'impact': 'critical',
                'description': 'Extreme heat significantly increases fire ignition and spread risk.'
            })
        elif temp > 30:
            factors.append({
                'factor': 'Elevated Temperature',
                'value': f'{temp}°C',
                'impact': 'high',
                'description': 'Above-average temperatures increase vegetation dryness.'
            })
        
        # Humidity
        humidity = weather_data.get('humidity', 50)
        if humidity < 30:
            factors.append({
                'factor': 'Very Low Humidity',
                'value': f'{humidity}%',
                'impact': 'critical',
                'description': 'Low humidity allows fuels to dry rapidly, increasing flammability.'
            })
        elif humidity < 50:
            factors.append({
                'factor': 'Low Humidity',
                'value': f'{humidity}%',
                'impact': 'moderate',
                'description': 'Below-normal humidity contributes to drier conditions.'
            })
        
        # Wind Speed
        wind = weather_data.get('wind_speed', 10)
        if wind > 20:
            factors.append({
                'factor': 'High Wind Speed',
                'value': f'{wind} km/h',
                'impact': 'critical',
                'description': 'Strong winds can rapidly spread fire and make control difficult.'
            })
        elif wind > 15:
            factors.append({
                'factor': 'Moderate Wind',
                'value': f'{wind} km/h',
                'impact': 'moderate',
                'description': 'Wind assists fire spread and reduces humidity.'
            })
        
        # Rainfall
        rain = weather_data.get('rainfall', 0)
        if rain > 5:
            factors.append({
                'factor': 'Recent Rainfall',
                'value': f'{rain} mm',
                'impact': 'protective',
                'description': 'Recent precipitation reduces fire risk by moistening fuels.'
            })
        elif rain < 1:
            factors.append({
                'factor': 'No Recent Rain',
                'value': f'{rain} mm',
                'impact': 'moderate',
                'description': 'Dry conditions persist without recent precipitation.'
            })
        
        # Fire Weather Index
        fwi = weather_data.get('fwi', 10)
        if fwi > 15:
            factors.append({
                'factor': 'High Fire Weather Index',
                'value': f'{fwi}',
                'impact': 'critical',
                'description': 'FWI indicates severe fire weather conditions.'
            })
        
        return {
            'summary': self._generate_summary(prediction['risk_category'], factors),
            'contributing_factors': factors,
            'top_features': sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _generate_summary(self, category: str, factors: List[Dict]) -> str:
        """Generate a summary explanation based on risk category."""
        critical_count = sum(1 for f in factors if f['impact'] == 'critical')
        
        if category == 'High':
            return f"⚠️ HIGH RISK: {critical_count} critical factors detected. Immediate preventive action recommended."
        elif category == 'Medium':
            return "⚡ MEDIUM RISK: Moderate fire conditions present. Enhanced monitoring advised."
        else:
            return "✅ LOW RISK: Current conditions are favorable. Standard monitoring sufficient."

# Singleton instance
_model_instance = None

def get_model() -> FireRiskModel:
    """Get or create the singleton model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = FireRiskModel()
    return _model_instance
