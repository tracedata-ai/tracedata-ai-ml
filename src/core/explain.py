import shap
import lime
import lime.lime_tabular
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

from src.core.config import DB_NAME, SMOOTHNESS_MODEL_PATH

class TripExplainer:
    def __init__(self):
        # Load model
        self.model = joblib.load(SMOOTHNESS_MODEL_PATH)
        
        # Load training data for background reference
        conn = sqlite3.connect(DB_NAME)
        self.data = pd.read_sql_query("""
            SELECT accel_fluidity, driving_consistency, comfort_zone_percent 
            FROM trips 
            WHERE accel_fluidity IS NOT NULL
        """, conn)
        conn.close()
        
        # Initialize SHAP Explainer
        # XGBoost models are best explained with TreeExplainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        # Initialize LIME Explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.data.values,
            feature_names=self.data.columns.tolist(),
            class_names=['smoothness_score'],
            mode='regression'
        )

    def explain_trip_shap(self, trip_features):
        """Generates SHAP values for a single trip's features."""
        df = pd.DataFrame([trip_features])
        shap_values = self.shap_explainer.shap_values(df)
        
        # Return a dictionary of feature impacts
        explanation = {}
        for i, col in enumerate(self.data.columns):
            explanation[col] = float(shap_values[0, i])
        
        # Base value (model's average prediction)
        explanation["base_value"] = float(self.shap_explainer.expected_value)
        return explanation

    def explain_trip_lime(self, trip_features):
        """Generates LIME explanation for a single trip."""
        df = pd.DataFrame([trip_features])
        exp = self.lime_explainer.explain_instance(
            data_row=df.values[0],
            predict_fn=self.model.predict
        )
        
        # Get feature impacts as list of (feature, weight)
        return exp.as_list()

    def get_global_importance(self):
        """Returns global feature importance using SHAP."""
        shap_values = self.shap_explainer.shap_values(self.data)
        importances = np.mean(np.abs(shap_values), axis=0)
        
        importance_dict = {}
        for i, col in enumerate(self.data.columns):
            importance_dict[col] = float(importances[i])
        
        return importance_dict

if __name__ == "__main__":
    explainer = TripExplainer()
    
    # Test with a sample trip
    sample_trip = {
        "accel_fluidity": 0.12,
        "driving_consistency": 0.35,
        "comfort_zone_percent": 88.0
    }
    
    print("🌍 Global Feature Importance (SHAP):")
    print(explainer.get_global_importance())
    
    print("\n🔍 Local Trip Explanation (SHAP Values):")
    shap_exp = explainer.explain_trip_shap(sample_trip)
    for feat, val in shap_exp.items():
        print(f"  {feat}: {val:+.2f}")
        
    print("\n🍋 Local Trip Explanation (LIME):")
    lime_exp = explainer.explain_trip_lime(sample_trip)
    for feat_desc, weight in lime_exp:
        print(f"  {feat_desc}: {weight:+.2f}")
