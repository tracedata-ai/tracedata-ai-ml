from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
import pandas as pd
import sqlite3
import numpy as np
import json

from src.core.config import DB_NAME

class FairnessAnalyzer:
    def __init__(self):
        self.conn = sqlite3.connect(DB_NAME)

    def get_fairness_data(self):
        """Joins trips with driver demographics."""
        query = """
            SELECT 
                d.age, 
                d.years_experience, 
                t.smoothness_score
            FROM trips t
            JOIN drivers d ON t.driver_id = d.driver_id
            WHERE t.smoothness_score IS NOT NULL
        """
        df = pd.read_sql_query(query, self.conn)
        
        # Define 'Favorable' outcome (e.g., score >= 80)
        df['favorable'] = (df['smoothness_score'] >= 80).astype(int)
        
        # Define protected attributes (Binary)
        # 1. Young vs Old (Threshold: 35)
        df['is_old'] = (df['age'] >= 35).astype(int)
        
        # 2. Novice vs Expert (Threshold: 10 years)
        df['is_expert'] = (df['years_experience'] >= 10).astype(int)
        
        return df

    def analyze_bias(self, attribute_name):
        """Analyzes bias for a given attribute using AIF360."""
        df = self.get_fairness_data()
        
        # Create AIF360 Dataset
        dataset = BinaryLabelDataset(
            df=df[[attribute_name, 'favorable']],
            label_names=['favorable'],
            protected_attribute_names=[attribute_name],
            favorable_label=1,
            unfavorable_label=0
        )
        
        # Privileged = 1, Unprivileged = 0
        privileged_groups = [{attribute_name: 1}]
        unprivileged_groups = [{attribute_name: 0}]
        
        metric = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        
        results = {
            "disparate_impact": metric.disparate_impact(),
            "statistical_parity_difference": metric.statistical_parity_difference(),
            "privileged_favorable_rate": metric.num_positives(privileged=True) / metric.num_instances(privileged=True),
            "unprivileged_favorable_rate": metric.num_positives(privileged=False) / metric.num_instances(privileged=False)
        }
        
        return results

    def update_all_drivers_fairness(self):
        """Calculates and persists fairness metadata for all drivers."""
        df = self.get_fairness_data_with_ids()
        
        # Calculate group averages
        age_group_avgs = df.groupby('is_old')['smoothness_score'].mean().to_dict()
        exp_group_avgs = df.groupby('is_expert')['smoothness_score'].mean().to_dict()
        
        cursor = self.conn.cursor()
        
        # Group by driver to get their mean performance for the benchmark
        driver_stats = df.groupby('driver_id').agg({
            'smoothness_score': 'mean',
            'is_old': 'first',
            'is_expert': 'first'
        })

        for driver_id, row in driver_stats.iterrows():
            metadata = {
                "age_cohort_avg": round(age_group_avgs.get(row['is_old'], 0), 2),
                "experience_cohort_avg": round(exp_group_avgs.get(row['is_expert'], 0), 2),
                "diff_from_age_cohort": round(row['smoothness_score'] - age_group_avgs.get(row['is_old'], 0), 2),
                "status": "Above Average" if row['smoothness_score'] >= age_group_avgs.get(row['is_old'], 0) else "Below Average"
            }
            
            cursor.execute("""
                UPDATE drivers 
                SET fairness_metadata_json = ? 
                WHERE driver_id = ?
            """, (json.dumps(metadata), int(driver_id)))
            
        self.conn.commit()
        print(f"✅ Persisted fairness metadata for {len(driver_stats)} drivers.")

    def update_all_trips_fairness(self):
        """Persists fairness context for individual trips."""
        df = self.get_fairness_data_with_ids()
        
        # Calculate group averages
        age_group_avgs = df.groupby('is_old')['smoothness_score'].mean().to_dict()
        exp_group_avgs = df.groupby('is_expert')['smoothness_score'].mean().to_dict()
        
        # Get trip IDs
        query = "SELECT trip_id, driver_id FROM trips WHERE smoothness_score IS NOT NULL"
        trips_df = pd.read_sql_query(query, self.conn)
        
        cursor = self.conn.cursor()
        
        for _, trip in trips_df.iterrows():
            # Find driver info for this trip
            driver_info = df[df['driver_id'] == trip['driver_id']].iloc[0]
            trip_score = df[df['driver_id'] == trip['driver_id']]['smoothness_score'].values[0] # This is problematic if multiple trips
            # Fix: get_fairness_data_with_ids should include trip_id
            
    def get_fairness_data_with_ids(self):
        """Includes driver_id and trip_id for precise mapping."""
        query = """
            SELECT 
                t.trip_id,
                d.driver_id,
                d.age, 
                d.years_experience, 
                t.smoothness_score
            FROM trips t
            JOIN drivers d ON t.driver_id = d.driver_id
            WHERE t.smoothness_score IS NOT NULL
        """
        df = pd.read_sql_query(query, self.conn)
        df['is_old'] = (df['age'] >= 35).astype(int)
        df['is_expert'] = (df['years_experience'] >= 10).astype(int)
        return df

    def update_all_persistence(self):
        """Unified persistence update."""
        df = self.get_fairness_data_with_ids()
        age_group_avgs = df.groupby('is_old')['smoothness_score'].mean().to_dict()
        
        cursor = self.conn.cursor()

        # 1. Update Drivers
        driver_stats = df.groupby('driver_id').agg({'smoothness_score': 'mean', 'is_old': 'first'})
        for driver_id, row in driver_stats.iterrows():
            metadata = {"age_cohort_avg": round(age_group_avgs[row['is_old']], 2), "diff": round(row['smoothness_score'] - age_group_avgs[row['is_old']], 2)}
            cursor.execute("UPDATE drivers SET fairness_metadata_json = ? WHERE driver_id = ?", (json.dumps(metadata), int(driver_id)))

        # 2. Update Trips
        for _, row in df.iterrows():
            metadata = {
                "cohort_avg": round(age_group_avgs[row['is_old']], 2),
                "trip_vs_cohort": round(row['smoothness_score'] - age_group_avgs[row['is_old']], 2)
            }
            cursor.execute("UPDATE trips SET fairness_metadata_json = ? WHERE trip_id = ?", (json.dumps(metadata), int(row['trip_id'])))
        
        self.conn.commit()
        print("✅ Persisted bidirectional fairness metadata.")

    def __del__(self):
        self.conn.close()

if __name__ == "__main__":
    analyzer = FairnessAnalyzer()
    
    print("⚖️ Fairness Analysis: Age (Old vs Young)")
    age_bias = analyzer.analyze_bias('is_old')
    for k, v in age_bias.items():
        print(f"  {k}: {v:.4f}")
        
    print("\n⚖️ Fairness Analysis: Experience (Expert vs Novice)")
    exp_bias = analyzer.analyze_bias('is_expert')
    for k, v in exp_bias.items():
        print(f"  {k}: {v:.4f}")

    print("\n💾 Persisting Bidirectional Fairness Metadata (Trips & Drivers)...")
    analyzer.update_all_persistence()

    print("\n💡 Interpretation Guidance:")
    print("  - Disparate Impact: Should be near 1.0 (0.8 to 1.25 is usually acceptable).")
    print("  - Statistical Parity: Should be near 0.0.")
