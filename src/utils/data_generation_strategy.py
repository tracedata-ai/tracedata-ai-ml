"""
Synthetic Data Generation Strategy for Smoothness Scoring ML Pipeline

Since real telematics data is unavailable, this module creates realistic synthetic data
with configurable driver profiles for training and validation.

STRATEGY:
1. Simulate multiple driver profiles (smooth, normal, jerky, unsafe)
2. Generate realistic telematics events for each driver
3. Aggregate into trip features with 18-feature format
4. Create training/validation/test splits
5. All generation uses reproducible seeds
"""

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


class DriverProfile:
    """Defines a driving style with behavioral characteristics."""

    def __init__(
        self,
        name: str,
        smoothness_bias: float,
        aggression_factor: float,
        speed_aggressiveness: float,
        engine_efficiency: float,
    ):
        """
        Initialize driver profile.

        Parameters:
            name: Profile name (e.g., 'smooth', 'jerky')
            smoothness_bias: 0.0 (rough) to 1.0 (smooth) - affects jerk
            aggression_factor: 0.0 to 1.0 - affects harsh events
            speed_aggressiveness: 0.0 to 1.0 - effects speed control
            engine_efficiency: 0.0 (wasteful) to 1.0 (efficient) - affects RPM
        """
        self.name = name
        self.smoothness_bias = smoothness_bias
        self.aggression_factor = aggression_factor
        self.speed_aggressiveness = speed_aggressiveness
        self.engine_efficiency = engine_efficiency

    def get_parameters(self):
        """Get telematics range parameters for this profile."""
        return {
            # Jerk parameters (m/s³)
            "jerk_mean": 0.005 + (1 - self.smoothness_bias) * 0.020,
            "jerk_std": 0.003 + (1 - self.smoothness_bias) * 0.015,
            "jerk_max": 0.015 + (1 - self.smoothness_bias) * 0.045,
            # Acceleration consistency (g)
            "accel_std_range": (
                0.08 + (1 - self.smoothness_bias) * 0.25,
                0.12 + (1 - self.smoothness_bias) * 0.30,
            ),
            # Harsh events probability
            "harsh_brake_prob": 0.02 * self.aggression_factor,
            "harsh_accel_prob": 0.015 * self.aggression_factor,
            "harsh_corner_prob": 0.01 * self.aggression_factor,
            "over_rev_prob": 0.03 * (1 - self.engine_efficiency),
            # Speed control (km/h)
            "speed_std_range": (
                3.0 + self.speed_aggressiveness * 15.0,
                5.0 + self.speed_aggressiveness * 20.0,
            ),
            "max_speed_bias": 60 + self.speed_aggressiveness * 40,  # Target max
            # Engine behavior (RPM)
            "rpm_efficiency": self.engine_efficiency,
            "idle_time_factor": 1.0 - self.engine_efficiency,
        }


# Predefined Driver Profiles
DRIVER_PROFILES = {
    "smooth": DriverProfile(
        name="smooth",
        smoothness_bias=0.9,
        aggression_factor=0.05,
        speed_aggressiveness=0.1,
        engine_efficiency=0.9,
    ),
    "normal": DriverProfile(
        name="normal",
        smoothness_bias=0.6,
        aggression_factor=0.3,
        speed_aggressiveness=0.4,
        engine_efficiency=0.6,
    ),
    "jerky": DriverProfile(
        name="jerky",
        smoothness_bias=0.3,
        aggression_factor=0.6,
        speed_aggressiveness=0.7,
        engine_efficiency=0.4,
    ),
    "unsafe": DriverProfile(
        name="unsafe",
        smoothness_bias=0.1,
        aggression_factor=0.85,
        speed_aggressiveness=0.9,
        engine_efficiency=0.2,
    ),
}


class SyntheticTelemetryGenerator:
    """Generates realistic synthetic telematics events."""

    def __init__(self, profile: DriverProfile, random_seed: int = 42):
        """Initialize with driver profile and random seed."""
        self.profile = profile
        self.rng = np.random.RandomState(random_seed)
        self.params = profile.get_parameters()

    def generate_window(self, window_id: int, duration_seconds: int = 600) -> Dict:
        """
        Generate a 10-minute telematics window for a driver.

        Returns dict matching device telematics format with 18 features.
        """
        # Number of samples (1 per second)
        n_samples = duration_seconds

        # Initialize arrays
        jerk_samples = []
        accel_samples = []
        decel_samples = []
        harsh_brakes = 0
        harsh_accels = 0

        lateral_g_samples = []
        harsh_corners = 0

        speed_samples = []
        rpm_samples = []
        idle_seconds = 0
        over_revs = 0

        # Simulate driving for the window
        current_accel = 0.0
        current_speed = np.random.uniform(40, 80)
        current_rpm = 1800

        for i in range(n_samples):
            # Generate jerk (acceleration of acceleration)
            jerk = self.rng.normal(
                self.params["jerk_mean"],
                self.params["jerk_std"],
            )
            jerk_samples.append(jerk)

            # Update acceleration from jerk
            current_accel += jerk
            current_accel = np.clip(current_accel, -0.8, 0.8)
            accel_samples.append(current_accel)

            # Track max deceleration
            if current_accel < -0.3:
                decel_samples.append(abs(current_accel))

            # Check for harsh brake event
            if (
                current_accel < -0.6
                and self.rng.random() < self.params["harsh_brake_prob"]
            ):
                harsh_brakes += 1

            # Check for harsh accel event
            if (
                current_accel > 0.6
                and self.rng.random() < self.params["harsh_accel_prob"]
            ):
                harsh_accels += 1

            # Update speed from acceleration
            speed_delta = current_accel * (1.0 / 3.6)  # Convert to km/h
            current_speed += speed_delta
            current_speed = np.clip(
                current_speed, 0, self.params["max_speed_bias"] + 20
            )
            speed_samples.append(current_speed)

            # Generate lateral G-forces (turning)
            lateral_g = abs(
                self.rng.normal(0, 0.05 + self.profile.aggression_factor * 0.1)
            )
            lateral_g_samples.append(lateral_g)

            if lateral_g > 0.3 and self.rng.random() < self.params["harsh_corner_prob"]:
                harsh_corners += 1

            # Engine RPM based on speed and efficiency
            base_rpm = 1500 + (current_speed / 100) * 2000
            rpm_var = self.rng.normal(0, base_rpm * (1 - self.params["rpm_efficiency"]))
            current_rpm = np.clip(base_rpm + rpm_var, 600, 6500)
            rpm_samples.append(current_rpm)

            # Over-rev detection
            if current_rpm > 4500 and self.rng.random() < self.params["over_rev_prob"]:
                over_revs += 1

            # Idling detection (speed ~0 and low RPM)
            if current_speed < 5 and current_rpm < 1000:
                idle_seconds += 1

        # Aggregate samples into window statistics
        accel_samples = np.array(accel_samples)
        speed_samples = np.array(speed_samples)
        jerk_samples = np.array(jerk_samples)
        lateral_g_samples = np.array(lateral_g_samples)
        rpm_samples = np.array(rpm_samples)

        return {
            "window_id": window_id,
            "timestamp": datetime.now().isoformat(),
            "sample_count": n_samples,
            "window_seconds": duration_seconds,
            # Speed statistics
            "speed_mean_kmh": float(np.mean(speed_samples)),
            "speed_std": float(np.std(speed_samples)),
            "speed_max_kmh": float(np.max(speed_samples)),
            "speed_variance": float(np.var(speed_samples)),
            # Longitudinal (acceleration/braking)
            "longitudinal_mean_accel_g": float(np.mean(accel_samples)),
            "longitudinal_std_dev": float(np.std(accel_samples)),
            "longitudinal_max_decel_g": float(
                np.max(decel_samples) if decel_samples else 0.2
            ),
            "longitudinal_harsh_brake_count": int(harsh_brakes),
            "longitudinal_harsh_accel_count": int(harsh_accels),
            # Lateral (cornering)
            "lateral_mean_lateral_g": float(np.mean(lateral_g_samples)),
            "lateral_max_lateral_g": float(np.max(lateral_g_samples)),
            "lateral_harsh_corner_count": int(harsh_corners),
            # Jerk (smoothness)
            "jerk_mean": float(np.mean(jerk_samples)),
            "jerk_max": float(np.max(jerk_samples)),
            "jerk_std_dev": float(np.std(jerk_samples)),
            # Engine behavior
            "engine_mean_rpm": float(np.mean(rpm_samples)),
            "engine_max_rpm": float(np.max(rpm_samples)),
            "engine_idle_seconds": int(idle_seconds),
            "engine_idle_events": int(idle_seconds // 30),  # Rough estimate
            "engine_over_rev_count": int(over_revs),
        }

    def generate_trip(
        self, trip_duration_minutes: int = 90, num_windows: int = 12
    ) -> List[Dict]:
        """Generate multiple windows for a single trip."""
        windows = []
        seconds_per_window = trip_duration_minutes * 60 // num_windows

        for w_id in range(num_windows):
            window = self.generate_window(w_id, seconds_per_window)
            windows.append(window)

        return windows


class SyntheticDataPipeline:
    """Orchestrates synthetic data generation for ML training."""

    def __init__(self, config_path: str = "mlops_config.yaml"):
        """Initialize pipeline from configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_config = self.config["data"]
        self.env_config = self.config["environment"]["dev"]

        # Set random seeds for reproducibility
        seed = self.data_config["random_seed"]
        np.random.seed(seed)

        self.rng = np.random.RandomState(seed)

    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete synthetic dataset with aggregated trip features.

        Returns:
            DataFrame with 18 aggregated features per trip
        """
        trips_data = []

        num_drivers = self.env_config["num_drivers"]
        trips_per_driver = self.env_config["trips_per_driver"]

        # Generate for each driver
        for driver_idx in range(num_drivers):
            # Assign profile based on distribution
            profile_name = self._sample_profile()
            profile = DRIVER_PROFILES[profile_name]

            # Create generator with deterministic seed
            seed = 42 + driver_idx
            generator = SyntheticTelemetryGenerator(profile, random_seed=seed)

            # Generate multiple trips for this driver
            for trip_idx in range(trips_per_driver):
                trip_duration = self.rng.randint(
                    self.data_config["generation"]["trip_duration_minutes"]["min"],
                    self.data_config["generation"]["trip_duration_minutes"]["max"],
                )

                # Generate telematics windows
                windows = generator.generate_trip(
                    trip_duration_minutes=trip_duration,
                    num_windows=12,  # Standard: 12 windows for 2-hour trip
                )

                # Aggregate windows into trip features
                trip_features = self._aggregate_windows(windows)
                trip_features["driver_id"] = driver_idx
                trip_features["driver_profile"] = profile_name
                trip_features["trip_duration_minutes"] = trip_duration

                trips_data.append(trip_features)

        df = pd.DataFrame(trips_data)

        print(f"✅ Generated {len(df)} synthetic trips")
        print(f"   Drivers: {num_drivers}")
        print(f"   Trips per driver: {trips_per_driver}")
        print(f"   Profiles: {df['driver_profile'].value_counts().to_dict()}")

        return df

    def _sample_profile(self) -> str:
        """Sample driver profile according to distribution."""
        distribution = self.data_config["generation"]["driver_styles"]
        profiles = list(distribution.keys())
        weights = list(distribution.values())
        return self.rng.choice(profiles, p=weights)

    def _aggregate_windows(self, windows: List[Dict]) -> Dict:
        """Aggregate multiple windows into trip-level features."""
        windows_df = pd.DataFrame(windows)

        # Aggregate using appropriate method
        aggregated = {
            # Longitudinal
            "avg_accel_g": float(windows_df["longitudinal_mean_accel_g"].mean()),
            "avg_accel_std": float(windows_df["longitudinal_std_dev"].mean()),
            "max_decel_g": float(windows_df["longitudinal_max_decel_g"].max()),
            "total_harsh_brakes": int(
                windows_df["longitudinal_harsh_brake_count"].sum()
            ),
            "total_harsh_accels": int(
                windows_df["longitudinal_harsh_accel_count"].sum()
            ),
            # Lateral
            "avg_lateral_g": float(windows_df["lateral_mean_lateral_g"].mean()),
            "max_lateral_g": float(windows_df["lateral_max_lateral_g"].max()),
            "total_harsh_corners": int(windows_df["lateral_harsh_corner_count"].sum()),
            # Speed
            "avg_speed_kmh": float(windows_df["speed_mean_kmh"].mean()),
            "avg_speed_std": float(windows_df["speed_std"].mean()),
            "max_speed_kmh": float(windows_df["speed_max_kmh"].max()),
            # Jerk
            "avg_jerk": float(windows_df["jerk_mean"].mean()),
            "avg_jerk_std": float(windows_df["jerk_std_dev"].mean()),
            "max_jerk": float(windows_df["jerk_max"].max()),
            # Engine
            "avg_rpm": float(windows_df["engine_mean_rpm"].mean()),
            "max_rpm": float(windows_df["engine_max_rpm"].max()),
            "total_idle_seconds": int(windows_df["engine_idle_seconds"].sum()),
            "total_over_revs": int(windows_df["engine_over_rev_count"].sum()),
        }

        return aggregated

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test (60/20/20).

        Returns:
            (train_df, val_df, test_df)
        """
        # Shuffle
        df = df.sample(frac=1, random_state=self.data_config["random_seed"])

        n = len(df)
        train_size = int(0.6 * n)
        val_size = int(0.2 * n)

        train_df = df[:train_size]
        val_df = df[train_size : train_size + val_size]
        test_df = df[train_size + val_size :]

        print("\n📊 Data split:")
        print(f"   Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
        print(f"   Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
        print(f"   Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")

        return train_df, val_df, test_df


if __name__ == "__main__":
    # Test data generation
    pipeline = SyntheticDataPipeline("mlops_config.yaml")
    df = pipeline.generate_dataset()
    train_df, val_df, test_df = pipeline.split_data(df)

    print("\n📈 Sample trip features:")
    print(df.head(3))
    print("\n✅ Data generation pipeline ready!")
