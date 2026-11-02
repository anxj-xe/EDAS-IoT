"""Configuration loader for EDAS-IoT."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from src.common.schemas import DeviceProfile


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # Assuming config.py is in src/common/, project root is 2 levels up
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Safely load a YAML file."""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Config:
    """Application configuration manager."""

    _instance = None

    def __init__(self, settings_path: Optional[str] = None, profiles_path: Optional[str] = None):
        root = get_project_root()
        self.root_dir = root

        self.settings_file = Path(settings_path) if settings_path else root / "config" / "settings.yaml"
        self.profiles_file = Path(profiles_path) if profiles_path else root / "config" / "device_profiles.yaml"

        self.settings = load_yaml(self.settings_file)
        self.raw_profiles = load_yaml(self.profiles_file)

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure necessary data and model directories exist."""
        storage = self.settings.get("storage", {})
        for key in ["raw_data_dir", "processed_data_dir", "samples_dir", "models_dir"]:
            rel_path = storage.get(key)
            if rel_path:
                abs_path = self.root_dir / rel_path
                abs_path.mkdir(parents=True, exist_ok=True)

        db_path = storage.get("database_path")
        if db_path:
            abs_db_dir = (self.root_dir / db_path).parent
            abs_db_dir.mkdir(parents=True, exist_ok=True)

    def get_physical_limits(self) -> Dict[str, Dict[str, float]]:
        """Get physical sensor plausibility limits."""
        return self.settings.get("physical_limits", {
            "temperature_c": {"min": -40.0, "max": 85.0},
            "humidity_pct": {"min": 0.0, "max": 100.0},
            "pressure_hpa": {"min": 300.0, "max": 1100.0}
        })

    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection parameters."""
        return self.settings.get("detection", {
            "statistical": {
                "z_score_threshold": 3.0,
                "rolling_window_size": 10,
                "rolling_z_threshold": 2.5,
                "min_history_for_rolling": 5,
                "stuck_sensor_window": 5,
                "epsilon": 1.0e-6
            },
            "isolation_forest": {
                "n_estimators": 100,
                "contamination": 0.05,
                "random_state": 42,
                "features": ["temperature_c", "humidity_pct", "pressure_hpa"]
            }
        })

    def get_device_profiles(self) -> Dict[str, DeviceProfile]:
        """Load and return all parsed device profiles."""
        profiles = {}
        for key, data in self.raw_profiles.items():
            if isinstance(data, dict):
                profiles[key] = DeviceProfile(
                    name=data.get("name", key),
                    description=data.get("description", ""),
                    min_temperature_c=data.get("min_temperature_c"),
                    max_temperature_c=data.get("max_temperature_c"),
                    min_humidity_pct=data.get("min_humidity_pct"),
                    max_humidity_pct=data.get("max_humidity_pct"),
                    min_pressure_hpa=data.get("min_pressure_hpa"),
                    max_pressure_hpa=data.get("max_pressure_hpa"),
                    required_parameters=data.get("required_parameters", ["temperature_c", "humidity_pct"]),
                    safety_margins=data.get("safety_margins", {})
                )
        return profiles

    def get_device_profile(self, profile_key: str) -> Optional[DeviceProfile]:
        """Get a specific device profile by its key."""
        profiles = self.get_device_profiles()
        return profiles.get(profile_key)


_default_config = None

def get_config() -> Config:
    """Global configuration singleton."""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config
