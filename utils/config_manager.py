import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from config import DEFAULT_CONFIG


class ConfigManager:
    def __init__(self, configs_root: str = "configs"):
        self.configs_root = Path(configs_root)
        self.configs_root.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.configs_root / "training").mkdir(exist_ok=True)
        (self.configs_root / "inference").mkdir(exist_ok=True)
        (self.configs_root / "preprocessing").mkdir(exist_ok=True)

    def load_config(self, config_path: str, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load and merge YAML config with base config
        
        Args:
            config_path: Path to YAML config file
            base_config: Base configuration to merge with (defaults to DEFAULT_CONFIG)
            
        Returns:
            Merged configuration dictionary
        """
        if base_config is None:
            base_config = DEFAULT_CONFIG.copy()
            
        config_file = Path(config_path)
        
        if not config_file.exists():
            print(f"[WARNING] Config file {config_path} not found. Using defaults.")
            return base_config
            
        try:
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
                
            # Deep merge the configs
            merged_config = self._deep_merge(base_config, yaml_config)
            print(f"[INFO] Loaded configuration from {config_path}")
            return merged_config
            
        except Exception as e:
            print(f"[ERROR] Failed to load config {config_path}: {e}")
            print(f"[INFO] Falling back to default configuration")
            return base_config

    def get_training_config(self, config_name: str) -> Dict[str, Any]:
        """Load specific training configuration"""
        config_path = self.configs_root / "training" / f"{config_name}.yaml"
        return self.load_config(config_path)

    def get_inference_config(self, config_name: str) -> Dict[str, Any]:
        """Load specific inference configuration"""
        config_path = self.configs_root / "inference" / f"{config_name}.yaml"
        return self.load_config(config_path)

    def get_preprocessing_config(self, config_name: str) -> Dict[str, Any]:
        """Load specific preprocessing configuration"""
        config_path = self.configs_root / "preprocessing" / f"{config_name}.yaml"
        return self.load_config(config_path)

    def list_configs(self, config_type: str = "training") -> list:
        """List available configurations of a specific type"""
        config_dir = self.configs_root / config_type
        if not config_dir.exists():
            return []
        return [f.stem for f in config_dir.glob("*.yaml")]

    def save_config(self, config: Dict[str, Any], config_path: str):
        """Save configuration to YAML file"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"[INFO] Configuration saved to {config_path}")

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result


# Global instance
config_manager = ConfigManager()
