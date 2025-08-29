import torch
from pathlib import Path

# Default paths - will be overridden by YAML configs and serve as a fallback
RAW_DATA_ROOT = Path("DATA/raw_dataset")
PROCESSED_PATH = Path("DATA/processed_dataset")
FINAL_DATASET_PATH = Path("DATA/final_dataset")

# Minimal default configuration - most settings now come from YAML files
DEFAULT_CONFIG = {
    # Basic project info
    "project_name": "YourWandBProjectName",
    
    # Default paths (overridden by YAML configs)
    "raw_data_root": str(RAW_DATA_ROOT),
    "processed_path": str(PROCESSED_PATH),
    "final_dataset_path": str(FINAL_DATASET_PATH),
    
    # Scale bar removal 
    "scalebar_model_path": "models/model_weights/scale_bar_remover/best.pt", # Path to the YOLO model for scale bar removal
    "scalebar_img_size": 416,
    "scalebar_confidence": 0.4,
    "convert_grayscale": True, # convert all images to grayscale since some of them are RGB
    "grayscale_mode": "RGB",
    
    # Basic hardware settings
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8,
    
    # Fallback model settings
    "model_name": "yolo11l-cls.pt",
    "img_size": 224,
}