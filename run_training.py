#!/usr/bin/env python3
"""
Config-driven training script for YOLOTaxa models.
Uses the ConfigManager system for flexible, YAML-based configuration.
"""

import argparse
import sys
from pathlib import Path

# Add utils to path for ConfigManager
sys.path.append(str(Path(__file__).parent / "utils"))

from config_manager import ConfigManager
from training.train import train_model
from models.factory import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using config-driven approach")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Training config name (e.g., efficientnet_b5_production)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show configuration without training"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration using ConfigManager
    config_manager = ConfigManager()
    config = config_manager.get_training_config(args.config)
    
    print("=" * 60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Model: {config.get('model', {}).get('name', config.get('model_name', 'Unknown'))}")
    print(f"Variant: {config.get('model', {}).get('variant', 'Default')}")
    print(f"Pretrained: {config.get('model', {}).get('pretrained', True)}")
    print(f"Freeze Backbone: {config.get('model', {}).get('freeze_backbone', False)}")
    print(f"Batch Size: {config.get('training', {}).get('batch_size', config.get('batch_size', 32))}")
    print(f"Learning Rate: {config.get('training', {}).get('learning_rate', config.get('learning_rate', 0.001))}")
    print(f"Epochs: {config.get('training', {}).get('epochs', config.get('epochs', 50))}")
    print(f"Optimizer: {config.get('training', {}).get('optimizer', config.get('optimizer', 'adamw'))}")
    print(f"Loss Type: {config.get('loss', {}).get('type', config.get('loss_type', 'focal'))}")
    print(f"Dataset Path: {config.get('data', {}).get('dataset_path', config.get('final_dataset_path', 'DATA/final_dataset'))}")
    print(f"Input Size: {config.get('model', {}).get('input_size', config.get('img_size', 224))}")
    print(f"Device: {config.get('device', 'cuda')}")
    print("=" * 60)
    
    if args.dry_run:
        print("[INFO] Dry run mode - exiting without training")
        return
    
    # Normalize config structure for training.train compatibility
    normalized_config = normalize_config_for_training(config)
    
    # Create model using config
    model = create_model(
        model_name=normalized_config['model_name'],
        num_classes=normalized_config.get('num_classes', 76),  # Default to 76 classes
        pretrained=normalized_config.get('pretrained', True),
        freeze_backbone=normalized_config.get('freeze_backbone', False),
        efficientnet_variant=normalized_config.get('model_variant', 'b0'),
        densenet_variant=normalized_config.get('model_variant', '121'),
        resnet_variant=normalized_config.get('model_variant', '50')
    )
    
    print(f"[INFO] Starting training with config: {args.config}")
    
    # Train the model
    train_model(model, normalized_config)
    
    print("[SUCCESS] Training completed!")


def normalize_config_for_training(config):
    """
    Normalize nested YAML config structure to flat structure expected by training.train
    """
    normalized = config.copy()
    
    # Extract nested project config
    if 'project' in config:
        project_config = config['project']
        normalized['project_name'] = project_config.get('name', 'YOLOv11Classification500')
    
    # Extract nested data config
    if 'data' in config:
        data_config = config['data']
        normalized['raw_data_root'] = data_config.get('raw_data_root', 'DATA/DATA_500_INDIV')
        normalized['processed_path'] = data_config.get('processed_path', 'DATA/processed_dataset')
        normalized['final_dataset_path'] = data_config.get('dataset_path', 'DATA/final_dataset')
    
    # Extract nested model config
    if 'model' in config:
        model_config = config['model']
        normalized['model_name'] = model_config.get('name', config.get('model_name', 'efficientnet'))
        normalized['model_variant'] = model_config.get('variant', 'b0')
        normalized['pretrained'] = model_config.get('pretrained', True)
        normalized['freeze_backbone'] = model_config.get('freeze_backbone', False)
        normalized['img_size'] = model_config.get('input_size', config.get('img_size', 224))
        normalized['num_classes'] = model_config.get('num_classes', 76)
    
    # Extract nested training config
    if 'training' in config:
        training_config = config['training']
        normalized['batch_size'] = training_config.get('batch_size', 32)
        normalized['learning_rate'] = training_config.get('learning_rate', 0.001)
        normalized['weight_decay'] = training_config.get('weight_decay', 0.01)
        normalized['epochs'] = training_config.get('epochs', 50)
        normalized['optimizer'] = training_config.get('optimizer', 'adamw')
        normalized['early_stopping_patience'] = training_config.get('early_stopping_patience', 15)
        normalized['device'] = training_config.get('device', 'cuda')
        normalized['num_workers'] = training_config.get('num_workers', 8)
    
    # Extract nested loss config
    if 'loss' in config:
        loss_config = config['loss']
        normalized['loss_type'] = loss_config.get('type', 'focal')
        normalized['focal_gamma'] = loss_config.get('focal_gamma', 2.0)
        normalized['use_per_class_alpha'] = loss_config.get('use_per_class_alpha', True)
    
    return normalized


if __name__ == "__main__":
    main()
