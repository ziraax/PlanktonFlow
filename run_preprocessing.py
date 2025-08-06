#!/usr/bin/env python3
"""
Config-driven preprocessing runner script
Supports multiple input formats: hierarchical, CSV mapping, and EcoTaxa
"""

import sys
import os
import argparse
import wandb
from datetime import datetime

# Ensure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.config_preprocessing import ConfigPreprocessor
from utils.config_manager import config_manager
from utils.pytorch_cuda import check_pytorch_cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Run config-driven preprocessing pipeline.")
    
    #------------ Configuration Arguments ------------
    parser.add_argument('--config', type=str, default=None,
        help='Path to preprocessing configuration YAML file'
    )
    parser.add_argument('--preprocessing-config', type=str, default=None,
        help='Name of preprocessing configuration (e.g., simple_hierarchical)'
    )
    
    #------------ Override Arguments (optional) ------------
    parser.add_argument('--input-path', type=str, default=None,
        help='Override input data path'
    )
    parser.add_argument('--output-path', type=str, default=None,
        help='Override output base path'
    )
    parser.add_argument('--no-wandb', action='store_true',
        help='Disable W&B logging even if enabled in config'
    )
    parser.add_argument('--dry-run', action='store_true',
        help='Validate configuration without running preprocessing'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate arguments
    if not args.config and not args.preprocessing_config:
        print("[ERROR] Either --config or --preprocessing-config must be specified")
        sys.exit(1)
    
    # Load configuration
    if args.config:
        # Load from full config file path
        config = config_manager.load_config(args.config)
        print(f"[INFO] Loaded preprocessing configuration from: {args.config}")
    else:
        # Load from preprocessing config name
        config = config_manager.get_preprocessing_config(args.preprocessing_config)
        print(f"[INFO] Loaded preprocessing configuration: {args.preprocessing_config}")
    
    # Apply argument overrides
    if args.input_path:
        config['input_source']['data_path'] = args.input_path
        print(f"[INFO] Override input path: {args.input_path}")
    
    if args.output_path:
        config['output']['base_path'] = args.output_path
        config['output']['processed_path'] = f"{args.output_path}/processed"
        config['output']['final_dataset_path'] = f"{args.output_path}/final_dataset"
        print(f"[INFO] Override output path: {args.output_path}")
    
    if args.no_wandb:
        config['logging']['wandb_enabled'] = False
        print("[INFO] W&B logging disabled by --no-wandb flag")
    
    # Print configuration summary
    print("\n" + "="*60)
    print("PREPROCESSING CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Input Type:     {config['input_source']['type']}")
    
    # Handle different input source structures
    input_type = config['input_source']['type']
    if input_type == 'csv_mapping':
        print(f"Images Path:    {config['input_source']['images_path']}")
        print(f"Metadata File:  {config['input_source']['metadata_file']}")
    elif input_type == 'ecotaxa':
        print(f"Data Path:      {config['input_source']['data_path']}")
        print(f"Metadata File:  {config['input_source']['metadata_file']}")
    elif input_type == 'hierarchical':
        print(f"Input Path:     {config['input_source'].get('data_path', 'N/A')}")
    else:
        # For other types, try data_path as fallback
        print(f"Input Path:     {config['input_source'].get('data_path', 'N/A')}")
    
    print(f"Output Path:    {config['output']['base_path']}")
    print(f"Scalebar Removal: {config['preprocessing']['scalebar_removal']['enabled']}")
    print(f"Grayscale Conv: {config['preprocessing']['grayscale_conversion']['enabled']}")
    print(f"Augmentation:   {config['preprocessing']['augmentation']['enabled']}")
    print(f"W&B Logging:    {config['logging']['wandb_enabled']}")
    print("="*60)
    
    if args.dry_run:
        print("[INFO] Dry run completed. Configuration is valid.")
        return
    
    # Check PyTorch and CUDA
    check_pytorch_cuda()
    
    # Initialize W&B if enabled
    if config['logging'].get('wandb_enabled', False):
        config_name = args.preprocessing_config or os.path.basename(args.config).replace('.yaml', '')
        
        wandb.init(
            project=config.get('project_name', 'YOLOTaxa-preprocessing'),
            name=f"preprocessing_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            job_type="preprocessing",
            config=config,
            tags=["preprocessing", config['input_source']['type']],
            notes=f"Preprocessing with {config_name} configuration"
        )
        print("[INFO] W&B initialized for experiment tracking")
    
    try:
        # Run preprocessing
        print("\n[INFO] Starting preprocessing pipeline...")
        preprocessor = ConfigPreprocessor(config)
        preprocessor.run_full_pipeline()
        
        print("\n[SUCCESS] Preprocessing completed successfully!")
        
        # Print final dataset info
        final_path = config['output']['final_dataset_path']
        print(f"\n[INFO] Final dataset available at: {final_path}")
        if os.path.exists(f"{final_path}/dataset.yaml"):
            print(f"[INFO] Dataset YAML created: {final_path}/dataset.yaml")
        
    except Exception as e:
        print(f"\n[ERROR] Preprocessing failed: {e}")
        raise
    
    finally:
        # Finish W&B run
        if config['logging'].get('wandb_enabled', False) and wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
