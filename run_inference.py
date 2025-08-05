#!/usr/bin/env python3
"""
Inference script that can be run from the root directory.
This is a wrapper around the inference module to handle import paths correctly.
"""

import sys
import os
import argparse
import wandb
import torch
from PIL import Image
from datetime import datetime

# Ensure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.runner import run_inference
from config import DEFAULT_CONFIG
from utils.config_manager import config_manager

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on images using trained models.")
    
    #------------ Configuration Arguments ------------
    parser.add_argument('--config', type=str, default=None,
        help='Path to inference configuration YAML file (e.g., configs/inference/efficientnet_b5_best.yaml)'
    )
    parser.add_argument('--inference-config', type=str, default=None,
        help='Name of inference configuration (e.g., efficientnet_b5_best)'
    )
    
    #------------ Required Arguments (if not using config) ------------
    parser.add_argument('--image_dir', type=str, 
        help='Directory containing images for inference')
    parser.add_argument('--model_name', type=str, choices=['resnet', 'densenet', 'efficientnet', 'yolov11'],
        help='Name of the model to use for inference')
    parser.add_argument('--weights_path', type=str,
        help='Path to the trained model weights')
    
    #------------ Model Variant Arguments ------------
    parser.add_argument('--efficientnet_variant', type=str)
    parser.add_argument('--resnet_variant', type=str)
    parser.add_argument('--densenet_variant', type=str)

    #------------ Inference Arguments ------------
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--save_csv', type=str, help='Path to save inference results as CSV')
    parser.add_argument('--wandb_log', action='store_true', help='Log results to Weights & Biases')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Handle config loading
    if args.config:
        # Load from full config file path
        config = config_manager.load_config(args.config)
        print(f"[INFO] Loaded inference configuration from: {args.config}")
    elif args.inference_config:
        # Load from inference config name
        config = config_manager.get_inference_config(args.inference_config)
        print(f"[INFO] Loaded inference configuration: {args.inference_config}")
    else:
        # Use command line arguments (existing behavior)
        config = DEFAULT_CONFIG.copy()
        
        # Validate required arguments
        if not args.image_dir:
            raise ValueError("--image_dir is required when not using config file")
        if not args.model_name:
            raise ValueError("--model_name is required when not using config file")
        if not args.weights_path:
            raise ValueError("--weights_path is required when not using config file")

    # Extract inference parameters from config or use CLI args
    if args.config or args.inference_config:
        # Extract from config file
        model_config = config.get('model', {})
        inference_config = config.get('inference', {})
        
        # Build args-like object from config
        class ConfigArgs:
            def __init__(self, model_cfg, inf_cfg, cli_args):
                self.image_dir = cli_args.image_dir or inf_cfg.get('image_dir')
                self.model_name = model_cfg.get('name')
                self.weights_path = model_cfg.get('weights_path')
                
                # Model variants
                variant = model_cfg.get('variant', 'Default')
                self.efficientnet_variant = variant if self.model_name == 'efficientnet' else None
                self.densenet_variant = variant if self.model_name == 'densenet' else None
                self.resnet_variant = variant if self.model_name == 'resnet' else None
                
                # Inference params
                self.device = inf_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
                self.batch_size = inf_cfg.get('batch_size', 16)
                self.top_k = inf_cfg.get('top_k', 3)
                self.save_csv = inf_cfg.get('output_path') if inf_cfg.get('save_csv') else None
                self.wandb_log = config.get('wandb', {}).get('log_results', False)
                
                # Validate required fields
                if not self.image_dir:
                    raise ValueError("image_dir must be specified in config or command line")
                if not self.model_name:
                    raise ValueError("model.name must be specified in config")
                if not self.weights_path:
                    raise ValueError("model.weights_path must be specified in config")
        
        # Create args object from config
        inference_args = ConfigArgs(model_config, inference_config, args)
        print(f"[INFO] Running inference: {inference_args.model_name} | Weights: {inference_args.weights_path}")
    else:
        # Use CLI args directly
        inference_args = args
        print(f"[INFO] Running inference from CLI args: {args.model_name}")

    # Run inference
    results = run_inference(inference_args, config)
    
    # Handle W&B logging
    if inference_args.wandb_log:
        wandb_config = config.get('wandb', {}) if (args.config or args.inference_config) else {}
        
        wandb.init(
            project=config.get('project_name', 'classification'),
            name=f"{inference_args.model_name}_{getattr(inference_args, 'efficientnet_variant', '') or getattr(inference_args, 'densenet_variant', '') or getattr(inference_args, 'resnet_variant', '') or 'Default'}_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            job_type="Inference",
            config={
                "image_dir": inference_args.image_dir,
                "model_name": inference_args.model_name,
                "weights_path": inference_args.weights_path,
                "device": inference_args.device,
                "batch_size": inference_args.batch_size,
                "top_k": inference_args.top_k,
            },
            tags=wandb_config.get('tags', ["inference", inference_args.model_name]),
            notes=wandb_config.get('notes', "Inference run"),
        )

        # Create W&B table
        columns = ["Image"]
        for i in range(inference_args.top_k):
            columns += [f"Top{i+1}", f"Top{i+1} Score"]

        table = wandb.Table(columns=columns)

        for item in results:
            img = Image.open(item['image_path'])
            preds = item['predictions']
            row = [wandb.Image(img)]

            for pred in preds:
                label, score = pred
                row += [label, score]

            # Pad incomplete rows if fewer than top_k
            while len(row) < len(columns):
                row += ["", 0.0]

            table.add_data(*row)

        wandb.log({"predictions": table})
        wandb.finish()
        
    print(f"[INFO] Inference completed successfully!")

if __name__ == "__main__":
    main()
