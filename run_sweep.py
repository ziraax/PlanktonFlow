import wandb
import yaml
import random
import numpy as np
import torch
import sys
import copy
import argparse
from config import DEFAULT_CONFIG
from models.factory import create_model
from training.train import train_model
from utils.classes import get_num_classes


def sweep_train():
    # Initialize wandb (this is done by the sweep agent)
    wandb.init()
    wandb_config = wandb.config

    # Start with base config and update with sweep parameters
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(dict(wandb_config))
    config["num_classes"] = get_num_classes()
    
    # Set model variant based on the model type and variant parameters
    if config['model_name'] == 'densenet':
        config['model_variant'] = f"densenet{config.get('densenet_variant', '121')}"
    elif config['model_name'] == 'resnet':
        config['model_variant'] = f"resnet{config.get('resnet_variant', '50')}"
    elif config['model_name'] == 'efficientnet':
        config['model_variant'] = f"efficientnet_b{config.get('efficientnet_variant', 'b0').replace('b', '')}"
    else:
        config['model_variant'] = 'default'
    
    # Mark as config loaded to prevent defaults override
    config['_config_loaded'] = True
    
    # Ensure W&B is enabled for sweeps (since wandb.init() already called)
    config['log_results'] = True
    
    # Set a custom run name for the sweep
    config['run_name'] = f"sweep_{wandb.run.name}"
    
    model = create_model(
        model_name=config['model_name'],
        num_classes=config["num_classes"],
        pretrained=True,
        freeze_backbone=False,
        efficientnet_variant=config.get("efficientnet_variant", "b0"),
        resnet_variant=config.get("resnet_variant", "50"),
        densenet_variant=config.get("densenet_variant", "121"),
    )

    # Pass the existing wandb run to avoid double initialization
    train_model(model, config, training_config_name=None)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep')
    parser.add_argument('--sweep_config', type=str, required=True,
                        help='Path to the sweep configuration YAML file')
    args = parser.parse_args()
    
    # Load the YAML file properly
    with open(args.sweep_config) as f:
        sweep_config = yaml.safe_load(f)

    # This returns the sweep ID
    sweep_id = wandb.sweep(sweep=sweep_config, project=DEFAULT_CONFIG['project_name'])
    
    # This launches the agent to run sweeps
    wandb.agent(sweep_id, function=sweep_train)
