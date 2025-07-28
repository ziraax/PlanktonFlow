import wandb
import yaml
import random
import numpy as np
import torch
import sys
import copy
from config import DEFAULT_CONFIG
from models.factory import create_model
from training.train import train_model
from utils.classes import get_num_classes


def sweep_train():
    #initialize wandb
    wandb.init()
    wandb_config = wandb.config

    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(dict(wandb_config))
    config["num_classes"] = get_num_classes()
    
    model = create_model(
        model_name=config['model_name'],
        num_classes=config["num_classes"],
        pretrained=True,  # if you sweep over this too
        freeze_backbone=False,
        efficientnet_variant=config.get("efficientnet_variant", "b0"),
        resnet_variant=config.get("resnet_variant", "50"),
        densenet_variant=config.get("densenet_variant", "121"),
        mc_dropout=False,
        mc_p=0.0,
    )

    
    train_model(model, config)


if __name__ == "__main__":
    # Load the YAML file properly
    with open("sweep.yaml") as f:
        sweep_config = yaml.safe_load(f)

    # This returns the sweep ID
    sweep_id = wandb.sweep(sweep=sweep_config, project=DEFAULT_CONFIG['project_name'])
    
    # This launches the agent to run sweeps
    wandb.agent(sweep_id, function=sweep_train)
