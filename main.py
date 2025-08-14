# /!\ This file is legacy and will be removed in future versions.
# /!\ It is kept for reference only and should not be used in new implementations.

import argparse
import copy
import wandb

from config import DEFAULT_CONFIG 
from utils.config_manager import config_manager

from training.train import train_model
from models.factory import create_model

from utils.classes import get_num_classes


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom model.")
    
    #------------ Configuration Arguments ------------
    parser.add_argument('--config', type=str, default=None, 
        help='Path to training configuration YAML file (e.g., configs/training/efficientnet_b5_production.yaml)'
    )
    parser.add_argument('--training-config', type=str, default=None,
        help='Name of training configuration (e.g., efficientnet_b5_production)'
    )
    
    #------------ Training Arguments ------------
    parser.add_argument('--model_name', type=str, default='densenet',
        choices=['yolov11', 'resnet', 'densenet', 'efficientnet'], help='Name of the model to train (default: densenet)'
    )
    # Add pretrained argument like --pretrained true or --pretrained false
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='Do not use pretrained weights')
    parser.set_defaults(pretrained=True)  # default = True

    # Add freeze_backbone argument like --freeze_backbone true or --freeze_backbone false
    parser.add_argument('--freeze_backbone', dest='freeze_backbone', action='store_true', help='Freeze the backbone of the model')
    parser.add_argument('--no-freeze_backbone', dest='freeze_backbone', action='store_false', help='Do not freeze the backbone of the model')
    parser.set_defaults(freeze_backbone=False)  # default = False

    # Add efficientnet_variant argument like --efficientnet_variant b0 or --efficientnet_variant b1
    parser.add_argument('--efficientnet_variant', type=str, default='b0',
        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'], help="Variant of EfficientNet to use (default: b0)"
    )

    # Add densenet_variant argument like --densenet_variant 121 or --densenet_variant 169
    parser.add_argument('--densenet_variant', type=str, default='121',
        choices=['121', '169', '201', '161'], help="Variant of DenseNet to use (default: 121)"
    )

    # add resnet_variant argument like --resnet_variant 18 or --resnet_variant 34
    parser.add_argument("--resnet_variant", type=str, default='50',
        choices=['18', '34', '50', '101', '152'], help="Variant of ResNet to use (default: 50)"
    )

    return parser.parse_args()

def configure_model(args, config):
    # Proper construction of full model variant names
    if args.model_name == 'densenet':
        model_variant = args.densenet_variant
    elif args.model_name == 'resnet':
        model_variant = args.resnet_variant
    elif args.model_name == 'efficientnet':
        model_variant = args.efficientnet_variant
    elif args.model_name == 'yolov11':
        model_variant = 'Default-cls'
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    print(f"[INFO][configure_model] Using model variant: {model_variant} for model: {args.model_name}")


    config.update({
        'model_name': args.model_name,
        'model_variant': model_variant,
        'num_classes': get_num_classes(),
        'img_size': 224,
    })

    return config

def main():
    try:
        args = parse_args()

        # Handle config loading
        if args.config:
            # Load from full config file path
            config = config_manager.load_config(args.config)
            print(f"[INFO] Loaded configuration from: {args.config}")
        elif args.training_config:
            # Load from training config name
            config = config_manager.get_training_config(args.training_config)
            print(f"[INFO] Loaded training configuration: {args.training_config}")
        else:
            # Use default config with command line overrides
            config = copy.deepcopy(DEFAULT_CONFIG)
            config = configure_model(args, config)

        # Extract model configuration from loaded config or use CLI args
        if args.config or args.training_config:
            # When using config files, extract model parameters
            if 'model' in config:
                model_config = config['model']
                model_name = model_config.get('name', 'densenet')
                model_variant = model_config.get('variant', 'Default')
                pretrained = model_config.get('pretrained', True)
                freeze_backbone = model_config.get('freeze_backbone', False)
            else:
                # Fallback if model section missing in config
                model_name = config.get('model_name', 'densenet')
                model_variant = config.get('model_variant', 'Default')
                pretrained = True
                freeze_backbone = False
            
            # Update config with extracted model info
            config.update({
                'model_name': model_name,
                'model_variant': model_variant,
                'num_classes': get_num_classes(),
                'img_size': 224,
            })
        else:
            # Use CLI arguments (existing logic)
            config = configure_model(args, config)
            model_name = config['model_name']
            model_variant = config['model_variant']
            pretrained = args.pretrained
            freeze_backbone = args.freeze_backbone

        print(f"[INFO] Creating model: {model_name} | Variant: {model_variant} | "
              f"Pretrained: {pretrained} | Freeze Backbone: {freeze_backbone}")      
          
        model = create_model(
                model_name, 
                num_classes=config.get('num_classes', get_num_classes()), 
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                efficientnet_variant=model_variant if model_name == 'efficientnet' else args.efficientnet_variant,
                resnet_variant=model_variant if model_name == 'resnet' else args.resnet_variant,
                densenet_variant=model_variant if model_name == 'densenet' else args.densenet_variant
        )
            
        # Run training with optional config name
        training_config_name = args.training_config if args.training_config else None
        train_model(model, config, training_config_name)

        
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()