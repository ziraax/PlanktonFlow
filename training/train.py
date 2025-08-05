from datetime import datetime 
from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm 

import os
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from training.dataloader import get_test_dataloader, get_train_dataloader, get_val_dataloader
from training.eval import evaluate_metrics_only, evaluate_yolo_model, run_test_full_evaluation
from models.factory import create_model
from training.loss import FocalLoss, WeightedLoss, LabelSmoothingLoss
from utils.classes import get_class_weights
from config import DEFAULT_CONFIG
from utils.input_size import get_input_size_for_model
from training.early_stopping import EarlyStopping
from utils.config_manager import config_manager

def _initialize_wandb(config):

    model_name = config["model_name"]

    # Get the appropriate variant depending on the model
    if model_name == "resnet":
        variant = config.get("model_variant", "unknown_resnet")
    elif model_name == "densenet":
        variant = config.get("model_variant", "unknown_densenet")
    elif model_name == "efficientnet":
        variant = config.get("model_variant", "unknown_efficientnet")
    elif model_name == "yolov11":
        variant = config.get("model_variant", "Default-cls")
    else:
        variant = "n/a"

    config['model_variant'] = variant

    return wandb.init(
        project=DEFAULT_CONFIG['project_name'],
        name=f"{config['model_name']}_{config['model_variant']}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        job_type = "Training",
        config = DEFAULT_CONFIG,
        tags = ["training", config['model_name'], config['model_variant']],
    )

def _setup_training_environment(config):
    """
    Setup training environment with config-driven approach and fallback defaults.
    Loads from config file if specified, otherwise uses the existing fallback system.
    """
    if 'img_size' not in config:
        print(f"[WARNING] 'img_size' not found in config. Using default input size for model: {config['model_name']}")
        config['img_size'] = get_input_size_for_model(config, config['model_name'], config.get('model_variant', 'Default-cls'))
    
    if 'num_classes' not in config:
        print(f"[WARNING] 'num_classes' not found in config. Using class weights to determine number of classes.")
        config['num_classes'] = len(get_class_weights(config, strategy="balanced")[0])

    # Model-specific defaults (keeping existing fallback logic)
    if config['model_name'] == "yolov11":
        _apply_defaults(config, {
            "model_variant": "Default-cls",
            "batch_size": 32,
            "epochs": 30,
            "optimizer": "adamw",
            "early_stopping_patience": 15
        })
    else:
        _apply_defaults(config, {
            "loss_type": "focal",
            "focal_gamma": 2.0,
            "use_per_class_alpha": True,
            "early_stopping_patience": 20,
            "batch_size": 64,
            "optimizer": "adamw",
            "learning_rate": 0.0004175,
            "weight_decay": 0.125,
            "epochs": 30,
            "labelsmoothing_epsilon": 0.1
        })

        print(f"[INFO] Using input size: {config['img_size']} for model: {config['model_name']}")
        print(f"[INFO] Using num classes: {config['num_classes']} for model: {config['model_name']}")
        print(f"[INFO] Using batch size: {config['batch_size']} for model: {config['model_name']}")
        print(f"[INFO] Using learning rate: {config['learning_rate']} for model: {config['model_name']}")
        print(f"[INFO] Using weight decay: {config['weight_decay']} for model: {config['model_name']}")
        print(f"[INFO] Using epochs: {config['epochs']} for model: {config['model_name']}")
        print(f"[INFO] Using early stopping patience: {config['early_stopping_patience']} for model: {config['model_name']}")
        print(f"[INFO] Using loss type: {config['loss_type']}")
        
        # Show loss-specific parameters based on loss type
        if config['loss_type'] == 'focal':
            print(f"[INFO] Using focal gamma: {config['focal_gamma']}")
            print(f"[INFO] Using focal alpha per class?: {config.get('use_per_class_alpha')}")
        elif config['loss_type'] == 'labelsmoothing':
            print(f"[INFO] Using label smoothing epsilon: {config['labelsmoothing_epsilon']}")
        elif config['loss_type'] == 'weighted':
            print(f"[INFO] Using weighted loss with class weights")


def _apply_defaults(config, defaults):
    """Apply default values only if not already present in config and no config file was loaded"""
    # Don't apply defaults if config was loaded from file (except for missing keys)
    if config.get('_config_loaded', False):
        # Only set defaults for keys that are completely missing
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
    else:
        # Normal fallback behavior when no config file is used
        for key, value in defaults.items():
            config.setdefault(key, value)


def _select_loss_function(weights, device, config):
    loss_type = config['loss_type']
    if loss_type == 'focal':
        gamma = config.get('focal_gamma', 2.0)
        # Use per-class alpha if specified
        if config.get('use_per_class_alpha', False):
            _, class_frequencies = get_class_weights(config, strategy="balanced")  # You may already have this
            alpha = torch.tensor(class_frequencies, dtype=torch.float32).to(device)  # or adjust per your needs
        else:
            alpha = config.get('focal_alpha', 0.25)  # scalar fallback
        return FocalLoss(alpha=alpha, gamma=gamma, device=device)
    elif loss_type == 'labelsmoothing':
        epsilon = config.get('labelsmoothing_epsilon', 0.1)
        return LabelSmoothingLoss(epsilon=epsilon)
    elif loss_type == 'weighted':
        return WeightedLoss(class_weights=weights, device=device)
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")
    
def _select_optimizer(model, config):
    lr = float(config['learning_rate'])
    wd = float(config['weight_decay'])

    if config['optimizer'] == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif config['optimizer'] == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif config['optimizer'] == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer type: {config['optimizer']}")
    
def _select_scheduler(optimizer, config):
    # TODO : Implement a more sophisticated scheduler if needed & make parameters sweepable
    return CosineAnnealingLR(optimizer, T_max=100)

def _train_one_epoch(model, train_loader, criterion, optimizer, device, epoc, total_epochs):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoc + 1}/{total_epochs}", unit="batch", total=len(train_loader))
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

def train_model(model, config, training_config_name=None):
    """
    Train a model with optional config file support
    
    Args:
        model: The model to train
        config: Base configuration (from main.py)
        training_config_name: Optional name of training config file to load
    """
    # Load training config if specified
    if training_config_name:
        print(f"[INFO] Loading training configuration: {training_config_name}")
        training_config = config_manager.get_training_config(training_config_name)
        
        # Override model parameters from config file
        if 'model' in training_config:
            model_cfg = training_config['model']
            config.update({
                'model_name': model_cfg.get('name', config.get('model_name')),
                'model_variant': model_cfg.get('variant', config.get('model_variant')),
                'pretrained': model_cfg.get('pretrained', True),
                'freeze_backbone': model_cfg.get('freeze_backbone', False)
            })
        
        # Override training parameters
        if 'training' in training_config:
            config.update(training_config['training'])
            
        # Override loss parameters  
        if 'loss' in training_config:
            loss_config = training_config['loss'].copy()
            
            # Map 'type' to 'loss_type' for backward compatibility
            if 'type' in loss_config:
                loss_config['loss_type'] = loss_config.pop('type')
                
            config.update(loss_config)
            
        print(f"[INFO] Training config loaded: {training_config_name}")
        
        # Set flag to indicate config was loaded to prevent defaults override
        config['_config_loaded'] = True
    else:
        print(f"[INFO] Using fallback defaults for training configuration")
        config['_config_loaded'] = False

    _setup_training_environment(config)

    run= _initialize_wandb(config)
    run_name = run.name

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    if config["model_name"] == "yolov11":
        model = create_model('yolov11')
        model_path = model.train(run, config)
        evaluate_yolo_model(model_path, config['final_dataset_path'], run_name, config)
        return
    
    model.to(device)
    train_loader = get_train_dataloader(config)
    val_loader = get_val_dataloader(config)
    test_loader = get_test_dataloader(config)
    _, class_weights = get_class_weights(config, strategy="balanced")

    optimizer = _select_optimizer(model, config)
    scheduler = _select_scheduler(optimizer, config)
    criterion = _select_loss_function(class_weights, device, config)

    # Set up early stopping
    checkpoint_path = os.path.join(wandb.run.dir, "checkpoints", "checkpoint.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    early_stopper = EarlyStopping(
        patience=config["early_stopping_patience"],
        delta=0.001,
        path=checkpoint_path,
        monitor_acc=False,
    )

    for epoch in range(config['epochs']):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config['epochs'])

        val_loss, acc1, acc5, y_true, y_pred, *_ = evaluate_metrics_only(model, val_loader, criterion=criterion)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

        early_stopper(val_loss, acc1, model)

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/metrics/accuracy_top1": acc1,
            "val/metrics/accuracy_top5": acc5,
            "val/metrics/f1_macro": f1_macro,
            "val/metrics/recall_macro": recall_macro,
            "lr": scheduler.get_last_lr()[0],
        }, step=epoch + 1)

        print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Top-1 Acc: {acc1:.4f} | Top-5 Acc: {acc5:.4f}")

        scheduler.step()

        if early_stopper.early_stop:
            print(f"[INFO] Early stopping at epoch {epoch + 1}. Best Val Loss: {early_stopper.best_loss:.4f}, Top-1 Acc: {early_stopper.best_acc:.4f}")
            break

    print("[INFO] Running final evaluation on best model...")

    model.load_state_dict(torch.load(checkpoint_path))
    run_test_full_evaluation(model, test_loader, config['final_dataset_path'])

    final_model_path = os.path.join("model_weights", config["model_name"], config["model_variant"], run.name, "best.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)

    print(f"[INFO] Best model saved at: {final_model_path}")




