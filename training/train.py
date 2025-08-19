from datetime import datetime 
from sklearn.metrics import f1_score, recall_score, precision_score
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
        print(f"[WARNING] 'img_size' not found in config. Using standard 224x224 input size for all models.")
        config['img_size'] = 224
    
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
            "early_stopping_patience": 15,
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
    # Config is already loaded and normalized by run_training.py
    # Just ensure the _config_loaded flag is set
    if training_config_name:
        print(f"[INFO] Training config loaded: {training_config_name}")
        config['_config_loaded'] = True
    else:
        print(f"[INFO] Using fallback defaults for training configuration")
        config['_config_loaded'] = False

    _setup_training_environment(config)

    # Handle W&B initialization (check if already active for sweeps)
    if config.get('log_results', True):
        # Check if W&B is already active (e.g., from sweep)
        if wandb.run is not None:
            run = wandb.run
            run_name = run.name
            print(f"[INFO] Using existing W&B run (sweep): {run_name}")
        else:
            run = _initialize_wandb(config)
            run_name = run.name
            print(f"[INFO] W&B logging enabled - Run: {run_name}")
    else:
        run = None
        # Use user-specified run name or generate default
        user_run_name = config.get('run_name')
        if user_run_name:
            run_name = user_run_name + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"[INFO] W&B logging disabled - Using custom run name: {run_name}")
        else:
            run_name = f"no_wandb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['model_name']}_{config['model_variant']}"
            print(f"[INFO] W&B logging disabled - Using auto-generated run name: {run_name}")

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    if config["model_name"] == "yolov11":
        model = create_model('yolov11')
        model_path = model.train(run, config)
        evaluate_yolo_model(model_path, config['final_dataset_path'], config['run_name'], config, run=run)
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
    if run:
        checkpoint_path = os.path.join(run.dir, "checkpoints", "checkpoint.pt")
    else:
        # Create local checkpoint directory when W&B is disabled
        checkpoint_path = os.path.join("model_weights", config["model_name"], 
                                     config.get("model_variant", "default"), 
                                     run_name, "checkpoint.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Set up training log file for non-W&B runs
    if not run:
        log_dir = os.path.join("model_weights", config["model_name"], 
                              config.get("model_variant", "default"), run_name)
        os.makedirs(log_dir, exist_ok=True)
        training_log_path = os.path.join(log_dir, "training_log.txt")
        training_csv_path = os.path.join(log_dir, "training_log.csv")
        
        # Write header to text log file
        with open(training_log_path, 'w') as f:
            f.write(f"Training Log - {run_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Model: {config['model_name']} {config.get('model_variant', 'default')}\n")
            f.write(f"Dataset: {config.get('final_dataset_path', 'Unknown')}\n")
            f.write(f"Batch Size: {config['batch_size']}\n")
            f.write(f"Learning Rate: {config['learning_rate']}\n")
            f.write(f"Loss Type: {config['loss_type']}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            f.write("Epoch | Train Loss | Val Loss | Top-1 Acc | Top-5 Acc | F1 Macro | Recall Macro | Precision Macro | LR\n")
            f.write("-" * 80 + "\n")
        
        # Write header to CSV log file
        with open(training_csv_path, 'w') as f:
            f.write("epoch,train_loss,val_loss,top1_acc,top5_acc,f1_macro,recall_macro,precision_macro,learning_rate,timestamp\n")
        
        print(f"[INFO] Training log will be saved to: {training_log_path}")
        print(f"[INFO] Training CSV will be saved to: {training_csv_path}")
    else:
        training_log_path = None
        training_csv_path = None

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
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)

        early_stopper(val_loss, acc1, model)

        # Only log to W&B if enabled
        if run:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/metrics/accuracy_top1": acc1,
                "val/metrics/accuracy_top5": acc5,
                "val/metrics/f1_macro": f1_macro,
                "val/metrics/recall_macro": recall_macro,
                "val/metrics/precision_macro": precision_macro,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch + 1)
        else:
            # Log to text file when W&B is disabled
            with open(training_log_path, 'a') as f:
                f.write(f"{epoch + 1:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {acc1:9.4f} | {acc5:9.4f} | {f1_macro:8.4f} | {recall_macro:12.4f} | {precision_macro:12.4f} | {scheduler.get_last_lr()[0]:.2e}\n")

            # Log to CSV file when W&B is disabled
            if training_csv_path:
                with open(training_csv_path, 'a') as f:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{epoch + 1},{train_loss:.6f},{val_loss:.6f},{acc1:.6f},"
                           f"{acc5:.6f},{f1_macro:.6f},{recall_macro:.6f},{precision_macro:.6f},{scheduler.get_last_lr()[0]:.8e},{timestamp}\n")

        print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Top-1 Acc: {acc1:.4f} | Top-5 Acc: {acc5:.4f} | F1 Macro: {f1_macro:.4f} | Recall Macro: {recall_macro:.4f} | Precision Macro: {precision_macro:.4f}")

        scheduler.step()
        
        # Save epoch checkpoint
        if run:
            epoch_checkpoint_path = os.path.join(run.dir, "checkpoints", f"checkpoint_epoch_{epoch + 1}.pt")
        else:
            checkpoint_dir = os.path.join("model_weights", config["model_name"], 
                                        config.get("model_variant", "default"), run_name, "checkpoints")
            epoch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        
        # Ensure checkpoint directory exists
        os.makedirs(os.path.dirname(epoch_checkpoint_path), exist_ok=True)
        
        # Save the epoch checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': acc1,
            'f1_macro': f1_macro,
            'recall_macro': recall_macro,
            'precision_macro': precision_macro,
        }, epoch_checkpoint_path)

        if early_stopper.early_stop:
            print(f"[INFO] Early stopping at epoch {epoch + 1}. Best Val Loss: {early_stopper.best_loss:.4f}, Top-1 Acc: {early_stopper.best_acc:.4f}")
            # Log early stopping to text file
            if training_log_path:
                with open(training_log_path, 'a') as f:
                    f.write(f"\nEarly stopping at epoch {epoch + 1}\n")
                    f.write(f"Best Val Loss: {early_stopper.best_loss:.4f}, Top-1 Acc: {early_stopper.best_acc:.4f}\n")
            
            # Log early stopping to CSV file
            if training_csv_path:
                with open(training_csv_path, 'a') as f:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"# Early stopping at epoch {epoch + 1} - Best Val Loss: {early_stopper.best_loss:.4f}, Top-1 Acc: {early_stopper.best_acc:.4f}, Time: {timestamp}\n")
            break

    # Log completion to text file
    if training_log_path:
        with open(training_log_path, 'a') as f:
            f.write(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
    
    # Log completion to CSV file
    if training_csv_path:
        with open(training_csv_path, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"# Training completed at: {timestamp}\n")

    print("[INFO] Running final evaluation on best model...")

    model.load_state_dict(torch.load(checkpoint_path))
    run_test_full_evaluation(model, test_loader, config['final_dataset_path'], run=run, run_name=run_name, config=config)

    # Save final model
    if run:
        final_model_path = os.path.join("model_weights", config["model_name"], config["model_variant"], run.name, "best.pt")
    else:
        final_model_path = os.path.join("model_weights", config["model_name"], config["model_variant"], run_name, "best.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)

    print(f"[INFO] Best model saved at: {final_model_path}")




