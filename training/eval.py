from datetime import datetime 
import wandb
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


from PIL import Image
from tqdm import tqdm

def evaluate_yolo_model(model_path, data_dir, run_name, config):

    wandb.init(
        project=config['project_name'],
        name=f"{config['model_name']}_{config['model_variant']}_evaluation_{run_name}",
        job_type="Evaluation",
        config=config,
        tags=["evaluation", "custom_model", "classification", config['model_name']],
    )


    data_dir = Path(data_dir).resolve()
    yaml_path = data_dir / "dataset.yaml"
        
    with open(yaml_path) as f:
        data_cfg = yaml.safe_load(f)
        
    class_names = data_cfg['names']
    test_dir = data_dir / data_cfg['test']
        
    # Map: class name -> class index
    class_to_idx = {name: i for i, name in enumerate(class_names)}
        
    # Load model
    model = YOLO(model_path)
        
    y_true, y_pred = [], []
        
    # Loop through val folder
    for class_name in os.listdir(test_dir):
        class_folder = test_dir / class_name
        if not class_folder.is_dir():
            continue
        label_idx = class_to_idx[class_name]
        for img_file in tqdm(list(class_folder.glob("*"))):
            if not img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                continue
            try:
                pred = model.predict(str(img_file), verbose=False)[0]
                pred_label = pred.probs.top1
                y_true.append(label_idx)
                y_pred.append(pred_label)
            except Exception as e:
                print(f"Error on image {img_file}: {e}")
        

    # Log standard metrics
    print("[INFO] Logging standard metrics...")
    log_core_metrics(y_true, y_pred, class_names)
    print("[INFO] Logging evaluation details...")
    log_evaluation_details(y_true, y_pred, class_names)  
    print("[INFO] Evaluation complete.")
    wandb.finish()

def evaluate_metrics_only(model, split, criterion):
    model.eval()
    device = next(model.parameters()).device

    y_true, y_pred, y_pred_logits = [], [], []
    val_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(split, desc="Validating", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_pred_logits.append(outputs.cpu())

            if criterion:
                batch_loss = criterion(outputs, labels).item()
                val_loss += batch_loss
                pbar.set_postfix(loss=batch_loss)


    if criterion:
        val_loss /= len(split)

    y_pred_logits = torch.cat(y_pred_logits)
    top1 = accuracy_score(y_true, y_pred)

    y_true_tensor = torch.tensor(y_true)
    top5_preds = torch.topk(y_pred_logits, 5, dim=1).indices
    top5_correct = top5_preds.eq(y_true_tensor.view(-1, 1)).sum().item()
    top5 = top5_correct / len(y_true)



    return val_loss, top1, top5, y_true, y_pred, y_pred_logits

# ---- LOGGING FUNCTIONS ---- #

def log_core_metrics(y_true, y_pred, class_names, prefix="test"):
    wandb.log({
        f"{prefix}/metrics/accuracy": accuracy_score(y_true, y_pred),        
        f"{prefix}/metrics/precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}/metrics/precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}/metrics/recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}/metrics/recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}/metrics/f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}/metrics/f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),

    })

def log_evaluation_details(y_true, y_pred, class_names):
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df.rename(columns={"index": "class"}, inplace=True)
    # Filter the accuracy, macro avg, and weighted avg rows
    report_df = report_df[~report_df['class'].isin(['accuracy', 'macro avg', 'weighted avg'])]
    wandb.log({"classification_report": wandb.Table(dataframe=report_df)})

    # F1 Score
    f1_sorted_df = report_df[report_df['class'].isin(class_names)].sort_values("f1-score")
    plt.figure(figsize=(12, max(6, len(f1_sorted_df) * 0.3)))
    sns.barplot(data=f1_sorted_df, x='f1-score', y='class', hue='class', dodge=False, legend=False, palette='viridis')
    plt.title("F1 Score per Class (Sorted)")
    plt.xlim(0, 1)
    wandb.log({"f1_score_per_class": wandb.Image(plt)})
    plt.close()

    # Precision
    precision_sorted = report_df.sort_values("precision")
    plt.figure(figsize=(12, max(6, len(precision_sorted) * 0.3)))
    sns.barplot(data=precision_sorted, x='precision', y='class', hue='class', dodge=False, legend=False, palette='mako')
    plt.title("Precision per Class (Sorted)")
    plt.xlim(0, 1)
    wandb.log({"precision_per_class": wandb.Image(plt)})
    plt.close()

    # Recall
    recall_sorted = report_df.sort_values("recall")
    plt.figure(figsize=(12, max(6, len(recall_sorted) * 0.3)))
    sns.barplot(data=recall_sorted, x='recall', y='class', hue='class', dodge=False, legend=False, palette='crest')
    plt.title("Recall per Class (Sorted)")
    plt.xlim(0, 1)
    wandb.log({"recall_per_class": wandb.Image(plt)})
    plt.close()

    # Confusion matrix raw
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 15))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Raw Counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    wandb.log({"confusion_matrix_raw": wandb.Image(plt)})
    plt.close()

    # Confusion matrix normalized
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(20, 15))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    wandb.log({"normalized_confusion_matrix": wandb.Image(plt)})
    plt.close()



def run_test_full_evaluation(model, test_loader, data_dir, criterion=None):
    test_loss, top1, top5, y_true, y_pred, y_pred_logits = evaluate_metrics_only(model, test_loader, criterion=None)

    with open(Path(data_dir) / "dataset.yaml") as f:
        class_names = yaml.safe_load(f)['names']

    log_core_metrics(y_true, y_pred, class_names, prefix="test")
    log_evaluation_details(y_true, y_pred, class_names)
