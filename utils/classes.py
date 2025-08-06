from collections import defaultdict
from config import DEFAULT_CONFIG
from sklearn.utils import compute_class_weight

import os
import yaml
import numpy as np

from utils.logs import log_class_weights

def get_classes_names(dataset_path=None):
    """
    Returns a sorted list of class names from the training directory.
    If the directory doesn't exist, falls back to reading dataset.yaml.
    
    Args:
        dataset_path (str): Path to dataset directory. If None, uses DEFAULT_CONFIG.
    """
    if dataset_path is None:
        dataset_path = DEFAULT_CONFIG['final_dataset_path']
    
    train_path = os.path.join(dataset_path, 'train')
    
    if os.path.isdir(train_path):
        return sorted([
            d for d in os.listdir(train_path) 
            if os.path.isdir(os.path.join(train_path, d))
        ])

    # Fall back to reading from dataset.yaml
    yaml_path = os.path.join(dataset_path, 'dataset.yaml')
    if os.path.isfile(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return sorted(data.get('names', []))
    
    # If neither source exists
    return []


def get_num_classes(dataset_path=None):
    """
    Returns the number of classes in the dataset.
    Defaults to 76 if no classes are found.
    
    Args:
        dataset_path (str): Path to dataset directory. If None, uses DEFAULT_CONFIG.
    """
    return len(get_classes_names(dataset_path)) or 76


def get_class_distribution(dataset_path=None):
    """
    Analyzes how many images exist per class across train/val/test.
    Returns a dictionary mapping class names to image counts.
    
    Args:
        dataset_path (str): Path to dataset directory. If None, uses DEFAULT_CONFIG.
    """
    if dataset_path is None:
        dataset_path = DEFAULT_CONFIG['final_dataset_path']
        
    class_counts = defaultdict(int)
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        if os.path.isdir(split_path):
            for class_dir in os.listdir(split_path):
                class_path = os.path.join(split_path, class_dir)
                if os.path.isdir(class_path):
                    count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    class_counts[class_dir] += count
    return class_counts



def get_class_weights(config, strategy="balanced"):
    """
    Computes class weights using sklearn's compute_class_weight.
    
    Args:
        config (dict): Training configuration containing dataset_path
        strategy (str): "balanced" or "uniform". Only "balanced" computes true weights.
    
    Saves:
        CONFIG['class_weights' ] - a list of float weights matching the class order. 
    """ 
    
    # Get dataset path from config - try multiple possible keys
    dataset_path = (
        config.get('dataset_path') or 
        config.get('final_dataset_path') or 
        config.get('data', {}).get('dataset_path') or
        DEFAULT_CONFIG['final_dataset_path']
    )
    
    class_counts = get_class_distribution(dataset_path)
    classes = sorted(class_counts.keys())

    if strategy == "uniform":
        weights = np.ones(len(classes))
    else:
        y = []
        for cls_idx, cls in enumerate(classes):
            y.extend([cls_idx] * class_counts[cls])
        weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(classes)), y=np.array(y))

    config['class_weights'] = weights.tolist()
    print(f"[INFO] Computed class weights for {len(classes)} classes: {config['class_weights']}")
    return classes, weights




