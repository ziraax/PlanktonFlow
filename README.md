# PlanktonFlow - an End-to-end Deep Learning Pipeline for Automatic Plankton Classification

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![W&B Integration](https://img.shields.io/badge/Weights_&_Biases-Integrated-yellow)

An end-to-end deep learning solution supporting multiple model architectures with advanced features for training, evaluation, and production-ready inference.


## Table of Contents

- [Features](#features)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Inference](#inference)
- [Installation](#installation)
- [Usages](#usages)
  - [To preprocess your data](#to-preprocess-your-data)
    - [1. Hierarchical Data Format (Folder-based)](#1-hierarchical-data-format-folder-based)
    - [2. CSV/TSV Mapping Format](#2-csvtsv-mapping-format)
    - [3. EcoTaxa Format](#3-ecotaxa-format)
  - [To train a model](#to-train-a-model)
    - [Basic Training Example](#basic-training-example)
    - [Advanced Training Features](#advanced-training-features)
  - [Inference (Making predictions)](#inference-making-predictions)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Quick Start Examples](#quick-start-examples)
- [Results and monitoring](#results-and-monitoring)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Features

### Preprocessing
Tailored for our dataset — customizable for yours.
- **Multiple Data Input Format**: Wether your dataset is from EcoTaxa, already organised in classical classification form or using a CSV/TSV file, everything is implemented
- **Data Augmentation**: Allows to augment the training data to limit the class imbalance issue
- **Scalebar Removal**: Allows to detect and delete scale bars that were originally present in some images using a YOLOv8 model. 
- **Automatic Data Splitting**: The tool makes preprocessing entirely configurable, with automatic data splitting for training/evaluation and generates automaticly yaml configuration files


### Training
- **Multi-model Support**: YOLOv11, ResNet, DenseNet, EfficientNet
- **Advanced Training**:
  - Configurable hyperparameters
  - Early stopping with checkpointing
  - Multiple loss functions
  - More features
- **Model Factory Pattern**: Dynamic model creation with variants
- **Tracking & Integration**: Real-time tracking of metrics, weights, and model versions using either Weights & Biases or our custom module

### Inference
- **Batch Processing**: Efficient handling of image directories
- **Flexible Output**:
  - Top-K predictions
  - CSV export capabilities
- **Production Ready**: Device-aware execution (CPU/GPU)

## Installation

This installation process covers all steps since this project aims to be used by biologists that could be not familiar with setting up such projects. For more exeperienced users, it follows the general process of setting up a virtual environment, activating it, installing dependencies and running Python scripts. 

1. **Install Python**:

This project uses Python to work, and was developped using Python 3.12.3. Please download Python with this link (official) : [Python-3.12.3](https://www.python.org/downloads/release/python-3123/) by clicking on the version corresponding to your operating system. 

Make sure to check "Add Python to PATH" during installation. 

2. **Clone or Download the Repository**:

Option 1 : Using Git for more experienced users

```bash
git clone https://github.com/ziraax/TaxoNet.git
cd TaxoNet
```

Option 2 : Using the download button

If you are not familiar with Git, you can simply click the green "<> Code" button and click on "Download ZIP". Then, simply extract the project where you want it to be on your computer. 

3. **Create a virtual environment**:

Virtual environments in Python are isolated directories that contain their own Python interpreter and libraries, allowing you to manage dependencies for each project separately. This prevents conflicts between packages required by different projects and ensures reproducible setups. This comes in handy in a project like this one, where there are a lot of dependencies.

To create a virtual environment, open a terminal in the folder where you downloaded the project and run :

```bash
python3 -m venv .venv
```

Where **.venv** will be the name of the folder holding the virtual environment. 


4. **Activate the Environment**:

Now, depending on your Operating System : 

 - On Windows using your Terminal (CMD), type:
```bash
.venv\Scripts\activate
```

- On Windows Powershell : 

```
.\.venv\Scripts\Activate.ps1
```

- Using bash : 

```bash
source .venv/bin/activate
```

After activation, your terminal will change to show the venv name. 

⚠️ In case you run into a bug like "Cannot load the file 
C:\.\TaxoNet\venv\Scripts\Activate.ps1 because script execution is disabled on this system.", it means that your current script execution policy is blocking scripts by default for security reason. To fix this issue, type in a Powershell terminal: 

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then activate your virtual environment.

5. **Install Dependencies**:

Type in your terminal:

```bash
pip install -r requirements.txt
``` 

This installs all the packages listed in the file into the virtual environment. This can take several minutes. 

You can confirm it worked by typing : 

```bash
pip list
```

6. **(Optional) Log into Weights & Biases**
```bash
wandb login
``` 
Then follow instructions. 


## Usages

This chapter goes through all the different usages that a user may have with the pipeline. The system is designed around YAML configuration files that make it easy to reproduce experiments and manage different setups.

### To preprocess your data

The preprocessing system supports three different data input formats. Choose the configuration that matches your data organization:

#### 1. Hierarchical Data Format (Folder-based)
If your data is organized in folders where each folder name represents a class:

```bash
python3 run_preprocessing.py --config configs/preprocessing/PreprocessWithHierarchical.yaml
```

Example configuration (`configs/preprocessing/simple_hierarchical.yaml`):
```yaml
# Simple Hierarchical Dataset Preprocessing Configuration
input_source:
  type: "hierarchical"
  data_path: "DATA/your_hierarchical_dataset"
  subdirs: []  # Empty = direct class folders under data_path

preprocessing:
  scalebar_removal:
    enabled: true
    model_path: "models/model_weights/scale_bar_remover/best.pt"
    confidence: 0.4
    img_size: 416
  
  grayscale_conversion:
    enabled: true
    mode: "RGB"
  
  image_filtering:
    min_images_per_class: 100
    max_images_per_class: 4000
    skip_corrupted_images: true
  
  data_splitting:
    train_ratio: 0.7
    val_ratio: 0.2
    test_ratio: 0.1
    stratified: true
    random_seed: 42
  
  augmentation:
    enabled: true
    target_images_per_class: 1500
    max_copies_per_image: 10
    techniques:
      horizontal_flip: 0.5
      vertical_flip: 0.2
      rotate_90: 0.3
      brightness_contrast: 0.4
      hue_saturation: 0.3
      
output:
  base_path: "DATA/your_dataset"
  processed_path: "DATA/your_dataset_processed"
  final_dataset_path: "DATA/your_dataset_final"
  create_dataset_yaml: true
  
logging:
  wandb_enabled: false   # In most cases, you don't need WandB logging at the preprocessing step
  log_class_distribution: true  
  log_sample_images: false
  log_processing_times: true
```

#### 2. CSV/TSV Mapping Format
If you have a CSV/TSV file mapping image paths to class labels:

```bash
python3 run_preprocessing.py --config configs/preprocessing/PreprocessWithCSV-TSV.yaml
```

Example configuration:
```yaml
# CSV Mapping Dataset Preprocessing Configuration
input_source:
  type: "csv_mapping"
  images_path: "DATA/your_images_directory"
  metadata_file: "DATA/your_labels.csv"
  image_column: "filename"
  label_column: "species"
  separator: ","
  image_path_prefix: ""

preprocessing:
  # Same preprocessing parameters over all configs 
 [...]
      
output:
  base_path: "DATA/your_dataset"
  processed_path: "DATA/your_dataset_processed"
  final_dataset_path: "DATA/your_dataset_final"
  create_dataset_yaml: true
  
logging:
  wandb_enabled: false
  log_class_distribution: true
  log_sample_images: false
  log_processing_times: true
```

#### 3. EcoTaxa Format
For data exported from EcoTaxa platform:

```bash
python3 run_preprocessing.py --config configs/preprocessing/PreprocessWithEcotaxa.yaml
```

Example configuration:
```yaml
# EcoTaxa TSV Preprocessing Configuration
input_source:
  type: "ecotaxa"
  data_path: "DATA/your_ecotaxa_folder"
  metadata_file: "ecotaxa_export_TSV_xxxxx.tsv"
  separator: "\t"

preprocessing:
  # Same preprocessing parameters over all configs 
  [...]

output:
  base_path: "DATA/your_ecotaxa"
  processed_path: "DATA/your_ecotaxa_processed"
  final_dataset_path: "DATA/your_ecotaxa_final"
  create_dataset_yaml: true
  
logging:
  wandb_enabled: false
  log_class_distribution: true
  log_sample_images: false
  log_processing_times: true
```

### To train a model

Training is fully configuration-driven. You can train different model architectures with various hyperparameters:

```bash
python3 run_training.py --config configs/training/TrainDefault{modelName}.yaml
```

#### Basic Training Example

Example configuration (`configs/training/TrainDefaultEfficientNet.yaml`):
```yaml
# ============================================================
# EfficientNet B5 Training Configuration
# ============================================================

run_name: "efficientnet_b5_experiment"  # Custom run name in case you don't want it to be generated

# DATA CONFIGURATION
data:
  dataset_path: "DATA/your_dataset_final"  # Path to preprocessed dataset
  
# PROJECT CONFIGURATION
project:
  name: "YOLOv11Classification500"     # W&B project name
  
# MODEL CONFIGURATION
model:
  name: "efficientnet"      # Options: "efficientnet", "resnet", "densenet", "yolov11"
  variant: "b5"             # EfficientNet variants: "b0"-"b7"
  pretrained: true          # Use pretrained weights
  freeze_backbone: false    # Freeze backbone layers
  input_size: 224           # Input image size
  num_classes: 76           # Number of output classes
  
# TRAINING CONFIGURATION
training:
  batch_size: 32               # Training batch size
  learning_rate: 0.001         # Initial learning rate
  weight_decay: 0.01           # Weight decay (L2 regularization) 
  epochs: 50                   # Number of training epochs
  optimizer: "adamw"           # Options: "adam", "adamw", "sgd", "rmsprop"
  early_stopping_patience: 15  # Early stopping patience
  device: "cuda"               # Options: "cuda", "cpu", "auto"
  num_workers: 8               # Number of data loader workers

# LOSS CONFIGURATION
loss:
  type: "focal"                # Options: "focal", "labelsmoothing", "weighted"
  focal_alpha: 1.0             # Focal loss alpha parameter
  focal_gamma: 2.0             # Focal loss gamma parameter
  use_per_class_alpha: true    # Use per-class weights

# WEIGHTS & BIASES CONFIGURATION
wandb:
  log_results: true            # Enable W&B logging
  tags: ["efficientnet", "b5", "production"]
  notes: "Production training run"
```

Training will always compute metrics on the test set at the end. 

#### Advanced Training Features

**Multiple Model Architectures:**
- **EfficientNet**: `variant: "b0"` to `"b7"`
- **ResNet**: `variant: "18"`, `"34"`, `"50"`, `"101"`, `"152"`
- **DenseNet**: `variant: "121"`, `"161"`, `"169"`, `"201"`
- **YOLOv11**: `name: "yolov11"`

**Loss Functions:**
```yaml
# Focal Loss (good for imbalanced datasets)
loss:
  type: "focal"
  focal_gamma: 2.0
  focal_alpha: 1.0
  use_per_class_alpha: true

# Label Smoothing
loss:
  type: "labelsmoothing"
  labelsmoothing_epsilon: 0.12

# Weighted Cross-Entropy (automatic class balancing)
loss:
  type: "weighted"

```

**Training Without Weights & Biases:**
```yaml
# Disable W&B for offline training
wandb:
  log_results: false
  tags: ["production", "efficientnet", "b5", "label_smoothing"]
  notes: "Production EfficientNet B5 with label smoothing"

run_name: "offline_experiment"

# All metrics and plots will be saved locally to:
# model_weights/{model_name}/{variant}/{run_name}/
```

### Inference (Making predictions)

Use trained models to make predictions on new images:

```bash
python3 run_inference.py --config configs/inference/your_inference_config.yaml
```


Example configuration (`configs/inference/DefaultInference.yaml`):
```yaml
# Model Configuration
model:
  name: "efficientnet"      # Model architecture: "efficientnet", "resnet", "densenet", "yolov11"
  variant: "b5"             # Model variant
  weights_path: "model_weights/efficientnet/b5/my_experiment/best.pt"
  num_classes: 76           # Number of classes (must match training)

# Dataset YAML for Class Names
dataset_yaml: "DATA/final_dataset/dataset.yaml"  # <-- Path to the YAML file containing class names
  # This file is generated automatically during preprocessing.
  # It MUST match the dataset used for training this model.
  # The class names will be read from the 'names' field in this YAML.

# Inference Configuration
inference:
  image_dir: "path/to/new/images"        # Directory containing images to predict
  batch_size: 32                        # Inference batch size
  top_k: 5                              # Return top K predictions per image
  device: "cuda"                        # Device: "cuda", "cpu", "auto"
  save_csv: true                        # Save results as CSV
  output_path: "outputs/predictions.csv" # Output file path

# Optional preprocessing during inference
preprocessing:
  scalebar_removal: true                # Apply scalebar removal if needed
  
# Weights & Biases Configuration
wandb:
  log_results: false                    # Usually disabled for inference
  tags: ["inference", "production"]
  notes: "Production inference run"
```


**How class names are handled:**
- The `dataset_yaml` field in your inference config should point to the YAML file generated during preprocessing (e.g., `DATA/final_dataset/dataset.yaml`).
- The system will read the class names from the `names` field in this YAML, ensuring the class order matches what was used during training.
- This prevents label mismatches and makes inference robust to changes in the dataset structure.

**Inference Output:**
- CSV file with detailed predictions and confidence scores (specified in `output_path`)
- Predictions include top-K classes with probabilities for each image
- Optional scalebar preprocessing applied automatically if enabled

### Hyperparameter Optimization

Perform automated hyperparameter sweeps using Weights & Biases:

```bash
python3 run_sweep.py --sweep_config configs/sweeps/densenet_sweep.yaml
```

Example sweep configuration (`configs/sweeps/densenet_sweep.yaml`):
```yaml
# Sweep configuration
program: run_sweep.py
method: bayes  # or random, grid
metric:
  name: val/metrics/accuracy_top1
  goal: maximize

# Fixed parameters
parameters:
  model_name:
    value: densenet
  epochs:
    value: 30
  device:
    value: cuda

  # Parameters to optimize
  densenet_variant:
    values: [121, 161, 169, 201]
  
  batch_size:
    values: [32, 64]
    
  learning_rate:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-3
    
  loss_type:
    values: ["focal", "labelsmoothing", "weighted"]
    
  # Focal loss parameters (when applicable)
  focal_gamma:
    distribution: uniform
    min: 1.0
    max: 3.0
```

**Running Sweeps:**
1. The sweep automatically creates multiple training runs
2. Each run tests different hyperparameter combinations
3. Results are logged to Weights & Biases for comparison
4. Best configurations are automatically identified
5. When you are satisfied with the results, you can just stop the sweep in the terminal

### Quick Start Examples

**Complete Workflow Example:**
```bash
# 1. Preprocess your data
python3 run_preprocessing.py --config configs/preprocessing/PreprocessWithHierarchical.yaml

# 2. Train a model
python3 run_training.py --config configs/training/TrainDefaultEfficientNet.yaml

# 3. Make predictions
python3 run_inference.py --config configs/inference/DefaultInference.yaml
```

## Results and monitoring

### Monitoring 

If you choose not to use Weights & Biases, our custom monitoring module will collect training data and write it to a log file in real time, allowing you to monitor your runs as they progress.

The file will be saved at: `model_weights/{model_name}/{variant}/{run_name}/training_log.txt`. 

Example: 

```yaml
Training Log - Reproduce_best_model_20250814_141638
============================================================
Model: efficientnet b5
Dataset: DATA/final_dataset
Batch Size: 64
Learning Rate: 1e-5
Loss Type: labelsmoothing
Started: 2025-08-14 14:16:39
============================================================
Epoch | Train Loss | Val Loss | Top-1 Acc | Top-5 Acc | F1 Macro | Recall Macro | Precision Macro | LR
---------------------------------------------------------------------------------------------------------
    1 |     2.7936 |   1.6791 |    0.7390 |    0.9518 |   0.6957 |       0.7385 |          0.7072 | 1.00e-05
    2 |     1.5423 |   1.3701 |    0.8247 |    0.9806 |   0.7969 |       0.8204 |          0.7892 | 1.00e-05
    3 |     1.3472 |   1.2886 |    0.8498 |    0.9877 |   0.8246 |       0.8387 |          0.8202 | 9.99e-06
    4 |     1.2635 |   1.2551 |    0.8603 |    0.9892 |   0.8375 |       0.8534 |          0.8298 | 9.98e-06
```

This same data is also stored in `training_log.csv` for further analysis.

### Results

A notebook named `results_analysis.ipynb` is available to further analyze training metrics and inference results.
To adapt it to your own runs, modify the first cell as follows: 

```python
# Replace with your model path
MODEL_DIR = "model_weights/{model_name}/{variant}/{run_name}"
TRAINING_LOG_PATH = f"{MODEL_DIR}/training_log.csv"
CLASSIF_REPORT = f"{MODEL_DIR}/classification_report.csv"
```

Once updated, click on Run All to generate figures and insights about your classification model and training process, including:

- Validation & training loss over epochs
- Evolution of accuracies over epochs
- Metrics over epochs
- F1-score vs. support
- Best and worst classified classes
- Confusion matrix
- Most frequently confused classes

This is a starter kit to evaluate your model, which can be easily extended for more advanced analyses.

## Contributing

We welcome all pull requests — from small fixes to big new features.  
If you’d like to help improve this project, please check out our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and the contribution workflow.


## Citation

If you use this project in your research, please cite it as:

```bibtex
@software{walter2025pipeline,
  author = {Hugo WALTER},
  title = {Deep Learning Classification Pipeline for Automatic Plankton Classification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ziraax/TaxoNet}
}
```

## License

This project is open source and distributed under the [MIT License](./LICENSE.md).  
See the `LICENSE.md` file for details.







