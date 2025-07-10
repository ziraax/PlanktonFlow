# Deep Learning Training & Inference Pipeline for Automatic Plankton classification

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![W&B Integration](https://img.shields.io/badge/Weights_&_Biases-Integrated-yellow)

An end-to-end deep learning solution supporting multiple model architectures with advanced features for training, evaluation, and production-ready inference.

## Features

### Preprocessing 
Tailored for our dataset — customizable for yours.

- **Data Augmentation**: Allows to augment the training data to limit the class imbalance issue
- **Scalebar Removal**: Allows to detect and delete scale bars that were originally present in some images using a YOLOv8 model. 


### Training 
- **Multi-model Support**: YOLOv11, ResNet, DenseNet, EfficientNet
- **Advanced Training**:
  - Configurable hyperparameters
  - Early stopping with checkpointing
  - Class-weighted loss functions
  - More features
- **Model Factory Pattern**: Dynamic model creation with variants
- **W&B Integration**: Real-time tracking of metrics, artifacts, and model versions

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

If you are not familiar with Git, you can simply click the green "<> Code" button and click on "Download ZIP". Then, simply extract the project where you want it to be on your computer. 

3. **Create a virtual environment**:

Virtual environments in Python are isolated directories that contain their own Python interpreter and libraries, allowing you to manage dependencies for each project separately. This prevents conflicts between packages required by different projects and ensures reproducible setups. This comes handy on project like this one where there a lot of dependencies. 

To create a virtual environment, write in your terminal :

```bash
python -m venv .venv
```

Where **.venv** will be the name of the folder holding the virtual environment. 


3. **Activate the Environment**:

Open a terminal in the project folder. Now, depending on your Operating System : 

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
source .venv/Scripts/activate
```

After activation, your terminal will change to show the venv name. 

⚠️ In case you run into a bug like "Cannot load the file 
C:\.\TaxoNet\venv\Scripts\Activate.ps1 because 
script execution is disabled on this system.", it means that your current script execution policy is blocking scripts by default for security reason. To fix this issue, type in a Powershell terminal: 

```bash
Set-ExecutionPolicy -Scope Process-ExecutionPolicy Bypass
```
Then activate your virtual environment.

2. **Install Dependencies**:

Type in your terminal:

```bash
pip install -r requirements.txt
``` 

This installs all the packages listed in the file into the virtual environment. This can take several minutes. 

You can confirm it worked by typing : 

```bash
pip list
```

3. **(Optional) Log into Weights & Biases**
```bash
wandb login
``` 
Then follow instructions. 

We recommend enabling WandB logging for experiment tracking. Weights & Biases includes a generous free tier.

## Usages

### To preprocess your data

TODO

### To train a model :

Your dataset should be organized in a directory structure compatible with common deep learning frameworks. Each class should have its own subfolder containing all images belonging to that class. For example:

```
dataset/
  classA/
    img1.jpg
    img2.jpg
    ...
  classB/
    img1.jpg
    img2.jpg
    ...
  classC/
    img1.jpg
    ...
```

This structure allows the pipeline to automatically infer class labels based on folder names and efficiently load images for training and inference.

If you already have your dataset split into `train`, `val`, and `test` directories and your dataset is already preprocessed, you can skip directly to the training step. The pipeline fully supports this structure, and will automatically use the provided splits for training, validation, and testing. Your directory should look like:

```
dataset/
  train/
    classA/
      img1.jpg
      ...
    classB/
      ...
  val/
    classA/
      ...
    classB/
      ...
  test/
    classA/
      ...
    classB/
      ...
```

Then modify in **config.py** the path to your own dataset. TODO

### Inference (Making predictions)

To do inference using the best model from our experimentations, you can directly do: 

```python
python -m inference.predict \
  --image_dir PATH/TO/YOUR/IMAGES \
  --model_name best_model
  --weights_path best_model
  --batch_size 64
  --top_k 3
  --device [cpu, cuda]
  --save_csv PATH/TO/SAVE/CSV
  --wandb_log
```

In case you have trained your own model, you want to use the inference pipeline with specified model parameters.


```bash
python -m inference.predict \
  --image_dir PATH/TO/YOUR/IMAGES \
  --model_name [resnet, densenet, efficientnet, yolov11] \
  --weights_path model_weights/[model_type]/[model_variant]/[run] \
  --densenet_variant 121 \
  --resnet_variant
  --efficientnet_variant 
  --save_csv predictions.csv \
  --wandb_log
```

An exemple could be : 

```bash
python -m inference.predict \
  --image_dir DATA/ecotaxa_infer_set 
  --model_name densenet 
  --densenet_variant 121 
  --weights_path model_weights/densenet/121/leafy-sweep-26/best.pt 
  --batch_size 64 
  --top_k 3 
  --device cuda 
  --save_csv outputs/res_cool.csv 
  --wandb_log
``` 

| Flag             | Description                       |
| ---------------- | --------------------------------- |
| `--image_dir`    | Folder containing images          |
| `--model_name`   | Model architecture                |
| `--weights_path` | Path to trained weights           |
| `--top_k`        | Number of top predictions to keep |
| `--device`       | `"cpu"` or `"cuda"` depending on your computer (use `"cuda"` if you have a modern GPU)              |
| `--save_csv`     | Path to save predictions as CSV   |
| `--wandb_log`    | Enable experiment logging to W\&B |


## Model Factory

Supported architectures and variants:

| Model        | Variants                  |
|--------------|---------------------------|
| YOLOv11      | Classification            |   
| DenseNet     | 121, 169, 201, 161        |
| EfficientNet | b0-b5                     |
| ResNet       | 18, 34, 50, 101, 152      |   


## Configuration

TODO

## Results and monitoring 

TODO

## Contributing 

TODO

## License

TODO





