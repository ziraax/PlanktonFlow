import sys
import os    
import torch
import torch.nn.functional as F

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.scalebar_removal import process_image
from pathlib import Path
from tqdm import tqdm
import tempfile
import shutil
import math
import hashlib
from ultralytics import YOLO
from torch.utils.data import DataLoader
from inference.dataset import InferenceDataset
from models.factory import create_model
from inference.utils import topk_predictions
from utils.classes import get_classes_names


def run_inference(args, config):
    """
    Run inference on a set of images using a specified model.
    Args:
        args: Command line arguments containing model and data parameters.
        config: Configuration dictionary containing model settings.
    Returns:
        all_results: List of dictionaries containing image paths and predictions.

    Supports:
        - Batch inference for multiple images.
    """

    # We want to store the preprocessed images in a temp dir that will be cleaned up after inference
    with tempfile.TemporaryDirectory(prefix="inference_processed_") as temp_dir:
        
        orig_image_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        temp_processed_dir = Path(temp_dir)

        print(f"[INFO] Temp folder is located at: {temp_processed_dir}")

        # Check if scalebar removal is enabled in config
        scalebar_enabled = config.get('preprocessing', {}).get('scalebar_removal', False)
        
        if scalebar_enabled:
            print("[INFO] Scalebar removal enabled - preprocessing images")
            scalebar_model = YOLO(config['scalebar_model_path'])
            scalebar_model.conf = config['scalebar_confidence']
        else:
            print("[INFO] Scalebar removal disabled - using original images")
            scalebar_model = None

        processed_to_original = {}
        processed_image_paths = []
        for orig_path in orig_image_paths:
            if scalebar_enabled:
                # Process with scalebar removal
                process_image(orig_path, temp_processed_dir, scalebar_model)
                with open(orig_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()[:8]
                processed_path = temp_processed_dir / f"{Path(orig_path).stem}_{file_hash}.jpg"
                if processed_path.exists():
                    processed_image_paths.append(str(processed_path))
                    processed_to_original[str(processed_path)] = orig_path
                else:
                    print(f"[WARNING] Processed image not found: {processed_path}. Skipping this image.")
            else:
                # Use original image without processing
                processed_image_paths.append(orig_path)
                processed_to_original[orig_path] = orig_path

        
        image_paths = processed_image_paths

        # --- Load class names from dataset_yaml if provided ---
        import yaml
        dataset_yaml_path = config.get('dataset_yaml', None)
        if dataset_yaml_path is not None and os.path.isfile(dataset_yaml_path):
            with open(dataset_yaml_path, 'r') as f:
                dataset_yaml = yaml.safe_load(f)
            class_names = dataset_yaml.get('names', [])
            if not class_names:
                print(f"[WARNING] No 'names' field found in {dataset_yaml_path}. Class names will be empty.")
        else:
            print(f"[WARNING] dataset_yaml not specified or file not found. Falling back to get_classes_names().")
            class_names = get_classes_names()
            print(f"[WARNING] Class names:", class_names)

        # Check if model config specifies num_classes, otherwise use detected classes
        model_config = config.get('model', {})
        num_classes = model_config.get('num_classes', len(class_names) if class_names else 76)
        print(f"[INFO] Using {num_classes} classes for model")
        
        # Get the correct variant directly from args
        if args.model_name == 'densenet':
            variant = args.densenet_variant
        elif args.model_name == 'resnet':
            variant = args.resnet_variant
        elif args.model_name == 'efficientnet':
            variant = args.efficientnet_variant
        else:
            variant = "Default-cls"

        # Use consistent 224x224 input size for all models
        img_size = 224
        print(f"[INFO] Using input size: {img_size}x{img_size} for {args.model_name} {variant}")
        all_results = []

        if args.model_name == "yolov11":
            model = YOLO(args.weights_path, task="classify")
            num_batches = math.ceil(len(image_paths) / args.batch_size)

            for i in tqdm(range(num_batches), desc="Inference", unit="batch"):
                batch_paths = image_paths[i * args.batch_size:(i + 1) * args.batch_size]

                # stream=True to avoid memory overflow
                results = model(batch_paths, imgsz=img_size, device=args.device, stream=True)

                for path, result in zip(batch_paths, results):
                    try:
                        probs = result.probs.data  # shape: (num_classes,)
                        topk = min(args.top_k, len(probs))  # safe-guard
                        topk_confs, topk_indices = probs.topk(topk)
                        preds = [(class_names[i], round(topk_confs[j].item(), 4)) for j, i in enumerate(topk_indices)]
                        all_results.append({
                            'image_path': processed_to_original.get(str(path), str(path)),
                            'predictions': preds
                        })
                    except Exception as e:
                        print(f"[ERROR] Failed inference on {path}: {e}")

        else:
            # Torchvision models
            dataset = InferenceDataset(image_paths, img_size)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            model = create_model(
                args.model_name,
                num_classes=num_classes,
                pretrained=False,
                efficientnet_variant=args.efficientnet_variant,
                densenet_variant=args.densenet_variant,
                resnet_variant=args.resnet_variant     
            )


            model.load_state_dict(torch.load(args.weights_path, map_location=args.device))
            model = model.to(args.device)
            model.eval()

            #---------------------------------------
            # Perform batch inference
            with torch.no_grad():
                for batch_images, batch_paths in tqdm(dataloader, desc="Inference", unit="batch"):
                    batch_images = batch_images.to(args.device)
                    logits = model(batch_images)  # (B, C)
                    probs = F.softmax(logits, dim=1)  # (B, C)

                    for i, path in enumerate(batch_paths):
                        topk_scores, topk_indices = probs[i].topk(args.top_k)
                        preds = []
                        for score, idx in zip(topk_scores, topk_indices):
                            class_name = class_names[idx]
                            preds.append((class_name, round(score.item(), 4)))
                        all_results.append({
                            'image_path': processed_to_original.get(str(path), str(path)),
                            'predictions': preds
                        })

        if args.save_csv:
            from inference.utils import save_results
            save_results(all_results, args.save_csv)

        return all_results
