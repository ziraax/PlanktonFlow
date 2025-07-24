import sys
import os    
import torch
import torch.nn.functional as F
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
from utils.input_size import get_input_size_for_model


def run_inference(args, DEFAULT_CONFIG):
    """
    Run inference on a set of images using a specified model.
    Args:
        args: Command line arguments containing model and data parameters.
        CONFIG: Configuration dictionary containing model settings.
    Returns:
        all_results: List of dictionaries containing image paths and predictions.

    Supports:
        - Batch inference for multiple images.
    """

    # We want to store the preprocessed images in a temp dire that will be cleaned up after inference
    with tempfile.TemporaryDirectory(prefix="inference_processed_") as temp_dir:
        
        orig_image_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        temp_processed_dir = Path(temp_dir)

        print(f"[INFO] Temp folder is located at: {temp_processed_dir}")

        scalebar_model = YOLO(DEFAULT_CONFIG['scalebar_model_path'])
        scalebar_model.conf = DEFAULT_CONFIG['scalebar_confidence']

        processed_to_original = {}
        processed_image_paths = []
        for orig_path in orig_image_paths:
            process_image(orig_path, temp_processed_dir, scalebar_model)

            with open(orig_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:8]
            processed_path = temp_processed_dir / f"{Path(orig_path).stem}_{file_hash}.jpg"
            if processed_path.exists():
                processed_image_paths.append(str(processed_path))
                processed_to_original[str(processed_path)] = orig_path
            else:
                print(f"[WARNING] Processed image not found: {processed_path}. Skipping this image.")

        
        image_paths = processed_image_paths
            
        class_names = get_classes_names()
        print("[INFO] Number of classes:", len(class_names))
        DEFAULT_CONFIG['num_classes'] = len(class_names) if class_names else 76
        # Get the correct variant directly from args
        if args.model_name == 'densenet':
            variant = args.densenet_variant
        elif args.model_name == 'resnet':
            variant = args.resnet_variant
        elif args.model_name == 'efficientnet':
            variant = args.efficientnet_variant
        else:
            variant = "Default-cls"

        img_size = 224
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
                num_classes=DEFAULT_CONFIG['num_classes'],
                pretrained=False,
                efficientnet_variant=args.efficientnet_variant,
                densenet_variant=args.densenet_variant,
                resnet_variant=args.resnet_variant,
                mc_dropout=False,
                mc_p=False        
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
