import sys
import os    
import torch
import torch.nn.functional as F
from preprocessing.scalebar_removal import process_image
from pathlib import Path
from tqdm import tqdm
import tempfile
import shutil 
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

        img_size = get_input_size_for_model(args.model_name, variant)
        all_results = []

        if args.model_name == "yolov11":
            # Load YOLOv11 classification model
            model = YOLO(args.weights_path, "classify")

            # Perform batch inference
            results = model(image_paths, imgsz=img_size, device=args.device)

            for path, result in zip(image_paths, results):
                top5_indices = result.probs.top5
                top5_confs = result.probs.top5conf
                preds = [(class_names[i], round(top5_confs[j].item(), 4)) for j, i in enumerate(top5_indices)]
                all_results.append({
                    'image_path': path,
                    'predictions': preds
                })

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


            model.load_state_dict(torch.load(args.weights_path, map_location="cuda"))
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
