"""
Config-driven preprocessing pipeline
Supports multiple input formats and flexible preprocessing steps
"""

import time
import wandb
import yaml
import os
import shutil
import tempfile
import hashlib
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Local imports
from preprocessing.data_input_handlers import create_data_handler
from preprocessing.scalebar_removal import process_image
from preprocessing.augmentor import augment_image_task
from utils.config_manager import config_manager
from utils.check_imgs import check_images
from utils.logs import log_class_distribution, log_class_distribution_comparison_before_after_aug, log_class_weights
from utils.classes import get_class_distribution, get_class_weights, get_classes_names
from utils.pytorch_cuda import check_pytorch_cuda
from ultralytics import YOLO


class ConfigPreprocessor:
    """Main preprocessing pipeline driven by configuration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_config = config['input_source']
        self.preprocessing_config = config['preprocessing']
        self.output_config = config['output']
        self.logging_config = config['logging']
        
        # Initialize paths
        self.base_path = Path(self.output_config['base_path'])
        self.processed_path = Path(self.output_config['processed_path'])
        self.final_dataset_path = Path(self.output_config['final_dataset_path'])
        
        # Create output directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.final_dataset_path.mkdir(parents=True, exist_ok=True)
        
    def run_full_pipeline(self):
        """Execute the complete preprocessing pipeline"""
        print("[INFO] Starting config-driven preprocessing pipeline...")
        
        # Initialize W&B if enabled
        if self.logging_config.get('wandb_enabled', False):
            if wandb.run is None:
                print("[INFO] WandB is not initialized. Please run 'wandb.init()' before calling this function.")
                return
        
        start_time = time.time()
        
        try:
            # Step 1: Load and validate input data
            image_label_pairs = self._load_input_data()
            if not image_label_pairs:
                print("[ERROR] No valid image-label pairs found. Aborting preprocessing.")
                return
            
            # Step 2: Process images (scalebar removal, grayscale conversion)
            processed_pairs = self._process_images(image_label_pairs)
            
            # Step 3: Filter classes and organize by class
            class_images = self._organize_by_class(processed_pairs)
            
            # Step 4: Create train/val/test splits
            split_data = self._create_splits(class_images)
            
            # Step 5: Save split data to directory structure
            self._save_split_structure(split_data)
            
            # Step 6: Apply augmentation if enabled
            if self.preprocessing_config['augmentation']['enabled']:
                self._apply_augmentation()
            
            # Step 7: Create dataset.yaml if requested
            if self.output_config.get('create_dataset_yaml', True):
                self._create_dataset_yaml()
            
            # Step 8: Final validation and logging
            self._final_validation_and_logging()
            
            duration = time.time() - start_time
            print(f"[INFO] Config-driven preprocessing completed in {duration:.2f} seconds.")
            
            if self.logging_config.get('wandb_enabled', False):
                wandb.log({"config_preprocessing_time_sec": duration})
                
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            raise
    
    def _load_input_data(self) -> List[Tuple[str, str]]:
        """Load image-label pairs using appropriate data handler"""
        print("[INFO] Loading input data...")
        
        # Create appropriate data handler
        data_handler = create_data_handler(self.input_config)
        
        # Validate input
        if not data_handler.validate_input():
            print("[ERROR] Input validation failed")
            return []
        
        # Get image-label pairs
        image_label_pairs = data_handler.get_image_label_pairs()
        print(f"[INFO] Loaded {len(image_label_pairs)} image-label pairs")
        
        return image_label_pairs
    
    def _process_images(self, image_label_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Process images with scalebar removal and grayscale conversion"""
        print("[INFO] Processing images...")
        
        processed_pairs = []
        scalebar_config = self.preprocessing_config.get('scalebar_removal', {})
        grayscale_config = self.preprocessing_config.get('grayscale_conversion', {})
        
        # Load scalebar model if enabled
        scalebar_model = None
        if scalebar_config.get('enabled', False):
            model_path = scalebar_config['model_path']
            if os.path.exists(model_path):
                scalebar_model = YOLO(model_path)
                scalebar_model.conf = scalebar_config.get('confidence', 0.4)
                print(f"[INFO] Loaded scalebar removal model: {model_path}")
            else:
                print(f"[WARNING] Scalebar model not found: {model_path}. Skipping scalebar removal.")
        
        error_count = 0
        
        for img_path, label in image_label_pairs:
            try:
                # Create output directory for this class
                class_output_dir = self.processed_path / label
                class_output_dir.mkdir(exist_ok=True)
                
                # Generate unique filename
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()[:8]
                
                output_filename = f"{Path(img_path).stem}_{file_hash}.jpg"
                output_path = class_output_dir / output_filename
                
                # Skip if already processed
                if output_path.exists():
                    processed_pairs.append((str(output_path), label))
                    continue
                
                # Process image
                if scalebar_model:
                    # Use existing process_image function with temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_dir_path = Path(temp_dir)
                        process_image(img_path, temp_dir_path, scalebar_model)
                        
                        # Find processed image
                        temp_processed = temp_dir_path / f"{Path(img_path).stem}_{file_hash}.jpg"
                        if temp_processed.exists():
                            shutil.copy(temp_processed, output_path)
                        else:
                            # Fallback: copy original
                            shutil.copy(img_path, output_path)
                else:
                    # No scalebar removal, just copy
                    shutil.copy(img_path, output_path)
                
                # Apply grayscale conversion if enabled
                if grayscale_config.get('enabled', False):
                    self._apply_grayscale_conversion(output_path, grayscale_config)
                
                processed_pairs.append((str(output_path), label))
                
            except Exception as e:
                error_count += 1
                print(f"[ERROR] Failed to process {img_path}: {e}")
                continue
        
        print(f"[INFO] Processed {len(processed_pairs)} images ({error_count} failures)")
        return processed_pairs
    
    def _apply_grayscale_conversion(self, image_path: str, grayscale_config: Dict[str, Any]):
        """Apply grayscale conversion to image"""
        try:
            from PIL import Image
            import cv2
            import numpy as np
            
            img = Image.open(image_path).convert("RGB")
            
            mode = grayscale_config.get('mode', 'RGB')
            
            if mode == "L":
                processed_img = img.convert("L")
            else:  # mode == "RGB"
                # Convert to grayscale but keep as RGB
                img_np = np.array(img)
                gray_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                rgb_gray_np = cv2.cvtColor(gray_np, cv2.COLOR_GRAY2RGB)
                processed_img = Image.fromarray(rgb_gray_np)
            
            # Save back to same path
            processed_img.save(image_path, quality=95, optimize=True)
            
        except Exception as e:
            print(f"[WARNING] Failed to apply grayscale conversion to {image_path}: {e}")
    
    def _organize_by_class(self, processed_pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """Organize processed images by class and apply filtering"""
        print("[INFO] Organizing images by class...")
        
        filtering_config = self.preprocessing_config.get('image_filtering', {})
        min_images = filtering_config.get('min_images_per_class', 10)
        max_images = filtering_config.get('max_images_per_class', None)
        
        # Group by class
        class_images = defaultdict(list)
        for img_path, label in processed_pairs:
            class_images[label].append(img_path)
        
        # Apply filtering
        filtered_class_images = {}
        skipped_classes = []
        
        for class_name, images in class_images.items():
            if len(images) < min_images:
                skipped_classes.append(f"{class_name}: {len(images)} images")
                continue
            
            # Apply max images limit if specified
            if max_images and len(images) > max_images:
                # Randomly sample max_images
                random.seed(self.preprocessing_config['data_splitting'].get('random_seed', 42))
                images = random.sample(images, max_images)
                print(f"[INFO] Class '{class_name}' downsampled from {len(class_images[class_name])} to {max_images} images")
            
            filtered_class_images[class_name] = images
        
        # Log skipped classes
        if skipped_classes:
            print(f"[WARNING] Skipped {len(skipped_classes)} classes with insufficient images:")
            for skipped in skipped_classes:
                print(f"  - {skipped}")
        
        print(f"[INFO] Organized {len(filtered_class_images)} valid classes")
        return filtered_class_images
    
    def _create_splits(self, class_images: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
        """Create train/val/test splits"""
        print("[INFO] Creating data splits...")
        
        split_config = self.preprocessing_config['data_splitting']
        train_ratio = split_config['train_ratio']
        val_ratio = split_config['val_ratio']
        test_ratio = split_config['test_ratio']
        random_seed = split_config.get('random_seed', 42)
        stratified = split_config.get('stratified', True)
        
        # Create full image list with labels
        all_images = [(img_path, class_name) for class_name, images in class_images.items() for img_path in images]
        
        if stratified:
            # Stratified split
            train_data, temp_data = train_test_split(
                all_images,
                test_size=1 - train_ratio,
                stratify=[x[1] for x in all_images],
                random_state=random_seed
            )
            
            val_data, test_data = train_test_split(
                temp_data,
                test_size=test_ratio / (val_ratio + test_ratio),
                stratify=[x[1] for x in temp_data],
                random_state=random_seed
            )
        else:
            # Random split
            train_data, temp_data = train_test_split(
                all_images,
                test_size=1 - train_ratio,
                random_state=random_seed
            )
            
            val_data, test_data = train_test_split(
                temp_data,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=random_seed
            )
        
        split_data = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split, data in split_data.items():
            print(f"[INFO] {split}: {len(data)} images")
        
        return split_data
    
    def _save_split_structure(self, split_data: Dict[str, List[Tuple[str, str]]]):
        """Save split data to directory structure"""
        print("[INFO] Saving split structure...")
        
        for split, data in split_data.items():
            split_dir = self.final_dataset_path / split
            split_dir.mkdir(exist_ok=True)
            
            for img_path, class_name in data:
                class_dir = split_dir / class_name
                class_dir.mkdir(exist_ok=True)
                
                src_path = Path(img_path)
                # Don't add split prefix - folder structure is sufficient
                dest_path = class_dir / src_path.name
                
                if not dest_path.exists() and src_path.exists():
                    shutil.copy(src_path, dest_path)
    
    def _apply_augmentation(self):
        """Apply data augmentation to training set using config-driven approach"""
        print("[INFO] Applying data augmentation...")
        
        aug_config = self.preprocessing_config['augmentation']
        target_images = aug_config.get('target_images_per_class', 1000)
        max_copies = aug_config.get('max_copies_per_image', 10)
        techniques = aug_config.get('techniques', {})
        
        # Apply config-driven augmentation
        self._apply_config_augmentation(target_images, max_copies, techniques)
    
    def _apply_config_augmentation(self, target_images_per_class, max_copies_per_image, techniques):
        """Config-driven augmentation implementation"""
        import math
        import concurrent.futures
        import albumentations as A
        import numpy as np
        from PIL import Image
        import hashlib
        import random
        import torch
        from tqdm import tqdm
        
        train_path = self.final_dataset_path / 'train'
        
        # Count current images per class
        class_counts = {}
        for class_dir in train_path.iterdir():
            if class_dir.is_dir():
                image_count = len([f for f in class_dir.glob('*.jpg')])
                class_counts[class_dir.name] = image_count
        
        tasks = []
        
        for class_name, count in class_counts.items():
            if count < target_images_per_class:
                class_dir = train_path / class_name
                original_images = sorted([
                    f for f in class_dir.glob('*.jpg')
                    if not f.name.startswith('aug_')
                ])
                
                needed = target_images_per_class - count
                num_originals = len(original_images)
                
                if num_originals == 0:
                    print(f"[WARNING] Skipping {class_name} — no original images")
                    continue
                
                per_image = max(1, math.ceil(needed / num_originals))
                if per_image > max_copies_per_image:
                    print(f"[WARN] Class '{class_name}' cannot reach {target_images_per_class} "
                          f"with only {num_originals} originals and multiplier={max_copies_per_image}.")
                    per_image = max_copies_per_image
                
                print(f"[INFO] Class {class_name}: {count} -> {target_images_per_class} | "
                      f"{per_image} augmentations per image")
                
                for idx, img_path in enumerate(original_images):
                    tasks.append((str(img_path), str(class_dir), idx, per_image, img_path.name, techniques))
        
        if not tasks:
            print("[INFO] No augmentation needed - all classes meet target threshold")
            return
        
        # Parallel processing
        total_augmented = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in tqdm(executor.map(self._augment_image_task_config, tasks), 
                             total=len(tasks), desc="Augmenting images"):
                total_augmented += result
        
        print(f"[INFO] Augmentation complete. Total augmented images: {total_augmented}")
        
        if self.logging_config.get('wandb_enabled', False):
            wandb.log({"total_augmented_images": total_augmented})
    
    @staticmethod
    def _augment_image_task_config(args):
        """Config-driven augmentation task for multiprocessing"""
        import albumentations as A
        import numpy as np
        from PIL import Image
        import hashlib
        import random
        import torch
        import os
        
        img_path, class_dir, idx, per_image, img_name, techniques = args
        
        # Build augmentation pipeline from config
        aug_list = []
        if techniques.get('horizontal_flip', 0) > 0:
            aug_list.append(A.HorizontalFlip(p=techniques['horizontal_flip']))
        if techniques.get('vertical_flip', 0) > 0:
            aug_list.append(A.VerticalFlip(p=techniques['vertical_flip']))
        if techniques.get('rotate_90', 0) > 0:
            aug_list.append(A.RandomRotate90(p=techniques['rotate_90']))
        if techniques.get('brightness_contrast', 0) > 0:
            aug_list.append(A.RandomBrightnessContrast(p=techniques['brightness_contrast']))
        if techniques.get('hue_saturation', 0) > 0:
            aug_list.append(A.HueSaturationValue(p=techniques['hue_saturation']))
        
        if not aug_list:
            return 0
            
        pipeline = A.Compose(aug_list)
        
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"[ERROR] Could not load image {img_path}: {str(e)}")
            return 0
        
        successful = 0
        
        for copy_num in range(per_image):
            seed = int(hashlib.sha256(img_name.encode()).hexdigest()[:8], 16) + copy_num
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            try:
                augmented = pipeline(image=image)['image']
                aug_img = Image.fromarray(augmented)
                aug_path = os.path.join(class_dir, f"aug_{idx}_{copy_num}_{img_name}")
                aug_img.save(aug_path)
                successful += 1
            except Exception as e:
                print(f"[ERROR] Augmentation failed for {img_path}: {str(e)}")
        
        return successful
    
    def _create_dataset_yaml(self):
        """Create dataset.yaml file"""
        print("[INFO] Creating dataset.yaml...")
        
        # Get class names from final dataset
        train_dir = self.final_dataset_path / 'train'
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        
        dataset_yaml = {
            'path': str(self.final_dataset_path),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = self.final_dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        print(f"[INFO] dataset.yaml created at: {yaml_path}")
    
    def _final_validation_and_logging(self):
        """Final validation and logging"""
        print("[INFO] Performing final validation...")
        
        # Check images
        if self.preprocessing_config['image_filtering'].get('skip_corrupted_images', True):
            check_images(str(self.final_dataset_path), delete=False)
        
        # Log to W&B if enabled
        if self.logging_config.get('wandb_enabled', False):
            if self.logging_config.get('log_class_distribution', True):
                # TODO: This would require adapting the existing logging functions
                print("[INFO] Class distribution logging would be applied here")
                
            if self.logging_config.get('log_sample_images', True):
                # TODO: Sample image logging would be applied here
                print("[INFO] Sample image logging would be applied here")


def run_config_preprocessing(config_name: str):
    """Run preprocessing with specified configuration"""
    
    # Load configuration
    config = config_manager.get_preprocessing_config(config_name)
    
    # Initialize W&B if enabled
    if config['logging'].get('wandb_enabled', False):
        wandb.init(
            project=config.get('project_name', 'preprocessing'),
            name=f"preprocessing_{config_name}",
            job_type="preprocessing",
            config=config
        )
    
    # Run preprocessing
    preprocessor = ConfigPreprocessor(config)
    preprocessor.run_full_pipeline()
    
    # Finish W&B run
    if config['logging'].get('wandb_enabled', False):
        wandb.finish()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run config-driven preprocessing")
    parser.add_argument('--config', type=str, required=True, 
                       help='Name of preprocessing configuration (without .yaml extension)')
    
    args = parser.parse_args()
    run_config_preprocessing(args.config)
