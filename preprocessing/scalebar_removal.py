import time
import numpy as np
import hashlib
import cv2
import wandb
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from config import DEFAULT_CONFIG
from ultralytics import YOLO


def process_image(src_path, output_dir, model):
    """Process a single image with scale bar removal and optional grayscale conversion"""
    try:
        # Generate unique filename using content hash
        with open(src_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        dest_path = output_dir / f"{Path(src_path).stem}_{file_hash}.jpg"

        if dest_path.exists():
            return  # Skip if already processed

        # Load and orient image
        img = Image.open(src_path).convert("RGB")

        # Detect scale bars
        results = model.predict(img, imgsz=DEFAULT_CONFIG['scalebar_img_size'])
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            best_box = boxes[boxes.conf.argmax()]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            crop_y = min(y1, y2)
            img_np = np.array(img)
            cropped_np = img_np[:crop_y, :]
            processed_img = Image.fromarray(cropped_np)
        else:
            processed_img = img  # Keep original if no detection

        # Grayscale conversion
        if DEFAULT_CONFIG["convert_grayscale"]:
            if DEFAULT_CONFIG["grayscale_mode"] == "L":
                processed_img = processed_img.convert("L")
            else:
                processed_img = cv2.cvtColor(np.array(processed_img), cv2.COLOR_RGB2GRAY)
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
                processed_img = Image.fromarray(processed_img)

        # Save final image
        processed_img.save(dest_path, quality=95, optimize=True)

    except UnidentifiedImageError:
        print(f"[INVALID] Skipping corrupt or unreadable image: {src_path}")
    except Exception as e:
        print(f"[ERROR] Failed to process image: {src_path}")
        print(f"        Reason: {e}")
        # Uncomment below for full traceback if needed
        # traceback.print_exc()

def remove_scale_bars():
    """Main function to process all images in the raw dataset"""

    print("[INFO] Starting scale bar removal...")
    start_time = time.time()

    raw_root = Path(DEFAULT_CONFIG['raw_data_root'])
    processed_root = Path(DEFAULT_CONFIG['processed_path'])
    processed_root.mkdir(parents=True, exist_ok=True)

    # Load model once
    model = YOLO(DEFAULT_CONFIG['scalebar_model_path'])
    model.conf = DEFAULT_CONFIG['scalebar_confidence']

    image_paths = []
    for year in DEFAULT_CONFIG['years']:
        for data_type in DEFAULT_CONFIG['types']:
            type_dir = raw_root / year / data_type
            if not type_dir.exists():
                continue

            for class_dir in type_dir.iterdir():
                if class_dir.is_dir():
                    output_dir = processed_root / class_dir.name
                    output_dir.mkdir(exist_ok=True)

                    for img_path in class_dir.glob('*'):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            image_paths.append((img_path, output_dir))

    print(f"[INFO] Found {len(image_paths)} images to process.")
    error_count = 0

    for img_path, output_dir in tqdm(image_paths, desc="Processing images"):
        try:
            process_image(img_path, output_dir, model)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Processing stopped by user.")
            return
        except Exception:
            error_count += 1
            continue

    duration = time.time() - start_time
    print(f"[INFO] Processing complete! Output saved to: {processed_root}")
    print(f"[INFO] Scale bar removal completed in {duration:.2f} seconds.")

    if error_count > 0:
        print(f"Warning:  {error_count} images failed during processing.")

    wandb.log({
        "scale_bar_removale_time_sec": duration, 
        "scale_bar_removal_failures": error_count,
        "scale_bar_total_images": len(image_paths),
    })

if __name__ == "__main__":
    remove_scale_bars()
