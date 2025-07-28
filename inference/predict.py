import sys
import os
import argparse
import wandb
import torch
from inference.runner import run_inference
from config import DEFAULT_CONFIG
from PIL import Image
from datetime import datetime 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, choices=['resnet', 'densenet', 'efficientnet', 'yolov11'], required=True)
    parser.add_argument('--efficientnet_variant', type=str)
    parser.add_argument('--resnet_variant', type=str)
    parser.add_argument('--densenet_variant', type=str)
    parser.add_argument('--weights_path', type=str, required=True)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--save_csv', type=str)
    parser.add_argument('--wandb_log', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    results = run_inference(args, DEFAULT_CONFIG)
    if args.wandb_log:
        wandb.init(
            project=DEFAULT_CONFIG.get('project_name', 'classification'),
            name=f"{args.model_name}_{args.efficientnet_variant or args.densenet_variant  or args.resnet_variant or "Default"}_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            job_type="Inference",
            config={
                "image_dir": args.image_dir,
                "model_name": args.model_name,
                "weights_path": args.weights_path,
                "device": args.device,
                "batch_size": args.batch_size,
                "top_k": args.top_k,
            },
            tags = ["inference", args.model_name, args.efficientnet_variant or args.densenet_variant  or args.resnet_variant or "Default"],
        )

        # Dynamically create column headers for top-k + uncertainty
        columns = ["Image"]
        for i in range(args.top_k):
            columns += [f"Top{i+1}", f"Top{i+1} Score"]

        table = wandb.Table(columns=columns)

        for item in results:
            img = Image.open(item['image_path'])
            preds = item['predictions']
            row = [wandb.Image(img)]

            for pred in preds:
                    label, score = pred
                    row += [label, score]

            # Pad incomplete rows if fewer than top_k
            while len(row) < len(columns):
                row += ["", 0.0]

            table.add_data(*row)

        wandb.log({"predictions": table})
        wandb.finish()

if __name__ == "__main__":
    main()
