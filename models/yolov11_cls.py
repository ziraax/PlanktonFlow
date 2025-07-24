import os 
from ultralytics import YOLO
import wandb
import shutil

from training.eval import evaluate_yolo_model

class YOLOv11Classifier:
        def __init__(self, model_path=None):
            """
            Initialize the YOLOv11Classifier with a pre-trained model.
            
            Args:
                model_path (str, optional): Path to the pre-trained YOLOv11 model. Defaults to None.
            """
            self.model = YOLO(model_path) if model_path else YOLO('yolo11l-cls.pt')
        
        def train(self, run, config):
            """
            Train the YOLOv11 classification model using the built-in .train() method.
            
            Args:
                run (wandb.run): The WandB run object for logging.
                config (dict): Configuration dictionary containing training parameters.

            """

            data_path = config["final_dataset_path"]

            results = self.model.train(
                data=data_path,
                epochs=config['epochs'],
                imgsz=config['img_size'],
                batch=config['batch_size'],
                device=config['device'],
                augment=False,
                optimizer=config['optimizer'],
                patience=config['early_stopping_patience'],
                save_period=1,
                project=config['project_name'],
                name=run.name,
            )
            # Save the model
            weights_src = os.path.join(results.save_dir, "weights", "best.pt")
            weights_dst = os.path.join("model_weights", config['model_name'], config['model_variant'], run.name, "best.pt")
            os.makedirs(os.path.dirname(weights_dst), exist_ok=True)
            shutil.copy2(weights_src, weights_dst)

            return weights_dst
        
        def predict(self, image_path):
            """
            Runs inference on a single image using the YOLOv11 model. 
            """
            return self.model.predict(image_path, verbose=True)[0] # Gets the first prediction result

        def get_best_model_path(self):
            """
            Returns the path to the best model saved during training.
            """
            return os.path.join(self.model.save_dir, "weights", "best.pt")
        
        