from models.yolov11_cls import YOLOv11Classifier
from models.resnet import ResNetClassifier
from models.densenet import DenseNetClassifier
from models.efficientnet import EfficientNetClassifier

def create_model(
        model_name, 
        num_classes=None, 
        pretrained=True, 
        freeze_backbone=False, 
        efficientnet_variant="b0", 
        densenet_variant="121",
        resnet_variant="50",
        mc_dropout=False,
        mc_p=0.3
        ):
    """
    Factory function to create a model instance based on the model name.
    
    Args:
        model_name : str
            One of 'yolov11', 'resnet', 'densenet', 'efficientnet'.
        num_classes : int
            Number of output classes.
        pretrained : bool
            Load ImageNet weights.
        freeze_backbone : bool
            Freeze feature extractor layers.
        efficientnet_variant : str
            b0 … b7 (only if model_name == 'efficientnet').
        densenet_variant : str
            '121','169','201','161' (only if model_name == 'densenet').
        resnet_variant : str
            '18','34','50','101','152' (only if model_name == 'resnet').
        mc_dropout : bool
            If True, build the classifier with MC-dropout layers and expose
            .predict_mc().
        mc_p : float
            Dropout probability to use when mc_dropout is True.
    Returns:
        nn.Module: An instance of the specified mode OR a YOLOv11Classifier with own training logic.
    """

    if model_name == 'efficientnet':
        if efficientnet_variant is None:
            raise ValueError("efficientnet_variant must be specified when using EfficientNet model.")
        if efficientnet_variant not in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']:
            raise ValueError(f"Invalid efficientnet_variant: {efficientnet_variant}. Choose from 'b0' to 'b7'.")

    # Ensure densenet_variant is provided only for DenseNet models
    if model_name == 'densenet':
        if densenet_variant is None:
            raise ValueError("densenet_variant must be specified when using DenseNet model.")
        if densenet_variant not in ['121', '169', '201', '161']:
            raise ValueError(f"Invalid densenet_variant: {densenet_variant}. Choose from '121', '169', '201', '161'.")

    # Create the model based on the specified name and other parameters
    if model_name == 'yolov11':
        return YOLOv11Classifier()
    

    elif model_name == 'resnet':
        print(f"[INFO] Using ResNet with variant {resnet_variant} with pretrained={pretrained} and freeze_backbone={freeze_backbone}")
        return ResNetClassifier(
            num_classes=num_classes,
            variant=resnet_variant,
            pretrained=pretrained, 
            freeze_backbone=freeze_backbone,
            mc_dropout=mc_dropout,
            mc_p=mc_p
        )
    

    elif model_name == 'densenet':
        print(f"[INFO] Using DenseNet with pretrained={pretrained} and freeze_backbone={freeze_backbone}, variant={densenet_variant}")
        return DenseNetClassifier(
            num_classes=num_classes,
            variant=densenet_variant,
            pretrained=pretrained, 
            freeze_backbone=freeze_backbone,
            mc_dropout=mc_dropout,
            mc_p=mc_p
        )
    elif model_name == 'efficientnet':
        print(f"[INFO] Using EfficientNet with variant {efficientnet_variant}, pretrained={pretrained}, freeze_backbone={freeze_backbone}, mc_dropout={mc_dropout}, mc_p={mc_p}")
        return EfficientNetClassifier(
            num_classes=num_classes, 
            variant=efficientnet_variant, 
            pretrained=pretrained, 
            freeze_backbone=freeze_backbone,
            mc_dropout=mc_dropout,
            mc_p=mc_p
        )
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    

def get_model_list():
    """
    Returns a list of available model names.
    
    Returns:
        list: A list of strings representing the names of available models.
    """
    return ['yolov11', 'resnet', 'densenet', 'efficientnet']

def get_model_description(model_name):
    """
    Returns a description of the specified model.
    
    Args:
        model_name (str): The name of the model to describe.
    
    Returns:
        str: A description of the specified model.
    """
    descriptions = {
        'yolov11': 'YOLOv11 is a state-of-the-art object detection model that also has a classification version.',
        'resnet': 'ResNet is a deep residual network that helps in training very deep networks.',
        'densenet': 'DenseNet connects each layer to every other layer in a feed-forward fashion.',
        'efficientnet': 'EfficientNet is a family of models that scale up efficiently.'
    }
    return descriptions.get(model_name, "Model not found.")



