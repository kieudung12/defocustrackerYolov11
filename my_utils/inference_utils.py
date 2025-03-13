import os
import cv2
import time
import datetime
import random
import numpy as np
import pandas as pd
import supervision as sv

from my_utils.image_utils import load_image

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.prediction import visualize_object_predictions
from sahi.utils.cv import read_image

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from IPython.display import clear_output


def load_fasterrcnn(model_path, device, num_classes = 2, box_nms_thresh=0.6):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=2, box_nms_thresh=0.3)
    checkpoint = torch.load(model_path, map_location=device, weights_only = True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    return model

def predict_one_image(model, image, device):
    # Move the image to the specified device
    image = image.to(device)
    # Add batch dimension since model expects a batch of images (batch size 1)
    image = image.unsqueeze(0)
    # Run the image through the model (inference mode)
    with torch.no_grad():
        predictions = model(image)

    return predictions
