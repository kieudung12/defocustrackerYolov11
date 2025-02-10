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


from yolov7_master.models.common import DetectMultiBackend
from yolov7_master.utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from yolov7_master.utils.torch_utils import select_device, smart_inference_mode
from yolov7_master.utils.augmentations import letterbox
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
    
@smart_inference_mode()
def inference_yolov9(image_set, 
                     weights, # Path to pretrained weights (.pt file)
                     imgsz=1024, # Image size
                     conf_thres=0.5, # Confidence threshold
                     iou_thres=0.5, # IOU threshold
                     device='0', # '0' GPU else 'cpu'
                     output_path = None # given by DefocusTrackerAI Class
                  ):
    '''
        Performs inference with YOLOv9 and returns a dataframe with detections.
        Detections from each image are individual save in a directory
    '''
    # Initialize and load model
    total_time = 0
    det_list = []
    frame = 0
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=False, data=None)
    stride, names, pt = model.stride, model.names, model.pt
    os.makedirs(output_path, exist_ok=True)

    # Process imageset
    for im in image_set:
      time_start = time.time()
      frame += 1

      # Read image
      image = cv2.imread(im, cv2.IMREAD_UNCHANGED) 

      # Conver to uint8 if necessary
      if image.dtype == np.uint16: 
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
      # Convert to RGB if necessary (for grayscale images)
      if len(image.shape) == 2:
        img0 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

      # Normalize, get torch tensor and else
      img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
      img = img[:, :, ::-1].transpose(2, 0, 1)  # HWC to CHW
      img = np.ascontiguousarray(img).astype(np.float32) / 255.0
      img = torch.from_numpy(img).to(device).float()
      gn = torch.tensor(img.shape)[[1, 0, 1, 0]]

      if img.ndimension() == 3:
        img = img.unsqueeze(0)

      # Inference
      pred = model(img, augment=False, visualize=False)
      # Apply NMS
      pred = non_max_suppression(pred[0][0], conf_thres, iou_thres)

      # Prepare to save detections from individual images to .txt files
      image_name = im.split('/')[-1]
      filename = output_path + '/'+ image_name.split('.')[0] + '.txt'
      # Open .txt
      with open(filename,'w') as out_file:
        # Analyze predictions
        for det in pred:
          if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape)
            for *xyxy, conf, cls in reversed(det):
                # Convert bounding boxes to x,y,w,h format                
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                # Format (Frame, id, X, Y, W, Z, Cm, S1, S2, S3)
                # Don't change, it has the format for the tracking scheme
                det_list.append(np.array([frame, -1, 
                                          xywh[0], xywh[1],
                                          xywh[2], xywh[3], 
                                          conf.cpu().numpy(),
                                          -1, -1, -1]))
                # Write detection to .txt
                print('%d,-1,%.4f,%.4f,%.4f,%.4f,%.4f,-1,-1,-1'%(frame, xywh[0], xywh[1],
                                                xywh[2], xywh[3], conf.cpu().numpy()),file=out_file)
      
      out_file.close()

      # Time counting section
      time_end = time.time()
      time_per_frame = time_end - time_start
      total_time += time_per_frame
      time_rem = np.round(total_time / frame * (len(image_set) - frame),3)
      tr = str(datetime.timedelta(seconds=time_rem))
      print(f"Processed image {im.split('/')[-1]}: Detections: {len(det)}, inference time: {int(time_per_frame * 1000 )} ms. Time remaining: {tr} (h:m:s).")

    # Save data to csv file as well
    det_df = pd.DataFrame(det_list, columns = ['fr','id','X', 'Y', 'W', 'H', 'Cm','S1','S2','S3'])
    det_df.to_csv(output_path + '/detections.csv', index=False)

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"That's it! Elapsed time: {hours} hours, {minutes} mins, {seconds} secs")
    print(f"Predictions saved as individual .txt files in {output_path}. CSV also available.")
    return det_df

# Function to make predictions on a single image
def inference_fasterrcnn(model, image_set, n_frames, cm_min, device):
    '''
    Output:
        - predictions: 
    '''
    count = 0
    Cm = [] 
    X = [] 
    Y = [] 
    W = [] 
    H = [] 
    annotations = []
    
    # Track total time spent and the frames processed
    total_time = 0
    
    for image_file in image_set:
        time_start = time.time()
        if count >= n_frames:
            break
        
        #Load image
        image = load_image(image_file)
        
        # Make predictions
        prediction = predict_one_image(model, image, device)
        
        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()

        filtered_boxes = boxes[scores >= cm_min]
        filtered_scores = scores[scores >= cm_min]
        
        for (box, score) in zip(filtered_boxes, filtered_scores):
            X.append(box[0] + (box[2] - box[0]) // 2)
            Y.append(box[1] + (box[3] - box[1]) // 2)
            W.append(box[2] - box[0]) 
            H.append(box[3] - box[1]) 
            Cm.append(score)
        
        annotations.append({
                    'image_name': os.path.basename(image_file),
                    'image_path': os.path.dirname(image_file),
                    'image_size': [image.shape[1], image.shape[0]],
                    'fr': float(count + 1),
                    'X': [float(x) for x in X],  # Convert to native Python float
                    'Y': [float(y) for y in Y],  # Convert to native Python float
                    'W': [float(w) for w in W],  # Convert to native Python float
                    'H': [float(h) for h in H],  # Convert to native Python float
                    'Cm': [float(cm) for cm in Cm],
                    'id': -1
                })
        
        Cm = [] 
        X = [] 
        Y = [] 
        W = [] 
        H = [] 
            
        time_end = time.time()
        time_per_frame = time_end - time_start
        total_time += time_per_frame
        
        remaining_frames = n_frames - count
        time_left = time_per_frame * remaining_frames
        time_left = str(datetime.timedelta(seconds=time_left))


        print(f"Processed {image_file.split('/')[-1]}: {count+1} out of {n_frames}. Time left: {time_left} (h:m:s). ")
        count += 1
        
    total_time = str(datetime.timedelta(seconds=total_time))
    print(f'Thats it! Elapsed time: {total_time} (h:m:s)')
    
    predictions = {
            'annotations': annotations,
            'imageset': image_set[0].split('/')[-2:-1],
            'cm_min': float(cm_min)
        }

    return predictions
    

# Function to make predictions on a single image
def inference_sahi(model, image_set, n_frames, cm_min, params = [], device = "cuda:0"):
    Cm = []
    X = []
    Y = []
    W = []
    H = []
    annotations = []
    total_time = 0
    image = load_image(image_set[0], to_float = False, to_tensor = False)
    h, w, c = image.shape
    s_h,s_w = h/4,w/4
    s_h ,s_w = int(s_h),int(s_w)
    image_size = [h if h > w else w]

    # Get params
    if not params:

        o_w, o_h = 0.1, 0.1

    if len(params) == 2:
        s_h, o_h = params
        s_w, o_w = params
    if len(params) == 4:
        s_h, s_w, o_h, o_w = params

    detection_model = AutoDetectionModel.from_pretrained(
        model_type = 'torchvision',
        model = model, #Faster RCNN Model
        confidence_threshold = cm_min,
        image_size = np.array(image_size), #Image's longest dimension
        device = "cuda:0", # or "cuda:0"
        load_at_init = True,
        )

    count = 0
    for image_file in image_set:
        time_start = time.time()
        if count >= n_frames:
            break
        image = load_image(image_file, to_float = False, to_tensor = False)

        result = get_sliced_prediction(
            image = image,
            detection_model = detection_model,
            slice_height = s_h,
            slice_width = s_w,
            overlap_height_ratio = o_h,
            overlap_width_ratio = o_w
            )

        boxes = np.array([pred.bbox.to_xyxy() for pred in result.object_prediction_list])
        scores = np.array([pred.score.value for pred in result.object_prediction_list])

        filtered_boxes = boxes[scores >= cm_min]
        filtered_scores = scores[scores >= cm_min]

        for (box, score) in zip(filtered_boxes, filtered_scores):
            X.append(box[0] + (box[2] - box[0]) // 2)
            Y.append(box[1] + (box[3] - box[1]) // 2)
            W.append(box[2] - box[0])
            H.append(box[3] - box[1])
            Cm.append(score)

        annotations.append({
                    'image_name': os.path.basename(image_file),
                    'image_path': os.path.dirname(image_file),
                    'image_size': [image.shape[1], image.shape[0]],
                    'fr': float(count + 1),
                    'X': [float(x) for x in X],  # Convert to native Python float
                    'Y': [float(y) for y in Y],  # Convert to native Python float
                    'W': [float(w) for w in W],  # Convert to native Python float
                    'H': [float(h) for h in H],  # Convert to native Python float
                    'Cm': [float(cm) for cm in Cm],
                    'id': -1
                })

        Cm = []
        X = []
        Y = []
        W = []
        H = []

        time_end = time.time()
        time_per_frame = time_end - time_start
        total_time += time_per_frame

        remaining_frames = n_frames - count
        time_left = time_per_frame * remaining_frames
        time_left = str(datetime.timedelta(seconds=time_left))

        print(f"Processed {image_file.split('/')[-1]}: {count+1} out of {n_frames}. Time left: {time_left} (h:m:s). ")
        count += 1

    total_time = str(datetime.timedelta(seconds=total_time))
    print(f'Thats it! Elapsed time: {total_time} (h:m:s)')

    predictions = {
            'annotations': annotations,
            'imageset': image_set[0].split('/')[-2:-1],
            'cm_min': float(cm_min)
        }

    return predictions