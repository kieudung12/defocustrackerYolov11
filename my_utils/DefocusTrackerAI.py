"""
    DefocusTracker-AI
    2024-2025 Gonçalo Coutinho goncalo.coutinho@tecnico.ulisboa.pt
"""
import argparse
import numpy as np
import pandas as pd
import random
import os
import json
import sys
import time
import datetime
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

random.seed(42)
np.random.seed(42)

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
if torch.cuda.is_available():
  torch.cuda.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.prediction import visualize_object_predictions
from sahi.utils.cv import read_image
from IPython.display import Image

ROOT = os.getcwd()
sys.path.append(ROOT + '/yolov9_main')

from my_utils.image_utils import image_set, show_imageset
from my_utils.inference_utils import load_fasterrcnn, predict_one_image
from my_utils.tracking_utils import init_tracking_file
from my_utils.datahandle_utils import WelcomeMessage, json2csv, json2txt, json2list, yolotxt2csv, project_root

from yolov7_master.models.common import DetectMultiBackend
from yolov7_master.utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from yolov7_master.utils.torch_utils import select_device, smart_inference_mode
from yolov7_master.utils.augmentations import letterbox
from IPython.display import clear_output

from sort_master.sort import Sort, parse_args

class DefocusTrackerAI():
  '''
      This class contains the necessary functions to perform particle detection
      using a pre-trained model (Faster R-CNN, YOLOV9 or SAHI) and automatically
      track the objects along the image set using SORT.

      Parameters:
          - image_dir: directory containing the imageset

      Methods:
  '''
  def __init__(
              self,
              image_dir = None , # directory containing the imageset
            ):
    WelcomeMessage()
    # Get current working directory
    self.working_dir = os.getcwd()
    print(f"Working directory: {self.working_dir}")
    # Create imageset
    if image_dir:
      self.image_dir = image_dir
      self.image_set = image_set(self.image_dir)
    else:
      print("WARNING: Initiating without image directory.")

  def update_imageset(self, image_dir):
    if image_dir:
      self.image_dir = image_dir
      self.image_set = image_set(self.image_dir)
    else:
      print("WARNING: Need to provide an image directory as in mytracker.update_imageset(image_dir = \"...\").")

  def detect_fasterrcnn(self,
                        weights = None,
                        conf_thres = 0.5,
                        N_frames = 10,
                        max_det = 500,
                        enable_tracking = False
                    ):
    '''
      Faster R-CNN inference - returns a pandas DataFrame with the detections in the
      following format ['fr', 'id', 'X','Y','W','H','Cm', 'S1','S2','S3']. id
      refers to the trajectory id (-1 before tracking); S1, S2 and S3 are required
      variables for SORT tracking algorithm.
    '''
    if not weights:
      weights = './my_models/detect/fasterrcnn/fasterrcnn_dpt.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_fasterrcnn(weights, device)
    model.eval()
    model.roi_heads.detections_per_img = max_det

    image_set = self.image_set[0:N_frames]
    filename = image_set[0].split('/')[-2]

    output_path = project_root(self.working_dir,
                               image_set[0].split('/')[-2],
                               weights.split('/')[-3],
                               task = 'detection')
    os.makedirs(output_path, exist_ok=True)

    # Track total time spent and the frames processed
    total_time = 0
    det_list = []
    frame = 0
    for im in image_set:
      frame += 1
      time_start = time.time()

      #Load image
      image = cv2.imread(im, cv2.IMREAD_UNCHANGED)
      # Change to 8-bit depth
      if image.dtype == np.uint16:
        image = (image / np.max(image) * 255).clip(0, 255)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
      # Convert to RGB
      if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
      else:
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

      image = image.astype(np.float32) / 255.0
      image = torch.from_numpy(image).permute(2, 0, 1)
      image = image.to(device)

      # Add batch dimension since model expects a batch of images (batch size 1)
      if image.ndimension() == 3:
        image = image.unsqueeze(0)
      # Run the image through the model (inference mode)
      with torch.no_grad():
        prediction = model(image)
      # Make predictions

      boxes = prediction[0]['boxes'].cpu().numpy()
      scores = prediction[0]['scores'].cpu().numpy()

      filtered_boxes = boxes[scores >= conf_thres]
      filtered_scores = scores[scores >= conf_thres]

      # Prepare to save detections from individual images to .txt files
      image_name = im.split('/')[-1]
      txt_path = os.path.join(output_path,'txt_files')
      os.makedirs(txt_path, exist_ok=True)
      filename = txt_path + '/' + image_name.split('.')[0] + '.txt'
      # Open .txt
      with open(filename,'w') as out_file:
          # Analyze predictions
          for (box, score) in zip(filtered_boxes, filtered_scores):
            det_list.append(np.array([frame, -1,
                                      box[0] + (box[2] - box[0]) / 2,
                                      box[1] + (box[3] - box[1]) / 2,
                                      box[2] - box[0],
                                      box[3] - box[1],
                                      score, -1, -1, -1
                                  ]))
            # Write detection to .txt
            print('%d,-1,%.4f,%.4f,%.4f,%.4f,%.4f,-1,-1,-1'%(frame,
                                                             box[0] + (box[2] - box[0]) / 2,
                                                             box[1] + (box[3] - box[1]) / 2,
                                                             box[2] - box[0],
                                                             box[3] - box[1],
                                                             score),file=out_file)
      out_file.close()
      # Time counting section
      time_end = time.time()
      time_per_frame = time_end - time_start
      total_time += time_per_frame
      time_rem = np.round(total_time / frame * (len(image_set) - frame),3)
      tr = str(datetime.timedelta(seconds=time_rem))
      print(f"Processed image {im.split('/')[-1]}: Detections: {len(filtered_scores)}, inference time: {int(time_per_frame * 1000 )} ms. Time remaining: {tr} (h:m:s).")

    # Save data to csv file as well
    det_df = pd.DataFrame(det_list, columns = ['fr','id','X', 'Y', 'W', 'H', 'Cm','S1','S2','S3'])
    det_df.to_csv(output_path + '/detections.csv', index=False)

    filename = output_path  + '/inference_hyp.txt'
    # Open .txt
    with open(filename,'w') as f:
      f.write(f"weights: {weights}\n")
      f.write(f"conf_thres: {conf_thres}\n")
      f.write(f"N_frames: {N_frames}\n")
    f.close()

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"That's it! Elapsed time: {hours} hours, {minutes} mins, {seconds} secs")
    print(f"Txt files saved to {output_path}. CSV also available.")

    if enable_tracking:
      det_df = self.tracking(detections_dir = output_path,
                             max_age = 40,
                             min_track_length=5,
                             iou_threshold = 0.05
                          )
    return det_df

  @smart_inference_mode()
  def detect_yolov7(
      self,
      image_dir: str = None,
      weights: str = None,
      conf_thres = 0.5,
      N_frames = None,
      max_det = 500,
      iou_thres = 0.5,
      imgsz = 1024,
      enable_tracking = False,
  ):
    '''
      YOLO inference - returns a pandas DataFrame with the detections in the
      following format ['fr', 'id', 'X','Y','W','H','Cm', 'S1','S2','S3']. id
      refers to the trajectory id (-1 before tracking); S1, S2 and S3 are required
      variables for SORT tracking algorithm.
    '''
    if not self.image_dir:
      if image_dir:
        self.image_dir = image_dir
        self.image_set = image_set(self.image_dir)
      else:
        print("WARNING: There is no image directory. Please provide one.")
        return

    if not weights:
      print("WARNING: Init with default model: YOLOv7-m")
      weights = './my_models/detect/yolov7-m/yolov7_m_dpt.pt'
    
    if not N_frames:
      N_frames = len(self.image_set)

    image_set = self.image_set[0:N_frames]
    filename = self.image_set[0].split('/')[-2]

    if torch.cuda.is_available():
      device = '0'
    else:
      device = 'cpu'

    output_path = project_root(self.working_dir,
                               self.image_set[0].split('/')[-2],
                               weights.split('/')[-2],
                               task = 'detection')

    # Initialize and load model
    total_time = 0
    det_list = []
    frame = 0
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=False, data=None)
    # Dummy inference
    for ii in range(10):
      dummy_input = torch.zeros(1, 3, 1024, 1024).to('cuda' if torch.cuda.is_available() else 'cpu')
      # Perform dummy inference
      model(dummy_input, augment=False, visualize=False)

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
        image = (image / np.max(image) * 255).clip(0, 255)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
      # Convert to RGB if necessary (for grayscale images)
      if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

      # Normalize, get torch tensor and else
      img = letterbox(image, imgsz, stride=stride, auto=True)[0]
      img = img[:, :, ::-1].transpose(2, 0, 1)  # HWC to CHW
      img = np.ascontiguousarray(img).astype(np.float32)
      img /= 255
      img = torch.from_numpy(img).to(device).float()

      if img.ndimension() == 3:
        img = img.unsqueeze(0)

      # Inference
      pred = model(img, augment=False, visualize=False)

      # Apply NMS
      pred = non_max_suppression(pred[0][0], conf_thres, iou_thres = iou_thres, max_det = max_det)

      # Prepare to save detections from individual images to .txt files
      image_name = im.split('/')[-1]
      txt_path = os.path.join(output_path,'txt_files')
      os.makedirs(txt_path, exist_ok=True)
      filename = txt_path + '/' + image_name.split('.')[0] + '.txt'
    # Open .txt
      with open(filename,'w') as out_file:
        # Analyze predictions
        for det in pred:
          if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image.shape)
            for *xyxy, conf, cls in reversed(det):
              # Convert bounding boxes to x,y,w,h format
              xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
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

    filename = output_path  + '/inference_hyp.txt'
    # Open .txt
    with open(filename,'w') as f:
       f.write(f"weights: {weights}\n")
       f.write(f"imgsz: {imgsz}\n")
       f.write(f"conf_thres: {conf_thres}\n")
       f.write(f"iou_thres: {iou_thres}\n")
       f.write(f"max_det: {max_det}\n")
       f.write(f"N_frames: {N_frames}\n")
       f.write(f"Tracking: {enable_tracking}\n")
    f.close()

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"That's it! Elapsed time: {hours} hours, {minutes} mins, {seconds} secs")
    print(f"Txt files saved to {output_path}. CSV also available.")

    if enable_tracking:
      det_df = self.tracking(detections_dir = output_path,
                    max_age = 40,
                    min_track_length=5,
                    iou_threshold = 0.05
                )
                
    return det_df

  def compute_metrics(self,
                      detections,   # Pandas dataframe with detections
                      ground_truth, # Pandas dataframe with ground true values
                      iou_thres = 0.5,   # Distance threshold in pixels
                      mat2python = True
                  ):
      def iou(box1, box2):
        # Convert center format to top-left corner format
        box1_xmin = box1[0] - box1[2] / 2
        box1_ymin = box1[1] - box1[3] / 2
        box1_xmax = box1[0] + box1[2] / 2
        box1_ymax = box1[1] + box1[3] / 2

        box2_xmin = box2[0] - box2[2] / 2
        box2_ymin = box2[1] - box2[3] / 2
        box2_xmax = box2[0] + box2[2] / 2
        box2_ymax = box2[1] + box2[3] / 2

        # Calculate the intersection rectangle
        x1 = max(box1_xmin, box2_xmin)
        y1 = max(box1_ymin, box2_ymin)
        x2 = min(box1_xmax, box2_xmax)
        y2 = min(box1_ymax, box2_ymax)

        # Compute intersection area
        inter_width = max(0, x2 - x1)
        inter_height = max(0, y2 - y1)
        inter_area = inter_width * inter_height

        # Compute areas of both boxes
        box1_area = (box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)
        box2_area = (box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)

        # Compute union area
        union_area = box1_area + box2_area - inter_area

        # Return IoU
        return inter_area / union_area if union_area > 0 else 0

      print("DefocusTrackerAI-1.0.0 🔍 - Computing metrics ⏳\n")

      # Initialize
      if mat2python:
        offset = 1
      else:
        offset = 0

      tp, fp, fn, sp = 0, 0, 0, 0 # True positives, false positives, false negatives
      sigma_x, sigma_y = 0, 0
      # Restrict to frames present in both ground truth and detections
      processed_frames = set(ground_truth['fr']) & set(detections['fr'])

      # Evaluate detections
      for frame in processed_frames:
        gth = ground_truth[ground_truth['fr'] == frame].to_numpy()
        detection = detections[detections['fr'] == frame].to_numpy()
        matched_gt = set()
        for det in detection:
            best_iou = 0
            best_gt = None
            for idx, gt in enumerate(gth):
                if idx in matched_gt:
                    continue
                gthxywh = [gt[1] - offset, gt[2] - offset, gt[4], gt[5]]
                current_iou = iou(gthxywh, det[2:6])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt = idx

            if best_iou > iou_thres:
                tp += 1
                matched_gt.add(best_gt)
                sigma_x += np.square(gth[best_gt][1] - det[2])
                sigma_y += np.square(gth[best_gt][2]  - det[3])
            else:
                fp += 1

        fn += gth.shape[0] - len(matched_gt)
        sp += gth.shape[0]
      self.sigma_x = np.sqrt(sigma_x / tp)
      self.sigma_y = np.sqrt(sigma_y / tp)
      print(f"Sigma x: {np.round(self.sigma_x,3)}")
      print(f"Sigma y: {np.round(self.sigma_y,3)}")
      self.tp = tp
      self.fp = fp
      self.fn = fn
      print(f"True positives: {tp}")
      print(f"False positives: {fp}")
      print(f"False negatives: {fn}")
      self.precision = tp / (tp + fp) if tp + fp > 0 else 0
      self.recall = tp / (tp + fn) if tp + fn > 0 else 0
      print(f"Precision of {np.round(self.precision* 100, 3)} % for IOU threshold of {iou_thres}.")
      print(f"Recall of {np.round(self.recall* 100, 3)} % for IOU threshold of {iou_thres}.")
      return 0

  def save2json(self,
                detections,
                output_path):
    with open(output_path, 'w') as f:
      json.dump(detections, f)

  def imageset_viewer(self, detections, ground_truth = pd.DataFrame(), n_frames = 10, bboxes = True, plt_size = (10,8)):
    print("DefocusTrackerAI-1.0.0 🔍 - Launching image viewer 🖼\n")
    show_imageset(self.image_set[0:n_frames], detections, ground_truth, bboxes, plt_size)

  def tracking(self,
               detections_dir= None,
               max_age = 5,
               min_track_length=1,
               iou_threshold = 0.1
               ):
    """
        Based on the SORT algorithm
    """
    # Load detections
    if not detections_dir:
      print("ERROR: Please provide the detection directory to continue.")

    # csv file with the detections
    detections_file = detections_dir + "/detections.csv"
    if os.path.isfile(detections_file):
      seq_dets = np.loadtxt(detections_file, delimiter=",", skiprows=1)
      case_name = detections_file.split('/')[-1][:-5]
    else:
      print("ERROR: Please provide a .csv file with columns (fr, id, X, Y, W, Cm, S1, S2, S3) to proceed with tracking.")
    print(" \n")
    print("DefocusTrackerAI-1.0.0 🔍 - Tracking detected particles with SORT")

    # txt file with the inference information
    hyptxt = detections_dir + '/inference_hyp.txt'
    if os.path.isfile(hyptxt):
      with open(hyptxt, 'r') as f:
        lines = f.readlines()
      model_type = lines[0].split('/')[-2]
    else:
      model_type = None

    # Output path and .txt tracking data file
    output_path = project_root(self.working_dir,
                               detections_file.split('/')[-2],
                               model_type,
                               task = 'tracking')
    # Save info
    tracking_file = init_tracking_file(output_path, case_name)

    args = parse_args()
    phase = args.phase
    total_time = 0.0
    total_frames = 0

    mot_tracker = Sort(max_age=max_age,
                       min_hits=min_track_length,
                       iou_threshold=iou_threshold)
  
    trackers_list = []
    with open(tracking_file,'w') as out_file:
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          trackers_list.append([frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]])
          print('%d,%d,%.2f,%.2f,%.2f,%.2f'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)

    df = pd.DataFrame(trackers_list, columns=['fr', 'id','X', 'Y', 'W', 'H'])
    df_file = output_path + '/' + case_name + '_tracking.csv'
    df.to_csv(df_file, index=False)
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
    return df