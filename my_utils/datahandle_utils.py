import os
import json
import pandas as pd
import numpy as np
import sys
import torch
from datetime import datetime
def json2csv(detections, output_path = []):
    dets = json2list(detections)
    dets = pd.DataFrame(dets, columns = ['fr','id','X','Y','W','H','Cm','S1','S2','S3'])
    if output_path:
        dets.to_csv(output_path, index=False)
    return dets

def json2txt(detections, output_path = []):
    dets = json2list(detections)
    with open(output_path, "w") as file:
        for line in dets:
            file.write(",".join(map(str, line)) + "\n")
    return dets
    
def json2list(detections):
    dets = []
    for i in range(len(detections['annotations'])):
      for k in range(len(detections['annotations'][i]['X'])):
        dets.append([int(detections['annotations'][i]['fr']), 
                     detections['annotations'][i]['id'], 
                     np.round(detections['annotations'][i]['X'][k], 5),
                     np.round(detections['annotations'][i]['Y'][k], 5),
                     np.round(detections['annotations'][i]['W'][k], 5),
                     np.round(detections['annotations'][i]['H'][k], 5),
                     np.round(detections['annotations'][i]['Cm'][k], 5),
                     -1, -1, -1
                     ])
    return dets

def yolotxt2csv(dets_dir, img_size):
  df = pd.DataFrame()
  fr = 1
  dets_file = dets_dir.split('labels')[0]
  output_path = dets_file + '/dets.csv'
  for dets_file in os.listdir(dets_dir):
    if dets_file.endswith('.txt'):
      with open(os.path.join(dets_dir, dets_file), 'r') as file:
        detection = []
        for line in file:
          vals = line.split()
          vals = [float(x) for x in vals]
          vals[1:5] = [x * img_size for x in vals[1:5]]
          x, y, w, h = vals[1:5]
          cm = vals[5]
          detection.append([fr, -1, x, y, w, h, cm, -1, -1, -1])
        df = pd.concat([df, pd.DataFrame(detection)])
        fr += 1
  df.columns = ['fr', 'id', 'X','Y','W','H','Cm', 'S1','S2','S3']
  df.to_csv(output_path, index=False)
  return df
  
def project_root(working_dir, filename, model_type, task = 'detection'):
    if model_type:
        output_path = os.path.join(working_dir, 'runs', task, model_type)
    else:
        output_path = os.path.join(working_dir, 'runs', task)
        
    os.makedirs(output_path, exist_ok = True)
    lf = os.listdir(output_path)
    if not lf:
      output_path = os.path.join(output_path, f"{filename}_{1}")
    else:
      existing_folders = [
        item for item in os.listdir(output_path)
        if os.path.isdir(os.path.join(output_path, item)) and item.startswith(filename)
    ]
      # Find the next available number
      max_number = 0
      for folder in existing_folders:
        # Check if the folder name ends with a number
        try:
          number = int(folder[len(filename)+1:])  # Extract the numeric part
          max_number = max(max_number, number)
        except ValueError:
          continue  # Skip non-numeric folder suffixes

      # Create a new folder with the next number
      new_folder_name = f"{filename}_{max_number + 1}"
      output_path = os.path.join(output_path, new_folder_name)
      os.makedirs(output_path, exist_ok = True)
    return output_path

def coco2yolo(coco_json, img_dir, output_dir):
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    # Create output directories if they do not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a dictionary for category names
    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Loop through the images in the dataset
    for img in coco_data['images']:
        img_id = img['id']
        img_filename = img['file_name']
        img_width = img['width']
        img_height = img['height']

        # Find the annotations for the image
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        # Create a YOLO-format annotation file
        txt_file = os.path.join(output_dir, os.path.splitext(img_filename)[0] + '.txt')
        with open(txt_file, 'w') as out_f:
            for ann in annotations:
                category_id = 0
                category_name = 'particle'

                # Convert bounding box from COCO format to YOLO format
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height

                # Write the annotation to the text file (class_id x_center y_center width height)
                out_f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")
                
def copy_imageset(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    ## Loop through the files in the source folder
    for file_name in os.listdir(source_folder):
        # Check if the file is an image (you can modify the extensions as needed)
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff','.tif')):
            # Define full path for the source and destination
            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(destination_folder, file_name)

            # Copy the file
            shutil.copy(source_file, destination_file)
            print(f"Copied {file_name} to {destination_folder}")
            
class WelcomeMessage:
    def __init__(self, emoji="🔍"):
        self.emoji = emoji
        # Automatically fetch current date
        self.date = datetime.now().strftime("%Y-%m-%d")
        # Automatically fetch Python version
        self.python_version = sys.version.split()[0]
        # Automatically fetch PyTorch version
        self.torch_version = torch.__version__
        # Automatically fetch GPU information (if CUDA is available)
        self.gpu_info = self.get_gpu_info()

        # Automatically print the message when an object is created
        self.print_message()

    def get_gpu_info(self):
        """Get GPU info if CUDA is available."""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            name = torch.cuda.get_device_name(device)
            memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)  # in MB
            return f"{device} ({name}, {memory:.0f}MiB)"
        else:
            return "No CUDA-enabled GPU found"

    def print_message(self):
        """Print the message with all the gathered information."""
        message = "DefocusTrackerAI-1.0.0"
        print(f"{message} {self.emoji} {self.date} Python-{self.python_version} torch-{self.torch_version} CUDA:{self.gpu_info}")