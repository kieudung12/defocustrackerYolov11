import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np

from matplotlib.colors import hsv_to_rgb

def image_set(image_dir):
    image_files = sorted(list((os.listdir(image_dir))))
    imtype = image_files[0].split('.')[-1]
    imset = [os.path.join(image_dir, image_file) for image_file in image_files if image_file.split('.')[-1].lower() in ['tif','tiff','png','jpeg']]
    return imset
    
def load_image(image_path, to_float = True, to_tensor = True):
    # Read
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Change to 8-bit depth
    if image.dtype == np.uint16:
      image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
      image = np.uint8(image)

    # Convert to RGB
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    # Convert to float
    if to_float:
        image = image.astype(np.float32) / 255.0
    # Convert to tensor (if using for PyTorch)
    if to_tensor:
      image = torch.from_numpy(image).permute(2, 0, 1)
    return image
    
def create_colormap(N=600, seed=42):
    """
    Generate a colormap with N distinct colors, avoiding similar adjacent tones.
    """
    np.random.seed(seed)  # Ensure reproducibility
    
    # Generate evenly spaced hues (0 to 1), with random jitter for saturation and value
    hues = np.linspace(0, 1, N, endpoint=False)
    np.random.shuffle(hues)  # Shuffle hues to randomize the sequence

    # Generate random saturation and value, keeping them within a visually appealing range
    saturations = np.random.uniform(0.6, 0.9, N)  # Keep saturation relatively high
    values = np.random.uniform(0.7, 1.0, N)       # Keep brightness relatively high

    # Combine HSV components
    hsv_colors = np.stack([hues, saturations, values], axis=1)

    # Convert HSV to RGB for plotting
    rgb_colors = hsv_to_rgb(hsv_colors)

    return rgb_colors

def show_imageset(image_set, detections=pd.DataFrame(), ground_truth=pd.DataFrame(), bboxes=True, plot_size=(10, 8)):
    current_index = 0
    offset = 20
    offset_values = np.linspace(0.000001, 10, 1000)
    loaded_images = []
    id_to_color = {}
    COUNT = 0
    colors = create_colormap(N=detections['id'].nunique())  # Generate a large palette of unique colors

    # Load images
    for image_file in image_set:
        original_image = load_image(image_file, to_tensor=False)
        loaded_images.append(original_image)

    # Define function to update plot
    def plot_image(index, offset):
        if not detections.empty:
            frame = detections[detections['fr'] == index + 1]
            x_coords, y_coords = frame['X'], frame['Y']
            widths, heights = frame['W'], frame['H']
            ids = frame['id']

        if not ground_truth.empty:
            true_val = ground_truth[ground_truth['fr'] == index + 1]
            xt, yt = true_val['X'], true_val['Y']
            wt, ht = true_val['W'], true_val['H']

        image_array = loaded_images[index] * offset_values[offset]
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
        fig, ax = plt.subplots(figsize=plot_size)
        ax.imshow(image_array, cmap='gray')

        # Set axis labels
        ax.set_xlabel('X [pxs]')
        ax.set_ylabel('Y [pxs]')

        if bboxes:
            if not ground_truth.empty:
                for x1, y1, w1, h1 in zip(xt, yt, wt, ht):
                    rect = plt.Rectangle((x1 - w1/2 - 1, y1 - h1/2 - 1), w1, h1, linewidth=1, edgecolor='green', facecolor='none')
                    ax.add_patch(rect)

            # Plot bounding boxes and confidence scores
            if not detections.empty:
                for x, y, w, h, id in zip(x_coords, y_coords, widths, heights, ids):
                    if id not in id_to_color:
                        # Assign a unique color for each ID
                        id_to_color[id] = colors[len(id_to_color) % len(colors)]
                    edge_color = id_to_color[id]

                    rect = plt.Rectangle((x - w/2, y - h/2 ), w, h, linewidth=1, edgecolor=edge_color, facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x - w/2 +1, y - h/2+1 , f'{int(id):d}', color='white', fontsize=12)
                    
        # Save before displaying
        #save_path = './drive/MyDrive/Colab Notebooks/CNN_DPTV/defocustracker_faster_rcnn/output'
        #os.makedirs(save_path, exist_ok=True)
        #save_file = os.path.join(save_path, f'frame_{index}.png')
        #plt.savefig(save_file, dpi=400)
        
        plt.show()

    # Handlers for widget controls
    def on_next(change):
        nonlocal current_index
        if current_index < len(image_set) - 1:
            current_index += 1
            clear_output(wait=True)
            plot_image(current_index, offset_slider.value)
            display(prev_button, next_button, offset_slider)

    def on_previous(change):
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            clear_output(wait=True)
            plot_image(current_index, offset_slider.value)
            display(prev_button, next_button, offset_slider)

    def on_offset_change(change):
        clear_output(wait=True)
        plot_image(current_index, change['new'])
        display(prev_button, next_button, offset_slider)

    # Set up widgets
    next_button = widgets.Button(description="Next")
    prev_button = widgets.Button(description="Previous")
    offset_slider = widgets.IntSlider(value=100, min=1, max=1000, step=1, description='Brightness')

    # Attach event handlers
    next_button.on_click(on_next)
    prev_button.on_click(on_previous)
    offset_slider.observe(on_offset_change, names='value')

    # Display initial plot and controls
    plot_image(current_index, offset_slider.value)
    display(prev_button, next_button, offset_slider)