# DefoscusTrackerAI - A generalized framework for automated defocus particle detection

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnclctnh/defocustrackerAI-notebooks/blob/main/DefocusTrackerAI_Ready2Use.ipynb)

This project is part of the ISPIV 2025 contribution and a journal publication currently under review.

## Overview

DefocusTrackerAI is a ready-to-use generalized framework for the automated detection of defocus particle images, including:
    - Astigmatic and non-astigmatic particle images
    - Sprays and droplets
    - Potentially to non-spherical particles, cells and micro-organisms

## Get started 

First, you need to download the YOLOv9-m model weights that we trained for defocus particle images. The weights for the YOLOv9-m model trained for defocus particle images is available in the following link: 
    [**YOLOv9-M-DPTAI**](https://drive.google.com/uc?export=download&id=1yqtbfV1t14viiFXcD4B8w5NmMxR7RCtW)

Second, use the 'run colab button' on this README file and run the cells until Sec.2.1 Initiate DefocusTrackerAI. 

    In Sec. 2.1 You need to provide the directory containing the images that you want to analyze.
    In Sec. 2.2 You can define the number of frames that you want to process, the confidence threshold and IOU as well.
    In Sec. 2.3 You can visualize the results.

## Project status - Updates

In the coming days, a tracking scheme will included in the source files. 

## Authors and acknowledgment
I want to specially thank Prof. Massimiliano Rossi from University of Bologna who contributed to success of this project. 

## License

This project is licensed under the GNU General Public License v3.0.

You can view the full license text in the LICENSE file.



