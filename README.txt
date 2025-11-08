YOLO4r
-----------------------
Automated Animal Behavior Detection Pipeline

Overview
--------
YOLO4r (v5) is a YOLO by Ultralytics-based pipeline developing to allow researchers to design custom deep-learning models for accessible detection and measurement of animal behaviors. There is a current developmental focus on house sparrows for our own use-case. The pipeline supports custom dataset training, multiple video inputs and camera feeds processed in parallel, logs detection counts and interactions, and exports structured metrics for analysis. 
YOLO4r is still in early development as an undergraduate research project. The scope is rather large and may take time to develop to completion. The project will always be open-source and contributions are always welcome!

Features  
--------
• Option to train custom model using transfer-learning, training from scratch, or updating the current model.
• Training metrics parsed to Weights & Biases and a quick-summary.txt file.
• Multi‑threaded inference on one or multiple video and USB sources.  
• Detection of seven classes: M (males), F (females), Feeder, Main_Perch, Wooden_Perch, Sky_Perch, Nesting_Box.
• Interval‑based aggregation with session summaries. (e.g., counts, rates, M:F ratios)  
• Interaction metrics between bird sexes and objects. (e.g., male visits feeder, duration logged)  
• Modular utilities for routing, logging, folder structure, video-rotation correction.  
• Configurable run modes: '–-test' vs production.  
• Example output includes: video recordings with bounding boxes + CSVs of counts and interactions.  

Installation
------------
python -m venv --system-site-packages ~/yolo4r
source ~/yolo4r/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install torch>=2.8.0 torchvision>=0.23.0 numpy>=1.23.0 opencv-python-headless>=4.7.0 Pillow>=10.0.1 matplotlib>=3.6.3 pandas>=1.5.3 pyyaml>=6.0.0 tqdm>=4.64.1 ultralytics==8.3.184 ultralytics-thop>=2.0.16 wandb>=0.21.1 psutil>=5.9.5 seaborn>=0.13.0

Execution
---------
• Initiate training:
	- python train.py --train # Transfer-learning from custom dataset.
	- python train.py --update # Update the most recently trained model.
	- python train.py --scratch # Train a model only from custom dataset.
	- python train.py --test # Designed to allow users to debug training operation.

• Initiate object detection:
- python train.py --train # Transfer-learning from custom dataset.
	- python detect.py # Defaults to mostly recently trained model and initiates usb0.
	- python detect.py --sources usb0 usb1 "video1.type" "video2.type" # Initiate multiple sources in parallel.
	- python detect.py --test # Designed to allow users to route to debug model.
