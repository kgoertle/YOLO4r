# YOLO4r
**You Only Look Once For Research**

An open-source, automated animal-behavior detection pipeline.

## Overview
**YOLO4r (beta-6.0)** is a research-oriented, Ultralytics-based pipeline designed to make custom deep-learning model training & behavioral detection accessible to field & laboratory researchers.  

**YOLO4r supports:**

- Multi-source real-time inference (video & live camera feeds).
- Structured logging of detections, interactions, & per-frame aggregate statistics.
- Automatic metadata extraction for precise timestamping.
- Full configurability & modular design for research reproducibility.

This project remains open-source & under active development as part of an undergraduate research initiative. Contributions & feedback are always welcome!

## Features

### Model Training
- Supports **transfer learning**, **training from scratch**, or **incremental updating** of an existing model.  
- Automatically exports **training metrics** to:
  - `Weights & Biases` (W&B)  
  - `quick-summary.txt` (local lightweight summary)
- Supports **aggressive data augmentation** & **auto-detection of new data** for retraining.

### Detection Pipeline
- **Multi-threaded inference** across multiple sources (camera feeds & videos).  
- **Metadata-aware timestamping** for accurate frame-aligned measurements.
- **Centralized message handling** using `Printer` for all info, warnings, errors, & save confirmations.
- **Robust exception handling** for model initialization, frame errors, & I/O failures.

### Classes & Configuration
- YOLO4r uses **user-defined class configurations**:
  - `FOCUS_CLASSES`: primary subjects (e.g., animal species)
  - `CONTEXT_CLASSES`: contextual or environmental elements (e.g., feeders, water trays, etc)
- Class lists are stored in & managed through `classes_config.yaml` within the config folder, allowing for easy modification without editing code.

Default example model trained on **7 classes**:
  - `M` (Male Passer domesticus), `F` (Female Passer domesticus), `Feeder`, `Main_Perch`, `Wooden_Perch`, `Sky_Perch`, `Nesting_Box`

### Measurement System
- Data collection centralized in single helper utility that handles:
  - Frame-level counts  
  - Interval-level aggregation  
  - Session summaries  
  - Interaction tracking (focus vs. context classes)
- Exports structured `.csv` summaries:
  - `counts.csv`, `average_counts.csv`
  - `interval_results.csv`, `session_summary.csv`
  - `interactions.csv`
Supports automatic calculation of ratios (e.g., M:F) & normalized detection rates.

### Directory and Output Structure
Integrates a **clean, timestamped log structure** for both camera feeds & videos:

**Camera sources:**
```
/YOLO4r/logs/(model_name)/measurements/camera-feed/(source_name)/(system_timestamp)/measurements/
├── recordings/
│   └── usb0.mp4
└── scores/
    ├── source_metadata.json
    ├── frame-data/
    │   ├── interval_results.csv
    │   └── session_summary.csv
    ├── counts/
    │   ├── counts.csv
    │   └── average_counts.csv
    └── interactions/
        └── interactions.csv
```

**Video sources:**
```
/YOLO4r/logs/(model_name)/measurements/video-in/(source_name)/(video_timestamp)/measurements/
├── recordings/
│   └── video.mp4
└── scores/
    ├── source_metadata.json
    ├── frame-data/
    │   ├── interval_results.csv
    │   └── session_summary.csv
    ├── counts/
    │   ├── counts.csv
    │   └── average_counts.csv
    └── interactions/
        └── interactions.csv
```

- Folder names are **automatically sanitized** to avoid filesystem errors.  
- Each source has its own **isolated measurement subdirectory**.  

## Installation
#### 1. Create the Python virtual environment:
`python -m venv --system-site-packages ~/yolo4r`

#### 2. Activate it using:
`source ~/yolo4r/bin/activate`

#### 3. Ensure Python wheels and installation tools are updated:
`python -m pip install --upgrade pip setuptools wheel`

#### 4. Install the library dependencies:
`pip install torch>=2.8.0 torchvision>=0.23.0 numpy>=1.23.0 opencv-python-headless>=4.7.0 Pillow>=10.0.1 matplotlib>=3.6.3 pandas>=1.5.3 pyyaml>=6.0.0 tqdm>=4.64.1 ultralytics==8.3.184 ultralytics-thop>=2.0.16 wandb>=0.21.1 psutil>=5.9.5 seaborn>=0.13.0`

#### 5. Clone the repository:
`git clone https://github.com/kgoertle/YOLO4r.git`
`cd ~/path/to/YOLO4r`

### Prerequisites
- Must use `Python 3.10` or older.
- Keep in mind, training and detection require entirely separate system requirements.
- A computer with a relatively powerful CPU or has a GPU with `CUDA enabled` is required.

## Execution
### Initiate Training
#### - Transfer-learning from custom dataset:
`python train.py --train`

#### - Update the most recently trained model:
`python train.py --update`

#### - Train a model only from custom dataset:
`python train.py --scratch`

#### - Designed to allow users to debug training operation:
`python train.py --test`

### Initiate Detection
#### - Defaults to mostly recently trained model and initiates usb0:
`python detect.py`

#### - Initiate multiple sources in parallel:
`python detect.py usb0 usb1 "video1.type" "video2.type"`

#### - Designed to allow users to route to debug model:
`python detect.py --test `
