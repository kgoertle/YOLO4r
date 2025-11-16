# YOLO4r
**You Only Look Once For Research**

An open-source, automated animal-behavior detection pipeline.

## Overview
**YOLO4r (beta-8.1)** is a research-oriented, Ultralytics-based pipeline designed to make custom deep-learning model training & behavioral detection accessible to field & laboratory researchers.  

**YOLO4r supports:**

- Choice in a **variety** of YOLO models, including YOLOv8, YOLO11, and YOLO12.
- Option to use **transfer-learning weights** or train a model from scratch only using **model architecture** backbone (.yaml).
- **Automatically** processes Label-Studio projects into the `data` folder.
- Multi-source **real-time inference** (video & live camera feeds).
- Structured logging of **detections**, **interactions**, & **per-frame aggregate statistics**.
- Automatic **metadata extraction** for precise timestamping for video sources.
- Full **configurability** & **modular** design for research reproducibility.

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
- Class lists are stored in & managed through `classes_config.yaml` per model within the config folder, allowing for easy modification without editing code.

Example model trained on **7 classes**:
  - `M` (Male Passer domesticus), `F` (Female Passer domesticus), `Feeder`, `Main_Perch`, `Wooden_Perch`, `Sky_Perch`, `Nesting_Box`

### Measurement System
- Data collection centralized in single helper utility that handles:
  - Frame-level counts  
  - Interval-level aggregation  
  - Session summaries  
  - Interaction tracking (focus vs. context classes if defined)
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
#### 1. Install MiniConda or Conda:
`https://www.anaconda.com/docs/getting-started/miniconda/main`

`https://www.anaconda.com/download`

#### 2. Create & activate environment using:
`conda create -n YOLO4r python=3.10`

`conda activate YOLO4r`

#### 3. Ensure Python wheels & installation tools are updated:
`python -m pip install --upgrade pip setuptools wheel`

#### 4. Use pip to install the package:
`pip install yolo4r`

### Prerequisites
- Must use `Python 3.10` or older.
- Keep in mind, training & detection require entirely separate system requirements.
- A computer with a relatively powerful CPU or has a GPU with `CUDA enabled` is required.

## Execution
### Initiate Training
#### - Transfer-learning by default:
`yolo4r train`

**Option to specify weights from either OBB or standard YOLO models:**

`yolo4r train --model <yolo11, yolo11-obb, yolov8, yolov8-obb, yolo12>`

`yolo4r train -m <yolo11, yolo11-obb, yolov8, yolov8-obb, yolo12>`

This will **default** to using YOLO11n.pt if not specified.


**Option to name the model:**

`yolo4r train --name <custom name>`

`yolo4r train -n <custom name>`

**Option to specify dataset within `data` folder.**

`yolo4r train --dataset <name of dataset>`

`yolo4r train -d <name of dataset>`

This will **default** to the most recent dataset within the /data folder.



#### - Update the most recently trained model:

`yolo4r train --update`

`yolo4r train --u <name of model>`

This refers to the most recent `best.pt` file by **default** to train from **IF there are new images found in the dataset folder**. The model **can be specified**.


#### - Train a model only from custom dataset:
`yolo4r train --scratch`

`yolo4r train -s`

**Option to specify model architecture from either OBB or standard YOLO11 model.**

`yolo4r train --arch <yolo11, yolo11-obb, yolov8, yolov8-obb, yolo12, yolo12-obb>`

This will **default** to YOLO11.yaml if not specified.


#### - Designed to allow users to debug training operation:
`yolo4r train --test`

`yolo4r train -t`


### Initiate Detection
#### - Prompts to select trained model & initiates usb0:
`yolo4r detect`

#### - Initiate multiple sources in parallel:
`yolo4r detect usb0 usb1 video1.type video2.type`

#### - Designed to allow users to route to debug model:
`yolo4r detect --test`





