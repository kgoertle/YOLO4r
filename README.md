# YOLO4r
**You Only Look Once For Research**

An open-source, automated animal-behavior detection pipeline.

## Overview
**YOLO4r (1.1.0)** is a research-oriented, Ultralytics-based pipeline designed to make custom deep-learning model training & behavioral detection accessible to field & laboratory researchers.  

**YOLO4r supports:**

- Multi-source real-time inference (video & live camera feeds).
- Structured logging of detections, interactions, & per-frame aggregate statistics.
- Automatic metadata extraction for precise timestamping for video and camera sources.
- Full configurability & modular design for research reproducibility.

This project remains open-source & under active development as part of an undergraduate research initiative. Contributions & feedback are always welcome!

## Features

### Model Training
- Supports **transfer learning** or **training from scratch** of an existing model.  
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


#### Here is an example of the a _classes config YAML_ file:
```
FOCUS_CLASSES:
- F
- M
- Feeder
- Main_Perch
- Nesting_Box
- Sky_Perch
- Wooden_Perch

CONTEXT_CLASSES: []

```

A model's class list is extracted straight from its `model.pt` file to prepare a `class_config.yaml` file, which will be located in the `/configs/<model_name>` path.
This allows for a class list to be divided between `focus` & `context` classes that simplifies output statistics & terminal logs.
This class setup is intended to set specific classes as _objects_ for _focus_ classes to interact with, giving context to those interactions primarily.
Please ensure that the [] are removed if defining `context` classes!

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

### Terminal UI
Note that YOLO4r is a _headless_ detection pipeline, meaning that live display windows will _not_ appear while running inference. 
Instead, the terminal logs & tracks initiation, FPS, and basic statistics.

#### Here is an example of what to expect from the terminal:
```
YOLO4r Detection
----------------

[MODEL] Model

[video1] Frames:-- | FPS:-- | Time:-- | ETA:--
  class1:-
  class2:-
  OBJECTS:-

[usb0] Frames:-- | FPS:-- | Time:--
  class1:-
  class2:-
  OBJECTS:-

------------------------------------------------------------------------------------------------

[MODEL] Model

[INFO] 3 models found in runs folder:
[INFO] Loaded 6 classes: ['class1', 'class2', 'class3', 'class4', 'class5', 'class6'
[INFO] Recording initialized at 11/26/2025 23:19:28
[INFO] Source 'video1' completed.
[INFO] Source 'usb0' completed.

------------------------------------------------------------------------------------------------

[MODEL] Model

[SAVE] Measurements for video1:
[SAVE] Measurements saved to: "measurements/video-in/video1/mm-dd-yyyy_hh-mm-ss/scores"
      - video1.mp4
      - video1_metadata.json
      - counts.csv
      - average_counts.csv
      - interval_results.csv
      - session_summary.csv
      - interactions.csv

[SAVE] Measurements for usb0:
[SAVE] Measurements saved to: "measurements/camera-feeds/usb0/mm-dd-yyyy_hh-mm-ss/scores"
      - usb0.mp4
      - usb0_metadata.json
      - counts.csv
      - average_counts.csv
      - interval_results.csv
      - session_summary.csv
      - interactions.csv

[EXIT] All detection threads safely terminated.
```

## Default example model trained on _7 classes_:
  - `M` (Male Passer domesticus)
  - `F` (Female Passer domesticus)
  - `Feeder`
  - `Main_Perch`
  - `Wooden_Perch`
  - `Sky_Perch`
  - `Nesting_Box`

The model was trained **using this pipeline** & has been used for primary testing purposes.

The purpose of this model in particular is use for tracking & logging basic behavioral attributes of captive _Passer domesticus_ subjects influenced by various intestinal microbial communities over an individual's development.

To be clear, the model is still in development & included for users to demonstrate a custom model trained through the pipeline.

## Installation
#### 1. Install MiniConda or Conda:
`https://www.anaconda.com/docs/getting-started/miniconda/main`

`https://www.anaconda.com/download`

#### 2. Create & activate environment using:
`conda create -n YOLO4r python=3.10`

`conda activate YOLO4r`

#### 3. Install the package:
`pip install yolo4r`

### Prerequisites
- Must use `Python 3.10` or older.
- Keep in mind, training & detection require entirely separate system requirements.
- A computer with a relatively powerful CPU or has a GPU with `CUDA enabled` is required.

## Execution
### Initiate Training
#### - Transfer-learning by default:
`yolo4r train`

**Option to specify weights from either OBB or standard YOLO model:**

`yolo4r train model=(yolo11n, yolo11l-obb, yolov8m, etc.)`

This will **default** to using YOLO11n.pt if not specified.


**Option to name the model:**

`yolo4r train name="my awesome run!!"`

**Option to specify dataset within `data` folder.**

`yolo4r train data="my awesome dataset!!`

This will **default** to the most recent dataset within the /data folder.


#### - Train a model only from custom dataset:

**Option to specify weights from either OBB or standard YOLO model.**

`yolo4r train architecture=(yolo11, yolo12, yolov8-obb, etc.)`

This will **default** to YOLO11.yaml if not specified.


#### - Designed to allow users to debug training operation:
`yolo4r train test`

#### - Process Label-Studio export folders:
`yolo4r train labelstudio="my awesome export!!"`

**NOTE: Many of these commands can be set together, so here are a few examples:**

`yolo4r train labelstudio=geckos model=yolo11m architecture=customgeckomodel`

`yolo4r train data=geckos model=yolo12 test`


### Initiate Detection
#### - Defaults to mostly recently trained model & initiates usb0:
`yolo4r detect`

#### - Initiate multiple sources in parallel:
`yolo4r detect usb0 usb1 "video1.type" "video2.type"`

#### - Designed to allow users to route to debug model:
`yolo4r detect test`






