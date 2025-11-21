# LICENSES.md  
### YOLO4r — Licenses & Acknowledgements

This document catalogs all licenses, external libraries, contributors, and external resources used directly or indirectly in **YOLO4r**.  

YOLO4r is licensed under the **MIT License**, unless otherwise stated.

---

# 1. YOLO4r License (MIT)

MIT License

Kyle S. Goertler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.


---

# 2. Third-Party Dependencies & Licenses

YOLO4r depends on the following open-source Python packages.  
Refer to each repository for full license text.

## Torch  
- **License:** BSD-3-Clause  
- **Repo:** https://github.com/pytorch/pytorch  

## TorchVision  
- **License:** BSD-3-Clause  
- **Repo:** https://github.com/pytorch/vision  

## NumPy  
- **License:** BSD-3-Clause  
- **Repo:** https://github.com/numpy/numpy  

## OpenCV-Python-Headless  
- **License:** Apache-2.0  
- **Repo:** https://github.com/opencv/opencv-python  

## Pillow  
- **License:** HPND (Historical Permission Notice and Disclaimer)  
- **Repo:** https://github.com/python-pillow/Pillow  

## Matplotlib  
- **License:** PSF License  
- **Repo:** https://github.com/matplotlib/matplotlib  

## Pandas  
- **License:** BSD-3-Clause  
- **Repo:** https://github.com/pandas-dev/pandas  

## PyYAML  
- **License:** MIT  
- **Repo:** https://github.com/yaml/pyyaml  

## tqdm  
- **License:** MPL-2.0  
- **Repo:** https://github.com/tqdm/tqdm  

## Ultralytics  
- **License:** AGPL-3.0  
- **Repo:** https://github.com/ultralytics/ultralytics  
- **Models used:** YOLOv8, YOLOv8-OBB, YOLO11, YOLO11-OBB, YOLO12, YOLO12-OBB  

## ultralytics-thop  
- **License:** BSD-3-Clause  
- **Repo:** https://github.com/cz92/thop  

## Weights & Biases (wandb)  
- **License:** MIT  
- **Repo:** https://github.com/wandb/wandb  

## psutil  
- **License:** BSD-3-Clause  
- **Repo:** https://github.com/giampaolo/psutil  

## seaborn  
- **License:** BSD-3-Clause  
- **Repo:** https://github.com/mwaskom/seaborn  

---

# 3. Ultralytics Models (AGPL-3.0 Notice)

YOLO4r relies on Ultralytics model architectures, downloaded dynamically:

- YOLOv8  
- YOLOv8-OBB  
- YOLO11  
- YOLO11-OBB  
- YOLO12  
- YOLO12-OBB  

YOLO4r includes example **YOLO model weight files** that were trained using the
Ultralytics YOLO framework. Under the **Ultralytics AGPL-3.0 license**, all YOLO
weights—pretrained or user-trained—are considered AGPL-licensed artifacts.

While the YOLO4r source code is MIT-licensed, the included model weights fall
under AGPL-3.0. Users must comply with the AGPL-3.0 when using, modifying, or
redistributing these weight files.

AGPL-3.0 Full License:  
https://github.com/ultralytics/ultralytics/blob/main/LICENSE

---

# 4. External Acknowledgements

## 4.1 Edje Electronics (EdjeElectronics)

YOLO4r acknowledges core inspiration and structural influence from Edje Electronics, whose YOLO training materials guided YOLO4r’s earliest development phases.

The following scripts informed YOLO4r’s early design:

- `train.py` (conceptual structure)  
- `detect.py` (argument flow & training patterns)  
- `train_val_split.py` (dataset splitting logic)  

Source:  
https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb

These resources were foundational during initial architecture planning.

---

## 4.2 Robert Boyd — Kent State University

Special thanks to Robert Boyd, a graduate researcher at Kent State University, who contributed early prototype scripts and shared practical guidance that shaped YOLO4r’s training workflow.

His contributions supported the foundation upon which the YOLO4r pipeline was built.

---

## 4.3 Dr. Brian Trevelline — Kent State University

Special thanks to Dr. Brian Trevelline, an assistant professor and microbiologist at Kent State University, who pushed me to start this project at all. The entire foundation for the idea was rooted in using deep-learning models to study behavorial aspects of house sparrows.

His guidance and mentorship throughout the project set standards and objectives along the developmental path.

---

# 5. Redistribution Notes & Legal Requirements

## 5.1 Ultralytics AGPL-3.0 Notice  
Using Ultralytics models in production requires compliance with AGPL-3.0, which may obligate you to:

- Publish your source code if deploying as a service  
- Disclose modifications to Ultralytics components  
- Ensure downstream users can request the full source  

## 5.2 YOLO4r MIT License Compatibility  
Again, YOLO4r includes example YOLO model weights (`best.pt`, `last.pt`) trained using the
Ultralytics YOLO framework. According to the Ultralytics AGPL-3.0 license, all YOLO
weights—pretrained or user-trained—are licensed under AGPL-3.0.

Therefore, although the YOLO4r source code is MIT-licensed, the included model
weights are subject to AGPL-3.0 requirements. Users must ensure compliance with the
AGPL-3.0 when redistributing or deploying these weight files.

## 5.3 User Responsibility  
End users must ensure they comply with:

- MIT license (YOLO4r)  
- AGPL-3.0 (Ultralytics)  
- Licenses of all dependencies listed above  

---

# 6. Full License Texts
To view the complete license text of all dependencies:

- Check the `LICENSE` file in each dependency folder (located inside your Python `site-packages` dir)  
- Or visit the GitHub pages listed earlier  

You may also use `pip show <package>` to locate its license file.

---

# End of LICENSES.md
