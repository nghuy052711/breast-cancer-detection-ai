# 🧠 Breast Cancer Detection System using Deep Learning

## 📌 Overview
This project presents an end-to-end AI system for breast cancer diagnosis from mammography images. It integrates object detection and image classification to support automated medical analysis.

The system consists of:
- Lesion detection using YOLOv8
- Tumor classification using ResNet50
- Optional GUI for visualization and interaction

---

## 🚀 Features
- Detect suspicious regions in mammography images
- Classify tumors as benign or malignant
- Modular pipeline (detection → classification)
- Ready for extension to real-time or deployment systems

---

## 🛠 Tech Stack
- **Programming:** Python  
- **Deep Learning:** PyTorch  
- **Detection:** YOLOv8  
- **Classification:** ResNet50  
- **Image Processing:** OpenCV  
- **ML Models:** Scikit-learn  
- **UI (optional):** PyQt5  
- **Database (optional):** MySQL  

---

## 📊 Performance
- **Detection Recall:** ~94%  
- **Classification Accuracy:** ~95%

- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Breast Tumor Detection App

A desktop application for detecting and classifying masses and calcifications in mammogram images using YOLO detection models and ResNet50-based feature extraction.

---

## Features

- Breast tissue density classification (A / B / C / D)
- Mass detection with adaptive preprocessing based on density
- Calcification detection using sliding window inference
- Benign / Malignant classification for each detected region
- DICOM file support
- Zoom and pan on result image
- Save result image to disk

---

## Folder Structure

Make sure your project folder looks exactly like this before running:

```
APP_MAIN/
├── detect_turmor_main_2.py
│
├── ui/
│   ├── __init__.py                        ← empty file, required
│   └── detect_turmor_ui.py
│
├── Detect_model/
│   ├── mass.pt
│   ├── calc.pt
│   └── resnet50_BreastDensity.pt
│
├── MassBM_model/
│   ├── Mass_BM.pt
│   ├── RandomForest_model.pkl
│   └── mass_preprocessing_objects.pkl
│
└── CalcBM_model/
    ├── Calc_BM.pt
    ├── XGBoost_model.pkl
    └── calc_preprocessing_objects.pkl
```

---

## Requirements

- Python 3.9 or higher (Python 3.10 recommended)
- pip

---

## Installation

**Step 1 — Clone or download the project**

Place all files according to the folder structure shown above.

**Step 2 — (Recommended) Create a virtual environment**

```bash
python -m venv venv
```

Activate it:
- Windows: `venv\Scripts\activate`
- macOS / Linux: `source venv/bin/activate`

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

> If you have an NVIDIA GPU and want to use CUDA, install PyTorch manually from https://pytorch.org before running the command above.

---

## How to Run

```bash
python detect_turmor_main_2.py
```

---

## How to Use

1. Click **Browser** to open a standard image file (`.png`, `.jpg`, `.jpeg`, `.bmp`)
2. Or click **Import DICOM** to open a `.dcm` DICOM file — it will be auto-converted to PNG
3. Click **Predict** to start detection
4. Wait for the progress bar at the bottom to complete
5. Use the radio buttons to switch between display modes:
   - **Show All** — display both mass and calcification boxes
   - **Show Mass** — display mass boxes only
   - **Show Calc** — display calcification boxes only
   - **Show None** — hide all boxes
6. Scroll the mouse wheel over the result image to zoom in/out
7. Click and drag on the result image to pan
8. Click **Save** to save the result image to the `File Folder Cancer Image` folder

---

## Output

- **Density label** — shown in the top-left of the result image and in the UI (e.g., `Density: C`)
- **Mass boxes** — blue = Benign, red = Malignant, green = Unknown
- **Calcification boxes** — purple = Benign, red = Malignant, magenta = Unknown
- **Classification summary** — shown in the UI (e.g., `Mass: 2B/1M | Calc: 0B/1M`)
- Saved images go to: `APP_MAIN/File Folder Cancer Image/`
- Converted DICOM images go to: `APP_MAIN/autosave_dicom_convert_jpg_folder/`

---

## Notes

- If a model file is missing, the app will still launch but that detection feature will be disabled
- The `.pkl` files are required for benign/malignant classification — without them, all detections will show as "Unknown"
- The app uses CPU by default — GPU (CUDA) will be used automatically if available
- Do not rename any model files or folders — the paths are hardcoded in the configuration section of the main file
