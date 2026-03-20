#pyinstaller --clean APP_main.spec
import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pydicom
import joblib
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QInputDialog, QMessageBox, QLabel, QRadioButton, QButtonGroup, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QIcon, QCursor
from PyQt5.QtCore import Qt, QPoint, QRect, QThread, pyqtSignal, QMutex
from ui.detect_turmor_ui import Ui_DetectTurmor
import gc
from tabulate import tabulate
import sys
import time
from pathlib import Path

# ==================== PATHS AND CONFIGURATION ====================

# Resolve the base path depending on whether running as executable or script
if getattr(sys, 'frozen', False):
    # Running from a compiled executable
    application_path = Path(sys.executable).parent
else:
    # Running from source script
    application_path = Path(__file__).parent

# Model file paths
MODEL_MASS_PATH = str(application_path / "Detect_model" / "mass.pt")
MODEL_CALC_PATH = str(application_path / "Detect_model" / "calc.pt")
DENSITY_MODEL_PATH = str(application_path / "Detect_model" / "resnet50_BreastDensity.pt")

MASS_CLASSIFICATION_MODEL = str(application_path / "MassBM_model" / "RandomForest_model.pkl")
MASS_PREPROCESSING = str(application_path / "MassBM_model" / "mass_preprocessing_objects.pkl")
MASS_RESNET_PT = str(application_path / "MassBM_model" / "Mass_BM.pt")

CALC_CLASSIFICATION_MODEL = str(application_path / "CalcBM_model" / "XGBoost_model.pkl")
CALC_PREPROCESSING = str(application_path / "CalcBM_model" / "calc_preprocessing_objects.pkl")
CALC_RESNET_PT = str(application_path / "CalcBM_model" / "Calc_BM.pt")

# Detection configuration
CONF_MASS = 0.4
CONF_CALC = 0.6
NMS_THRESHOLD = 0.3
WINDOW_SIZE = 640
STRIDE = 320
MAX_WINDOWS = 100
BATCH_SIZE = 32

# Zoom configuration
ZOOM_STEP = 0.15
MAX_ZOOM = 5.0
MIN_ZOOM = 0.1

# Padding and ROI configuration
# Add 100 pixels of border around the display image to prevent text from being clipped at edges
DISPLAY_PADDING = 100  # Padding for display image to avoid losing text labels
# Add 20 pixels of border when cropping calcification ROI from the original image
ROI_PADDING_CALC = 100  # Padding when cropping calc ROI from the original image
ROI_PADDING_MASS = 50   # Padding when cropping mass ROI from the original image

# ==================== WORKER THREAD CLASS ====================

class DetectionWorker(QThread):
    """Worker thread for running detection without freezing the UI."""

    # Signals to communicate with the main thread
    progress_update = pyqtSignal(str)                        # Progress status updates
    detection_finished = pyqtSignal(object, list, list, str, list)  # Final detection results
    error_occurred = pyqtSignal(str)                         # Error reporting

    def __init__(self, detect_image_instance, image_path):
        super().__init__()
        self.detect_image = detect_image_instance
        self.image_path = image_path

    def run(self):
        """Run detection in background thread."""
        try:
            self.progress_update.emit("Starting diagnosis...")

            # Step 1: Load the original image from file
            self.progress_update.emit("Loading original image...")
            original_image = cv2.imread(str(self.image_path))
            if original_image is None:
                raise ValueError(f"Cannot load image: {self.image_path}")

            # Store the original image in the main DetectImage instance
            self.detect_image.original_image = original_image
            img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # Step 2: Classify breast tissue density
            self.progress_update.emit("Classifying breast tissue density...")
            density_label, density_confidences = self.detect_image.classify_density(str(self.image_path))

            # Step 3: Detect masses using the enhanced algorithm
            self.progress_update.emit("Detecting masses with enhanced algorithm...")
            processed_img, mass_results = self.detect_image.detect_mass_enhanced_threaded(str(self.image_path), img.copy())

            # Step 4: Detect calcifications using sliding window
            self.progress_update.emit("Detecting calcifications with sliding window...")
            processed_img, calc_results = self.detect_image.detect_calc_with_sliding_threaded(str(self.image_path), processed_img.copy())

            # Step 5: Create the base display image with padding
            self.progress_update.emit("Processing final image...")
            base_image, padding_offset = add_padding_to_image(processed_img, DISPLAY_PADDING)

            # Emit final results back to the main thread
            self.detection_finished.emit(base_image, density_confidences, mass_results, density_label, calc_results)

        except Exception as e:
            self.error_occurred.emit(str(e))

# ==================== HELPER CLASSES ====================

class ResNet50FeatureExtractor(torch.nn.Module):
    """Feature extractor using a ResNet50 backbone."""

    def __init__(self):
        super().__init__()
        self.features = models.resnet50(weights=None)
        self.features.fc = torch.nn.Identity()  # Remove the final fully connected layer

    def forward(self, x):
        return self.features(x)

# ==================== HELPER FUNCTIONS ====================

def add_padding_to_image(img, padding=DISPLAY_PADDING, fill_value=0):
    """Add uniform padding around the image to prevent text labels from being clipped at edges."""
    if len(img.shape) == 3:
        h, w, c = img.shape
        padded_img = np.full((h + 2 * padding, w + 2 * padding, c), fill_value, dtype=img.dtype)
        padded_img[padding:padding + h, padding:padding + w] = img
    else:
        h, w = img.shape
        padded_img = np.full((h + 2 * padding, w + 2 * padding), fill_value, dtype=img.dtype)
        padded_img[padding:padding + h, padding:padding + w] = img

    return padded_img, padding


def adjust_bbox_coordinates(bbox, padding_offset):
    """Adjust bounding box coordinates to account for display padding offset."""
    x1, y1, x2, y2 = bbox
    return (x1 + padding_offset, y1 + padding_offset,
            x2 + padding_offset, y2 + padding_offset)


def crop_roi_from_original(original_img, bbox, padding=0):
    """Crop a region of interest from the original image with optional border padding."""
    x1, y1, x2, y2 = bbox
    h, w = original_img.shape[:2]

    # Extend by padding while staying within image boundaries
    x1_padded = max(0, x1 - padding)
    y1_padded = max(0, y1 - padding)
    x2_padded = min(w, x2 + padding)
    y2_padded = min(h, y2 + padding)

    # Crop the region
    roi = original_img[y1_padded:y2_padded, x1_padded:x2_padded]
    actual_bbox = (x1_padded, y1_padded, x2_padded, y2_padded)

    return roi, actual_bbox

# ==================== CLASSIFICATION HELPER FUNCTIONS ====================

def preprocess_image_array(image_array):
    """Preprocess a pre-loaded image array for classification."""
    if len(image_array.shape) == 3:
        img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        img = image_array
    img = cv2.resize(img, (224, 224))
    img_eq = cv2.equalizeHist(img)
    img_filtered = cv2.medianBlur(img_eq, 3)
    return img, img_eq, img_filtered


def extract_HOG(image):
    """Extract Histogram of Oriented Gradients (HOG) features from an image."""
    from skimage.feature import hog
    HOG_CELL_SIZE = (8, 8)   # Controls the gradient detail level
    HOG_BLOCK_SIZE = (2, 2)  # Controls the normalization range
    HOG_BINS = 9              # Controls gradient orientation resolution

    features, _ = hog(image,
                      orientations=HOG_BINS,
                      pixels_per_cell=HOG_CELL_SIZE,
                      cells_per_block=HOG_BLOCK_SIZE,
                      block_norm='L2-Hys',
                      visualize=True,
                      feature_vector=True)
    print("HOG features shape:", features.shape)
    return features


def extract_LBP_advanced(image, use_multi_scale=True):
    """Extract Local Binary Pattern (LBP) features, optionally at multiple scales."""
    from skimage.feature import local_binary_pattern
    LBP_RADIUS = 2              # Radius for LBP computation
    LBP_POINTS = 8 * LBP_RADIUS  # Number of sampling points = 8 * radius
    LBP_METHOD = 'uniform'      # Use the uniform LBP variant

    lbp = local_binary_pattern(image, P=LBP_POINTS, R=LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    multi_scale_hist = []
    if use_multi_scale:
        for r in [1, 2, 3]:
            points = 8 * r
            lbp_ms = local_binary_pattern(image, P=points, R=r, method='uniform')
            n_bins_ms = points + 2
            hist_ms, _ = np.histogram(lbp_ms.ravel(), bins=n_bins_ms, range=(0, n_bins_ms))
            hist_ms = hist_ms.astype("float")
            hist_ms /= (hist_ms.sum() + 1e-7)
            multi_scale_hist.extend(hist_ms)

    return hist, np.array(multi_scale_hist)


def extract_shape_features(preprocessed_image, threshold_value=120, min_area=50):
    """Extract shape-based features (area, perimeter, circularity, eccentricity) for a mass ROI."""
    if len(preprocessed_image.shape) == 3:
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)

    cropped = preprocessed_image[5:-5, 5:-5]

    # Enhance contrast with CLAHE to better highlight the lesion region
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(cropped)

    # Hard threshold to reduce false detection of weakly bright areas
    _, thresh = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    if len(contours) == 0:
        # No contour found; return zero values
        return {
            'area': 0,
            'perimeter': 0,
            'circularity': 0,
            'eccentricity': 0
        }

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-7)

    # Compute central moments for eccentricity
    M = cv2.moments(largest_contour)
    mu20 = M['mu20']
    mu02 = M['mu02']
    mu11 = M['mu11']

    A = (mu20 + mu02)
    if A != 0:
        eccentricity = ((mu02 - mu20) ** 2 + 4 * mu11 ** 2) / (A + 1e-7)
    else:
        eccentricity = 0

    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'eccentricity': eccentricity
    }


def extract_deep_features(image, model):
    """Extract deep feature vector from an image using a ResNet50 feature extractor."""
    torch_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_tensor = torch_transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
    with torch.no_grad():
        features = model(image_tensor).squeeze(0)
    return features.numpy()


def combine_features(hog_features, lbp_hist, ms_lbp=None, shape_features=None, deep_features=None, weights=(0.6, 0.4)):
    """Combine HOG, LBP, multi-scale LBP, shape, and deep features into a single feature vector."""
    hog_norm = hog_features / (np.linalg.norm(hog_features) + 1e-7)
    lbp_norm = lbp_hist / (np.linalg.norm(lbp_hist) + 1e-7)

    combined = np.concatenate([
        hog_norm * weights[0],
        lbp_norm * weights[1]
    ])

    if ms_lbp is not None and len(ms_lbp) > 0:
        ms_lbp_norm = ms_lbp / (np.linalg.norm(ms_lbp) + 1e-7)
        combined = np.concatenate([combined, ms_lbp_norm * 0.3])

    if shape_features is not None:
        shape_array = np.array([
            shape_features.get("area", 0),
            shape_features.get("perimeter", 0),
            shape_features.get("circularity", 0),
            shape_features.get("eccentricity", 0)
        ])
        shape_array_norm = shape_array / (np.linalg.norm(shape_array) + 1e-7)
        combined = np.concatenate([combined, shape_array_norm * 0.5])

    # Append deep features if provided
    if deep_features is not None:
        deep_features_norm = deep_features / (np.linalg.norm(deep_features) + 1e-7)
        combined = np.concatenate([combined, deep_features_norm * 0.7])

    return combined

# ==================== MAIN CLASS ====================

class DetectImage(QWidget):
    def __init__(self):
        super(DetectImage, self).__init__()

        # Ensure the window activates correctly without forcing focus
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        # Ensure detection window stays on top
        self.ui = Ui_DetectTurmor()
        self.ui.setupUi(self)

        self.setWindowFlags(Qt.Window)

        # ==================== VARIABLE INITIALIZATION ====================

        # Initialize all model references to None before loading
        self.mass_model = None
        self.calc_model = None
        self.density_model = None

        # Classification model references
        self.mass_classifier = None
        self.mass_preprocessor = None
        self.mass_resnet = None

        self.calc_classifier = None
        self.calc_preprocessor = None
        self.calc_resnet = None

        # Worker thread state
        self.detection_worker = None
        self.is_detecting = False

        # Storage for detected ROI results
        self.detected_masses = []
        self.detected_calcs = []

        # Storage for original image (used for ROI cropping and display)
        self.original_image = None
        self.base_image = None    # Base image without bounding boxes drawn
        self.padding_offset = 0   # Pixel offset introduced by display padding

        # Zoom and pan state
        self.zoom_factor = 1.0
        self.original_pixmap = None
        self.pan_offset = QPoint(0, 0)
        self.last_pan_point = QPoint()
        self.is_panning = False

        # ==================== UI SETUP ====================

        self.setup_ui_labels()
        self.setup_button_connections()
        self.setup_zoom_and_pan()

        # ==================== MODEL LOADING ====================

        print("Loading models...")
        self.load_models()
        print("All models loading completed!")

    def setup_ui_labels(self):
        """Set up dynamic UI labels for density, classification summary, and progress."""
        # Breast tissue density display label
        self.density_label = QLabel("Breast tissue density: undetermined", self)
        self.density_label.setStyleSheet("font-size: 14px; color: #1f618d; font-weight: bold;")
        self.ui.gridLayout_2.addWidget(self.density_label, 6, 0, 1, 2, Qt.AlignRight)

        # Classification summary label
        self.classification_summary_label = QLabel("Classification summary: No detections", self)
        self.classification_summary_label.setStyleSheet("font-size: 14px; color: #e74c3c; font-weight: bold;")
        self.ui.gridLayout_2.addWidget(self.classification_summary_label, 7, 0, 1, 2, Qt.AlignRight)

        # Progress status label
        self.progress_label = QLabel("Ready for detection", self)
        self.progress_label.setStyleSheet("font-size: 12px; color: #2c3e50; font-weight: bold;")
        self.ui.gridLayout_2.addWidget(self.progress_label, 11, 0, 1, 2, Qt.AlignLeft)

    def setup_button_connections(self):
        """Connect UI buttons and radio buttons to their handler functions."""
        self.save_result = self.ui.btn_save
        self.save_result.clicked.connect(self.show_save_dialog)

        self.browser = self.ui.btn_browser
        self.browser.clicked.connect(self.browser_file)

        self.importdicom = self.ui.btn_browser_2
        self.importdicom.clicked.connect(self.open_dicom)

        self.predict = self.ui.btn_predict
        self.predict.clicked.connect(self.diagnose)

        self.input_label = self.ui.label_input
        self.output_label = self.ui.label_output
        self.directImage = self.ui.lineEdit_direct

        # Create button group for display mode radio buttons
        self.bbox_button_group = QButtonGroup(self)
        self.bbox_button_group.addButton(self.ui.radio_show_all, 0)
        self.bbox_button_group.addButton(self.ui.radio_show_mass, 1)
        self.bbox_button_group.addButton(self.ui.radio_show_calc, 2)
        self.bbox_button_group.addButton(self.ui.radio_show_none, 3)

        # Connect radio button signal to handler
        self.bbox_button_group.buttonClicked.connect(self.on_bbox_display_changed)

    def setup_zoom_and_pan(self):
        """Configure output label for zoom and pan interactions."""
        # Disable auto-scaling so we can handle it manually
        self.output_label.setScaledContents(False)
        self.output_label.setMouseTracking(True)

        # Bind mouse/wheel events to custom handlers
        self.output_label.wheelEvent = self.wheel_event
        self.output_label.mousePressEvent = self.mouse_press_event
        self.output_label.mouseMoveEvent = self.mouse_move_event
        self.output_label.mouseReleaseEvent = self.mouse_release_event

    # ==================== QTHREAD FUNCTIONS ====================

    def diagnose(self):
        """Start the diagnosis pipeline using a background QThread."""
        if hasattr(self, 'opt_source') and self.opt_source:
            if self.is_detecting:
                QMessageBox.information(self, "Info", "Detection is already running. Please wait...")
                return

            try:
                self.is_detecting = True
                self.predict.setText("Detecting...")
                self.predict.setEnabled(False)

                # Clear previous detection results
                self.detected_masses = []
                self.detected_calcs = []

                # Create and start the detection worker thread
                self.detection_worker = DetectionWorker(self, self.opt_source)

                # Connect worker signals to handler slots
                self.detection_worker.progress_update.connect(self.update_progress)
                self.detection_worker.detection_finished.connect(self.on_detection_finished)
                self.detection_worker.error_occurred.connect(self.on_detection_error)

                # Start the thread
                self.detection_worker.start()

            except Exception as e:
                self.on_detection_error(str(e))
        else:
            QMessageBox.warning(self, "Error", "Please select an image before prediction.")

    def update_progress(self, message):
        """Update the progress label with the latest status message."""
        self.progress_label.setText(f"Progress: {message}")
        QApplication.processEvents()  # Allow the UI to refresh

    def on_detection_finished(self, base_image, density_confidences, mass_results, density_label, calc_results):
        """Handle successful completion of the detection pipeline."""
        self.current_density_label = density_label
        try:
            # Store detection results
            self.base_image = base_image
            self.padding_offset = DISPLAY_PADDING

            # Update breast density label with confidence breakdown
            self.density_label.setText(
                f"Breast tissue density: {density_label} "
                f"(A: {density_confidences[0]:.2%}, B: {density_confidences[1]:.2%}, "
                f"C: {density_confidences[2]:.2%}, D: {density_confidences[3]:.2%})"
            )

            # Refresh classification summary
            self.update_classification_summary()

            # Render the result image
            self.update_display_image()

            # Free memory after processing
            self.clear_memory()

            # Update UI state
            self.progress_label.setText(
                f"Completed! Found {len(self.detected_masses)} masses, {len(self.detected_calcs)} calcifications"
            )
            self.predict.setText("Predict")
            self.predict.setEnabled(True)
            self.is_detecting = False

            print("\nDiagnosis completed successfully!")
            print(f"Summary: {len(self.detected_masses)} masses, {len(self.detected_calcs)} calcifications detected")

        except Exception as e:
            self.on_detection_error(f"Error processing results: {str(e)}")

    def on_detection_error(self, error_message):
        """Handle errors that occur during the detection pipeline."""
        QMessageBox.warning(self, "Detection Error", f"Detection failed:\n{error_message}")
        self.progress_label.setText(f"Error: {error_message}")
        self.predict.setText("Predict")
        self.predict.setEnabled(True)
        self.is_detecting = False
        self.clear_memory()
        print(f"Detection error: {error_message}")

    def closeEvent(self, event):
        """Ensure the worker thread is properly cleaned up when the window is closed."""
        if self.detection_worker and self.detection_worker.isRunning():
            self.detection_worker.terminate()
            self.detection_worker.wait()
        event.accept()

    # ==================== MODEL LOADING FUNCTIONS ====================

    def load_models(self):
        """Load all required AI models with progress feedback."""
        try:
            self.progress_label.setText("Loading AI models...")
            QApplication.processEvents()

            # Load YOLO mass detection model
            if os.path.exists(MODEL_MASS_PATH):
                import warnings
                warnings.filterwarnings("ignore")

                self.mass_model = YOLO(MODEL_MASS_PATH)
                print(f"Mass model loaded: {MODEL_MASS_PATH}")
            else:
                self.mass_model = None
                print(f"Mass model not found: {MODEL_MASS_PATH}")

            self.progress_label.setText("Loading calcification model...")
            QApplication.processEvents()

            # Load YOLO calcification detection model
            if os.path.exists(MODEL_CALC_PATH):
                self.calc_model = YOLO(MODEL_CALC_PATH)
                print(f"Calc model loaded: {MODEL_CALC_PATH}")
            else:
                self.calc_model = None
                print(f"Calc model not found: {MODEL_CALC_PATH}")

            self.progress_label.setText("Loading density classification model...")
            QApplication.processEvents()

            # Load breast density classification model
            self.density_model = self.load_density_model(DENSITY_MODEL_PATH)

            self.progress_label.setText("Loading classification models...")
            QApplication.processEvents()

            # Load benign/malignant classification models
            self.load_classification_models()

            self.progress_label.setText("All models loaded successfully!")

        except Exception as e:
            self.progress_label.setText(f"Error loading models: {str(e)}")
            print(f"Error loading models: {str(e)}")
            QMessageBox.warning(self, "Model Error", f"Error loading models:\n{str(e)}")

    def load_classification_models(self):
        """Load benign/malignant classification models for mass and calcification."""
        try:
            # Load mass classification models
            if os.path.exists(MASS_CLASSIFICATION_MODEL) and os.path.exists(MASS_PREPROCESSING):
                self.mass_classifier = joblib.load(MASS_CLASSIFICATION_MODEL)
                self.mass_preprocessor = joblib.load(MASS_PREPROCESSING)
                print("Mass classifier loaded")

                # Load mass ResNet feature extractor
                if os.path.exists(MASS_RESNET_PT):
                    self.mass_resnet = ResNet50FeatureExtractor()
                    state_dict = torch.load(MASS_RESNET_PT, map_location='cpu')
                    self.mass_resnet.features.load_state_dict(state_dict, strict=False)
                    self.mass_resnet.eval()
                    print("Mass ResNet loaded")
                else:
                    print(f"Mass ResNet not found: {MASS_RESNET_PT}")
            else:
                print("Mass classification models not found")

            # Load calcification classification models
            if os.path.exists(CALC_CLASSIFICATION_MODEL) and os.path.exists(CALC_PREPROCESSING):
                self.calc_classifier = joblib.load(CALC_CLASSIFICATION_MODEL)
                self.calc_preprocessor = joblib.load(CALC_PREPROCESSING)
                print("Calc classifier loaded")

                # Load calc ResNet feature extractor
                if os.path.exists(CALC_RESNET_PT):
                    self.calc_resnet = ResNet50FeatureExtractor()
                    state_dict = torch.load(CALC_RESNET_PT, map_location='cpu')
                    self.calc_resnet.features.load_state_dict(state_dict, strict=False)
                    self.calc_resnet.eval()
                    print("Calc ResNet loaded")
                else:
                    print(f"Calc ResNet not found: {CALC_RESNET_PT}")
            else:
                print("Calc classification models not found")

        except Exception as e:
            print(f"Error loading classification models: {str(e)}")

    def load_density_model(self, path):
        """Load the breast tissue density classification model from disk."""
        if not os.path.exists(path):
            print(f"Density model not found: {path}")
            return None
        try:
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 4)
            state_dict = torch.load(path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Density model loaded: {path}")
            return model
        except Exception as e:
            print(f"Error loading density model: {str(e)}")
            return None

    # ==================== IMAGE PREPROCESSING FUNCTIONS ====================

    def preprocess_for_density(self, img_np):
        """Preprocess image for breast tissue density classification."""
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        elif img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        return transform(img_np)

    def preprocess_for_mass_detection(self, img, density_level):
        """
        Preprocess image for mass detection based on breast tissue density level.

        Args:
            img: Input image (RGB or grayscale)
            density_level: Breast tissue density category ('A', 'B', 'C', 'D')

        Returns:
            Preprocessed image array
        """
        # Convert to RGB if needed
        if img.ndim == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if density_level in ['A', 'B']:
            # Low density: apply mild CLAHE
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        else:  # density_level in ['C', 'D']
            # High density: apply stronger CLAHE
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img

    def preprocess_for_calc_detection(self, img, density_level):
        """Preprocess image for calcification detection based on breast tissue density level."""
        if img.ndim == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if density_level in ['A', 'B']:
            # Low density: apply mild CLAHE
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        else:  # density_level in ['C', 'D']
            # High density: apply slightly stronger CLAHE
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img

    # ==================== CLASSIFICATION FUNCTIONS ====================

    def classify_density(self, img_path):
        """Classify breast tissue density from a mammogram image."""
        if not hasattr(self, 'density_model') or self.density_model is None:
            print("Density model not available")
            return "N/A", [0.0, 0.0, 0.0, 0.0]

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return "N/A", [0.0, 0.0, 0.0, 0.0]
            input_tensor = self.preprocess_for_density(img).unsqueeze(0)
            with torch.no_grad():
                output = self.density_model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                classes = ['A', 'B', 'C', 'D']
                return classes[pred], torch.softmax(output, dim=1).squeeze().tolist()
        except Exception as e:
            print(f"Error classifying density: {str(e)}")
            return "N/A", [0.0, 0.0, 0.0, 0.0]

    def classify_mass_roi(self, roi_image):
        """Classify a mass ROI as benign or malignant."""
        if self.mass_classifier is None or self.mass_preprocessor is None or self.mass_resnet is None:
            return "Unknown", 0.5

        try:
            # Preprocess image separately for classification
            _, _, preprocessed = preprocess_image_array(roi_image)

            # Extract feature representations
            hog_feat = extract_HOG(preprocessed)
            lbp_hist, ms_lbp = extract_LBP_advanced(preprocessed, use_multi_scale=True)
            shape_feat = extract_shape_features(preprocessed)
            deep_feat = extract_deep_features(roi_image, self.mass_resnet)

            # Combine all features into a single vector
            full_feature = combine_features(hog_feat, lbp_hist, ms_lbp, shape_feat, deep_feat)

            # Adjust feature vector dimensionality to match preprocessor expectation
            expected_dim = self.mass_preprocessor['n_original_features']
            if len(full_feature) > expected_dim:
                full_feature = full_feature[:expected_dim]
            elif len(full_feature) < expected_dim:
                full_feature = np.pad(full_feature, (0, expected_dim - len(full_feature)))

            # Apply preprocessing pipeline (scaler, PCA, MI selection)
            scaled = self.mass_preprocessor['scaler'].transform([full_feature])
            pca_transformed = self.mass_preprocessor['pca'].transform(scaled)
            mi_selected = pca_transformed[:, self.mass_preprocessor['mi_indices']]

            # Use either GA or RFE selector depending on which is available
            selector_key = 'ga_selector' if 'ga_selector' in self.mass_preprocessor else 'rfe_selector'
            final_feature = mi_selected[:, self.mass_preprocessor[selector_key].support_]

            # Run prediction
            prediction = self.mass_classifier.predict(final_feature)[0]
            if hasattr(self.mass_classifier, 'predict_proba'):
                proba = self.mass_classifier.predict_proba(final_feature)[0]
                confidence = max(proba)
            else:
                confidence = 0.8

            class_name = "Benign" if prediction == 0 else "Malignant"
            return class_name, confidence

        except Exception as e:
            print(f"Error classifying mass: {str(e)}")
            return "Unknown", 0.5

    def classify_calc_roi(self, roi_image):
        """Classify a calcification ROI as benign or malignant."""
        if self.calc_classifier is None or self.calc_preprocessor is None or self.calc_resnet is None:
            return "Unknown", 0.5

        try:
            # Use previously determined density level, defaulting to 'C'
            density_level = getattr(self, 'current_density_label', 'C')

            # Preprocess image with density-aware CLAHE
            processed_roi = self.preprocess_for_calc_detection(roi_image, density_level)
            _, _, preprocessed = preprocess_image_array(roi_image)

            # Extract features (shape features are not used for calcifications)
            hog_feat = extract_HOG(preprocessed)
            lbp_hist, ms_lbp = extract_LBP_advanced(preprocessed, use_multi_scale=True)
            deep_feat = extract_deep_features(roi_image, self.calc_resnet)

            # Combine features without shape features
            full_feature = combine_features(hog_feat, lbp_hist, ms_lbp, None, deep_feat)

            # Adjust feature vector dimensionality
            expected_dim = self.calc_preprocessor['n_original_features']
            if len(full_feature) > expected_dim:
                full_feature = full_feature[:expected_dim]
            elif len(full_feature) < expected_dim:
                full_feature = np.pad(full_feature, (0, expected_dim - len(full_feature)))

            # Apply preprocessing pipeline
            scaled = self.calc_preprocessor['scaler'].transform([full_feature])
            pca_transformed = self.calc_preprocessor['pca'].transform(scaled)
            mi_selected = pca_transformed[:, self.calc_preprocessor['mi_indices']]

            # Use either GA or RFE selector depending on which is available
            selector_key = 'ga_selector' if 'ga_selector' in self.calc_preprocessor else 'rfe_selector'
            final_feature = mi_selected[:, self.calc_preprocessor[selector_key].support_]

            # Run prediction
            prediction = self.calc_classifier.predict(final_feature)[0]
            if hasattr(self.calc_classifier, 'predict_proba'):
                proba = self.calc_classifier.predict_proba(final_feature)[0]
                confidence = max(proba)
            else:
                confidence = 0.8

            # Map integer prediction to class label
            prediction = int(prediction)
            if prediction == 0:
                class_name = "Benign"
            elif prediction == 1:
                class_name = "Malignant"
            else:
                class_name = "Unknown"

            return class_name, confidence

        except Exception as e:
            print(f"Error classifying calc: {str(e)}")
            return "Unknown", 0.5

    # ==================== DETECTION FUNCTIONS ====================

    def apply_nms(self, boxes, confidences, classes, threshold=0.5):
        """Apply Non-Maximum Suppression to filter overlapping bounding boxes."""
        if len(boxes) == 0:
            return [], [], []
        boxes_array = np.array(boxes)
        confidences_array = np.array(confidences)
        boxes_xywh = []
        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            boxes_xywh.append([x1, y1, w, h])
        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences_array, CONF_MASS, threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_boxes = [boxes[i] for i in indices]
            filtered_confidences = [confidences[i] for i in indices]
            filtered_classes = [classes[i] for i in indices]
            return filtered_boxes, filtered_confidences, filtered_classes
        else:
            return [], [], []

    def detect_mass_enhanced_threaded(self, image_path, raw_img):
        """
        Detect masses using an adaptive enhanced algorithm based on breast tissue density,
        followed by benign/malignant classification. Thread-safe version.

        Args:
            image_path: Path to the original image file
            raw_img: Numpy array of the image (RGB format)

        Returns:
            tuple: (processed image array, list of detection result strings)
        """
        if self.mass_model is None:
            print("Mass model not loaded")
            return raw_img, []

        # Notify progress if running inside a worker
        if hasattr(self, 'detection_worker') and self.detection_worker:
            self.detection_worker.progress_update.emit("Classifying breast density first...")

        # ==================== STEP 1: CLASSIFY BREAST DENSITY FIRST ====================
        # Density must be determined before detecting masses for adaptive preprocessing
        if not hasattr(self, 'current_density_label') or self.current_density_label is None:
            print("Breast density not yet available, classifying now...")
            density_label, _ = self.classify_density(str(image_path))
            self.current_density_label = density_label
        else:
            density_label = self.current_density_label

        print(f"\nBreast tissue density detected: {density_label}")

        # ==================== STEP 2: ADAPTIVE PREPROCESSING ====================
        if hasattr(self, 'detection_worker') and self.detection_worker:
            self.detection_worker.progress_update.emit(
                f"Applying adaptive preprocessing for density {density_label}...")

        # Generate multiple image variants to improve detection accuracy
        processed_images = []
        clahe_variants = []

        if density_label in ['A', 'B']:
            print(f"Applying low-density preprocessing strategy ({density_label})")
            # Low density group: mild CLAHE variants
            clahe_variants = [None, 1.0, 1.5, 2.0, 2.5]  # None = no CLAHE applied
        else:  # density_label in ['C', 'D']
            print(f"Applying high-density preprocessing strategy ({density_label})")
            # High density group: strong CLAHE variants
            clahe_variants = [None, 2.0, 3.0, 4.0, 5.0]  # None = no CLAHE applied

        # Build each image variant with the corresponding CLAHE setting
        for clahe_val in clahe_variants:
            if clahe_val is None:
                # Use original image without CLAHE
                processed_img = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX)
            else:
                # Apply CLAHE to the L channel in LAB color space
                lab = cv2.cvtColor(raw_img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)

                # Adjust tile grid size according to density
                if density_label in ['A', 'B']:
                    tile_size = (8, 8)   # Smaller tiles for low density
                else:
                    tile_size = (16, 16)  # Larger tiles for high density

                clahe = cv2.createCLAHE(clipLimit=clahe_val, tileGridSize=tile_size)
                cl = clahe.apply(l)
                merged = cv2.merge((cl, a, b))
                processed_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
                processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX)

            processed_images.append(processed_img)

        print(f"   Created {len(processed_images)} image variants")

        # Labels for logging each variant
        clahe_labels = [f"CLAHE {v}" if v else "Original" for v in clahe_variants]

        # ==================== STEP 3: RUN DETECTION ON ALL VARIANTS ====================
        all_boxes = []
        all_confidences = []
        all_classes = []
        detection_results = []

        self.detected_masses = []  # Reset detected masses list

        print("\nMASS DETECTION RESULTS (Adaptive Enhanced):")

        for i, (img, label) in enumerate(zip(processed_images, clahe_labels)):
            if hasattr(self, 'detection_worker') and self.detection_worker:
                self.detection_worker.progress_update.emit(f"Detecting masses on {label}...")

            # Run YOLO inference
            results = self.mass_model(img, verbose=False, conf=CONF_MASS)

            if results[0].boxes:
                print(f"\n{label}:")
                for j, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Accumulate detections for NMS
                    all_boxes.append([x1, y1, x2, y2])
                    all_confidences.append(conf)
                    all_classes.append(cls_id)

                    detection_results.append(
                        f"Mass {cls_id}: Confidence {conf:.2%}, Box ({x1}, {y1}, {x2}, {y2})"
                    )
                    print(f"   Mass {j + 1}: Class {cls_id} | Confidence: {conf:.2%} | Box: ({x1}, {y1}, {x2}, {y2})")

        # ==================== STEP 4: APPLY NON-MAXIMUM SUPPRESSION ====================
        if all_boxes:
            if hasattr(self, 'detection_worker') and self.detection_worker:
                self.detection_worker.progress_update.emit("Applying Non-Maximum Suppression...")

            filtered_boxes, filtered_confidences, filtered_classes = self.apply_nms(
                all_boxes, all_confidences, all_classes, NMS_THRESHOLD
            )

            print(f"\nAfter NMS (threshold={NMS_THRESHOLD}):")
            print(f"   Before NMS: {len(all_boxes)} boxes")
            print(f"   After NMS: {len(filtered_boxes)} boxes")

            # ==================== STEP 5: CLASSIFY EACH DETECTED MASS ====================
            final_detection_results = []

            for i, (box, conf, cls_id) in enumerate(zip(filtered_boxes, filtered_confidences, filtered_classes)):
                if hasattr(self, 'detection_worker') and self.detection_worker:
                    self.detection_worker.progress_update.emit(f"Classifying mass {i + 1}/{len(filtered_boxes)}...")

                x1, y1, x2, y2 = box

                # Crop the ROI from the original image with padding
                roi, actual_bbox = crop_roi_from_original(
                    self.original_image, (x1, y1, x2, y2), ROI_PADDING_MASS
                )

                # Classify the mass ROI
                classification, class_conf = self.classify_mass_roi(roi)

                # Store mass detection info
                mass_info = {
                    'roi': roi,
                    'bbox': (x1, y1, x2, y2),
                    'actual_bbox': actual_bbox,
                    'confidence': conf,
                    'class_id': cls_id,
                    'classification': classification,
                    'classification_confidence': class_conf,
                    'density_level': density_label  # Store the density level for reference
                }
                self.detected_masses.append(mass_info)

                final_detection_results.append(
                    f"Final Mass {i + 1}: {classification} {class_conf:.2%}, Box ({x1}, {y1}, {x2}, {y2})"
                )
                print(
                    f"   Final Mass {i + 1}: {classification} | Confidence: {class_conf:.2%} | Box: ({x1}, {y1}, {x2}, {y2})")

            print(f"\nTotal: Detected {len(self.detected_masses)} masses at density {density_label}")
            return processed_images[-1], final_detection_results

        else:
            print(f"No masses detected at density {density_label}")
            return processed_images[-1], []

    def smart_sliding_windows(self, img_shape, window_size, stride, max_windows=MAX_WINDOWS):
        """Generate sliding window coordinates with adaptive stride for large images."""
        h, w = img_shape[:2]
        adaptive_stride = window_size if max(h, w) > 2000 else stride
        windows = []
        for y in range(0, h - window_size + 1, adaptive_stride):
            for x in range(0, w - window_size + 1, adaptive_stride):
                windows.append((x, y, x + window_size, y + window_size))
                if len(windows) >= max_windows:
                    return windows
        return windows

    def process_window_batch(self, model, img, window_batch, confidence_thresh):
        """Run YOLO inference on a batch of sliding windows."""
        batch_images = []
        batch_coords = []
        for x1, y1, x2, y2 in window_batch:
            window = img[y1:y2, x1:x2]
            window = cv2.resize(window, (WINDOW_SIZE, WINDOW_SIZE))
            batch_images.append(window)
            batch_coords.append((x1, y1, x2, y2))
        results = model(batch_images, verbose=False, conf=confidence_thresh)
        return results, batch_coords

    def detect_calc_with_sliding_threaded(self, image_path, raw_img):
        """
        Detect calcifications using sliding window inference, followed by
        benign/malignant classification. Thread-safe version.
        """
        if self.calc_model is None:
            print("Calc model not loaded")
            return raw_img, []

        if hasattr(self, 'detection_worker') and self.detection_worker:
            self.detection_worker.progress_update.emit("Preprocessing image for calcification detection...")

        # Step 1: Apply density-aware CLAHE preprocessing
        density_label = getattr(self, 'current_density_label', 'C')  # Default to 'C' if not set
        img = self.preprocess_for_calc_detection(raw_img, density_label)

        if hasattr(self, 'detection_worker') and self.detection_worker:
            self.detection_worker.progress_update.emit("Creating sliding windows...")

        # Step 2: Generate sliding window grid
        windows = self.smart_sliding_windows(img.shape, WINDOW_SIZE, STRIDE)
        detection_results = []

        self.detected_calcs = []  # Reset detected calcs list

        print("\nCALCIFICATION DETECTION RESULTS:")
        if not windows:
            print("Warning: No sliding windows could be created.")
            return img, []

        total_batches = (len(windows) + BATCH_SIZE - 1) // BATCH_SIZE

        # Step 3: Process windows in batches with progress updates
        for batch_idx in range(0, len(windows), BATCH_SIZE):
            current_batch = (batch_idx // BATCH_SIZE) + 1
            if hasattr(self, 'detection_worker') and self.detection_worker:
                self.detection_worker.progress_update.emit(
                    f"Processing sliding window batch {current_batch}/{total_batches}..."
                )

            batch = windows[batch_idx:batch_idx + BATCH_SIZE]
            results, coords = self.process_window_batch(self.calc_model, img, batch, CONF_CALC)

            for result, (x1, y1, x2, y2) in zip(results, coords):
                if result.boxes:  # Check if any detections exist in this window
                    for box in result.boxes:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                        bx1 += x1; bx2 += x1
                        by1 += y1; by2 += y1
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])

                        # Step 4: Crop ROI from original image with padding for classification
                        roi, actual_bbox = crop_roi_from_original(
                            self.original_image, (bx1, by1, bx2, by2), ROI_PADDING_CALC
                        )

                        # Step 5: Classify the calcification ROI
                        classification, class_conf = self.classify_calc_roi(roi)

                        # Store calcification detection info
                        calc_info = {
                            'roi': roi,
                            'bbox': (bx1, by1, bx2, by2),
                            'actual_bbox': actual_bbox,
                            'confidence': conf,
                            'class_id': cls_id,
                            'classification': classification,
                            'classification_confidence': class_conf
                        }
                        self.detected_calcs.append(calc_info)

                        detection_results.append(
                            f"Calc {cls_id}: {classification} {class_conf:.2%}, Box ({bx1}, {by1}, {bx2}, {by2})"
                        )
                        print(f"Calc: {classification} | Confidence: {class_conf:.2%} | Box: ({bx1}, {by1}, {bx2}, {by2})")

        if not detection_results:
            print("No calcifications detected.")

        return img, detection_results

    # ==================== DISPLAY FUNCTIONS ====================

    def on_bbox_display_changed(self):
        """Handle bounding box display mode changes triggered by radio buttons."""
        if hasattr(self, 'base_image') and self.base_image is not None:
            # Rebuild the display image with the updated display mode
            self.update_display_image()

    def get_display_mode(self):
        """Return the currently selected bounding box display mode."""
        if self.ui.radio_show_all.isChecked():
            return "all"
        elif self.ui.radio_show_mass.isChecked():
            return "mass"
        elif self.ui.radio_show_calc.isChecked():
            return "calc"
        else:
            return "none"

    def update_display_image(self):
        """Rebuild and refresh the output image according to the current display mode."""
        if not hasattr(self, 'base_image') or self.base_image is None:
            return

        # Work on a copy of the base image
        display_img = self.base_image.copy()
        display_mode = self.get_display_mode()

        # Draw bounding boxes based on selected mode
        if display_mode in ["all", "mass"]:
            self.draw_mass_bboxes(display_img)

        if display_mode in ["all", "calc"]:
            self.draw_calc_bboxes(display_img)

        # Overlay density label text
        density_text = self.density_label.text().replace("Breast tissue density: ", "")
        display_img = self.overlay_density_label(display_img, density_text)

        # Update the pixmap shown in the output label
        self.update_pixmap_from_numpy(display_img)

    def draw_mass_bboxes(self, img):
        """Draw bounding boxes for all detected masses on the given image."""
        for mass_info in self.detected_masses:
            x1, y1, x2, y2 = mass_info['bbox']
            classification = mass_info['classification']
            class_conf = mass_info['classification_confidence']

            # Choose color based on classification result
            if classification == "Benign":
                color = (0, 0, 255)    # Blue for benign
            elif classification == "Malignant":
                color = (255, 0, 0)    # Red for malignant
            else:
                color = (0, 255, 0)    # Green for unknown

            label = f"Mass: {classification} {class_conf:.2%}"

            # Adjust coordinates to account for display padding offset
            display_x1, display_y1, display_x2, display_y2 = adjust_bbox_coordinates(
                (x1, y1, x2, y2), self.padding_offset
            )

            # Draw the bounding box with label
            self.draw_enhanced_bbox(img, display_x1, display_y1,
                                    display_x2, display_y2, label, color, 6)

    def draw_calc_bboxes(self, img):
        """Draw bounding boxes for all detected calcifications on the given image."""
        for calc_info in self.detected_calcs:
            x1, y1, x2, y2 = calc_info['bbox']
            classification = calc_info['classification']
            class_conf = calc_info['classification_confidence']

            # Choose color based on classification result
            if classification == "Benign":
                color = (40, 0, 60)     # Purple for benign calcification
            elif classification == "Malignant":
                color = (255, 0, 0)     # Red for malignant calcification
            else:
                color = (255, 0, 255)   # Magenta for unknown calcification

            label = f"Calc: {classification} {class_conf:.2%}"

            # Adjust coordinates to account for display padding offset
            display_x1, display_y1, display_x2, display_y2 = adjust_bbox_coordinates(
                (x1, y1, x2, y2), self.padding_offset
            )

            # Draw the bounding box with label
            self.draw_enhanced_bbox(img, display_x1, display_y1,
                                    display_x2, display_y2, label, color, 6)

    def draw_enhanced_bbox(self, img, x1, y1, x2, y2, label, color, thickness=6):
        """Draw a thick bounding box with a filled background label for readability."""
        # Draw the rectangle border
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Compute text background position
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = x1
        text_y = y1 - 15

        # If the text would go above the image, move it below the box instead
        if text_y - text_size[1] < 0:
            text_y = y2 + text_size[1] + 15

        # Draw filled rectangle as background for the text
        cv2.rectangle(img, (text_x, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 10, text_y + 5), color, -1)

        # Draw label text in white over the colored background
        cv2.putText(img, label, (text_x + 5, text_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    def overlay_density_label(self, img_np, density_label):
        """Overlay the breast density label onto the image, accounting for padding offset."""
        text = f"Density: {density_label}"
        # Position text inside the padded content area
        text_x = 30 + getattr(self, 'padding_offset', 0)
        text_y = 60 + getattr(self, 'padding_offset', 0)

        # Draw white outline for contrast, then draw colored text on top
        cv2.putText(img_np, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 8)
        cv2.putText(img_np, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        return img_np

    def update_pixmap_from_numpy(self, img_np):
        """Convert a numpy image array to a QPixmap and update the output label."""
        try:
            # Convert to QImage depending on number of channels
            if len(img_np.shape) == 3:
                height, width, channel = img_np.shape
                bytes_per_line = 3 * width
                q_image = QImage(img_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                height, width = img_np.shape
                bytes_per_line = width
                q_image = QImage(img_np.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            if q_image.isNull():
                raise ValueError("Invalid QImage!")

            self.original_pixmap = QPixmap.fromImage(q_image)
            if self.original_pixmap.isNull():
                raise ValueError("Invalid QPixmap!")

            self.update_output_label()

        except Exception as e:
            print(f"Error updating pixmap: {str(e)}")

    def update_classification_summary(self):
        """Update the classification summary label with benign/malignant counts."""
        mass_benign = sum(1 for m in self.detected_masses if m.get('classification') == 'Benign')
        mass_malignant = sum(1 for m in self.detected_masses if m.get('classification') == 'Malignant')

        calc_benign = sum(1 for c in self.detected_calcs if c.get('classification') == 'Benign')
        calc_malignant = sum(1 for c in self.detected_calcs if c.get('classification') == 'Malignant')

        summary_parts = []
        if self.detected_masses:
            summary_parts.append(f"Mass: {mass_benign}B/{mass_malignant}M")
        if self.detected_calcs:
            summary_parts.append(f"Calc: {calc_benign}B/{calc_malignant}M")

        if summary_parts:
            summary_text = " | ".join(summary_parts)
        else:
            summary_text = "No detections"

        self.classification_summary_label.setText(f"Classification: {summary_text}")

    # ==================== FILE I/O FUNCTIONS ====================

    def open_dicom(self):
        """Open and convert a DICOM file to PNG for display and detection."""
        file_dialog = QFileDialog()
        dicom_path, _ = file_dialog.getOpenFileName(self, "Open DICOM File", "File Folder Dicom Data", "DICOM Files (*.dcm)")
        if dicom_path:
            try:
                ds = pydicom.dcmread(dicom_path, force=True)
                pixel_data = ds.pixel_array
                normalized_data = ((pixel_data - pixel_data.min()) / (pixel_data.max() - pixel_data.min()) * 255).astype('uint8')

                # Resolve output directory based on execution context
                if getattr(sys, 'frozen', False):
                    # Running from a compiled executable
                    current_dir = Path(sys.executable).parent
                else:
                    # Running from source script
                    current_dir = Path(__file__).parent

                dicom_folder_name = f'{datetime.now().strftime("Date_%Y_%m_%d_Time_%H_%M_%S")}_{Path(dicom_path).stem}'
                save_folder = current_dir / 'autosave_dicom_convert_jpg_folder' / dicom_folder_name
                save_folder.mkdir(parents=True, exist_ok=True)
                png_path = save_folder / f'{dicom_folder_name}.png'
                cv2.imwrite(str(png_path), normalized_data)

                # Display the converted image
                pixmap = QPixmap(str(png_path))
                self.input_label.setPixmap(pixmap.scaled(self.input_label.size(), Qt.KeepAspectRatio))
                self.input_label.setAlignment(Qt.AlignCenter)
                self.opt_source = Path(png_path)

                # Store the original image for use during detection
                self.original_image = cv2.imread(str(png_path))

                print(f"DICOM file converted and saved to: {png_path}")
                print(f"Original size preserved: {normalized_data.shape}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error loading DICOM file:\n{str(e)}")
                print(f"Error details: {str(e)}")

    def browser_file(self):
        """Open a file browser to select an image file."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "File Folder Image Data",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options
        )
        if file_name:
            pixmap = QPixmap(file_name)
            self.input_label.setPixmap(pixmap.scaled(self.input_label.size(), Qt.KeepAspectRatio))
            self.input_label.setAlignment(Qt.AlignCenter)
            self.directImage.setText(file_name)
            self.opt_source = Path(file_name)

    # ==================== ZOOM AND PAN FUNCTIONS ====================

    def update_output_label(self):
        """Refresh the output label with the current zoom level and pan position."""
        if self.original_pixmap and not self.original_pixmap.isNull():
            scaled_size = self.original_pixmap.size() * self.zoom_factor
            scaled_pixmap = self.original_pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label_size = self.output_label.size()

            if scaled_size.width() <= label_size.width() and scaled_size.height() <= label_size.height():
                self.output_label.setPixmap(scaled_pixmap)
                self.output_label.setAlignment(Qt.AlignCenter)
            else:
                max_x_offset = max(0, scaled_size.width() - label_size.width())
                max_y_offset = max(0, scaled_size.height() - label_size.height())
                pan_x = max(0, min(self.pan_offset.x(), max_x_offset))
                pan_y = max(0, min(self.pan_offset.y(), max_y_offset))
                self.pan_offset = QPoint(pan_x, pan_y)
                crop_rect = QRect(pan_x, pan_y, label_size.width(), label_size.height())
                cropped_pixmap = scaled_pixmap.copy(crop_rect)
                self.output_label.setPixmap(cropped_pixmap)
                self.output_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def wheel_event(self, event):
        """Handle mouse wheel events to zoom the output image in or out."""
        if self.original_pixmap and not self.original_pixmap.isNull():
            cursor_pos = event.pos()
            old_zoom = self.zoom_factor
            if event.angleDelta().y() > 0:
                self.zoom_factor = min(self.zoom_factor * (1 + ZOOM_STEP), MAX_ZOOM)
            else:
                self.zoom_factor = max(self.zoom_factor * (1 - ZOOM_STEP), MIN_ZOOM)

            if old_zoom != self.zoom_factor:
                zoom_ratio = self.zoom_factor / old_zoom
                new_pan_x = self.pan_offset.x() * zoom_ratio + cursor_pos.x() * (zoom_ratio - 1)
                new_pan_y = self.pan_offset.y() * zoom_ratio + cursor_pos.y() * (zoom_ratio - 1)
                self.pan_offset = QPoint(int(new_pan_x), int(new_pan_y))
                self.update_output_label()

    def mouse_press_event(self, event):
        """Handle mouse button press to begin panning the image."""
        if event.button() == Qt.LeftButton and self.original_pixmap and not self.original_pixmap.isNull():
            self.is_panning = True
            self.last_pan_point = event.pos()
            self.output_label.setCursor(QCursor(Qt.ClosedHandCursor))

    def mouse_move_event(self, event):
        """Handle mouse movement to pan the image when dragging."""
        if self.is_panning and self.original_pixmap and not self.original_pixmap.isNull():
            delta = event.pos() - self.last_pan_point
            new_pan_x = self.pan_offset.x() - delta.x()
            new_pan_y = self.pan_offset.y() - delta.y()
            self.pan_offset = QPoint(new_pan_x, new_pan_y)
            self.last_pan_point = event.pos()
            self.update_output_label()
        else:
            if self.original_pixmap and not self.original_pixmap.isNull():
                self.output_label.setCursor(QCursor(Qt.OpenHandCursor))

    def mouse_release_event(self, event):
        """Handle mouse button release to end panning."""
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            if self.original_pixmap and not self.original_pixmap.isNull():
                self.output_label.setCursor(QCursor(Qt.OpenHandCursor))

    # ==================== SAVE FUNCTIONS ====================

    def show_save_dialog(self):
        """Show a save dialog to save the result image to a fixed output folder."""
        if hasattr(self, 'base_image') and self.base_image is not None:
            # Fixed output directory
            save_dir = application_path / "File Folder Cancer Image"
            save_dir.mkdir(exist_ok=True)

            # Default filename based on source image name
            default_name = f"{self.opt_source.stem}.jpg" if hasattr(self, 'opt_source') else "result.jpg"

            # Prompt user to select the save location and filename
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Result", str(save_dir / default_name),
                "Image Files (*.jpg *.jpeg *.png *.bmp)"
            )

            if file_path:
                self.save_image(file_path)
        else:
            QMessageBox.warning(self, "Error", "Please run detection first.")

    def save_image(self, file_path):
        """Save the result image to disk according to the current display mode."""
        if hasattr(self, 'base_image') and self.base_image is not None:
            try:
                # Build the save image from the base image and current display settings
                save_img = self.base_image.copy()
                display_mode = self.get_display_mode()

                # Draw bounding boxes according to display mode
                if display_mode in ["all", "mass"]:
                    self.draw_mass_bboxes(save_img)

                if display_mode in ["all", "calc"]:
                    self.draw_calc_bboxes(save_img)

                # Overlay the density label
                density_text = self.density_label.text().replace("Breast tissue density: ", "")
                save_img = self.overlay_density_label(save_img, density_text)

                # Write image to disk
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                save_img_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, save_img_bgr)

                QMessageBox.information(self, "File save", f"Result image saved at: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error saving image:\n{str(e)}")
        else:
            QMessageBox.warning(self, "Error", "Please select an image and run detection before saving.")

    # ==================== UTILITY FUNCTIONS ====================

    def clear_memory(self):
        try:
            # Clear PyTorch GPU cache if CUDA is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Run garbage collector multiple times for thorough cleanup
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)

            print("Memory cleared")
        except Exception as e:
            print(f"Warning during memory cleanup: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = DetectImage()
    main_win.show()
    sys.exit(app.exec_())
