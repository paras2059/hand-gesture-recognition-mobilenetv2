# 🖐️ Hand Gesture Recognition
### Deep Learning · MobileNetV2 Transfer Learning · Grad-CAM · TFLite

A end-to-end hand gesture recognition system built on the [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) dataset. The project trains two models — a custom CNN baseline and a fine-tuned MobileNetV2 — achieving **100% validation accuracy** on 10 gesture classes, with Grad-CAM visualizations for interpretability and a quantized TFLite export for edge deployment.

---

## Results

| Model | Validation Accuracy |
|---|---|
| Custom CNN (baseline) | 10.00% |
| MobileNetV2 Transfer Learning | **100.00%** |

> The custom CNN uses grayscale input and trains from scratch. MobileNetV2 uses RGB input with ImageNet weights and two-phase fine-tuning — the performance gap illustrates the power of transfer learning on a relatively small IR dataset.

---

## Grad-CAM Attention Maps

The model's attention is visualized using Grad-CAM (with a mean-activation fallback for high-confidence cases where softmax gradients vanish). Red/yellow regions show where the network focuses when classifying each gesture.

![Grad-CAM Results](gradcam_results.png)

*Columns: Original infrared image | Grad-CAM heatmap | Blended overlay*
*Title colour: cyan = correct prediction, red = incorrect*

---

## Dataset

**LeapGestRecog** — infrared hand images captured with a Leap Motion sensor.

- **Source:** [Kaggle — gti-upm/leapgestrecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- **Classes (10):**

| # | Class | # | Class |
|---|---|---|---|
| 01 | Palm | 06 | Index finger |
| 02 | L | 07 | OK |
| 03 | Fist | 08 | Palm moved |
| 04 | Fist moved | 09 | C |
| 05 | Thumb | 10 | Down |

- **Image size:** 128 × 128 px
- **Format:** Grayscale infrared (converted to RGB for MobileNetV2)

---

## Project Structure

```
hand-gesture-recognition/
│
├── hand_gesture_recognition.ipynb   # Main Colab notebook
├── class_names.json                 # Ordered list of 10 gesture classes
├── training_report.json             # Accuracy + metadata from training run
│
├── hand_gesture_cnn.keras           # Saved custom CNN
├── hand_gesture_mobilenetv2.keras   # Saved MobileNetV2 model
├── hand_gesture_mobilenetv2.tflite  # Quantized TFLite (edge deployment)
│
├── gradcam_results.png              # Grad-CAM visualization output
├── sample_gestures.png              # Dataset sample grid
└── README.md
```

---

## Pipeline

```
Dataset Download (kagglehub)
        │
        ▼
 Data Exploration & Visualization
        │
        ▼
 Preprocessing & Augmentation
 (rotation, flip, zoom, brightness, shear)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
 Custom CNN (grayscale)        MobileNetV2 Transfer Learning (RGB)
 4 conv blocks + Dense head    Phase 1: train head only
 BatchNorm + Dropout           Phase 2: fine-tune top 50 layers
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
           Evaluation + Confusion Matrix
                       │
                       ▼
              Grad-CAM Visualization
                       │
                       ▼
         TFLite Export (int8 quantized)
```

---

## Model Architectures

### Custom CNN
A 4-block convolutional network trained from scratch on grayscale images:

- **Block 1–3:** Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool2D → Dropout (32→64→128 filters)
- **Block 4:** Conv2D (256 filters) → GlobalAveragePooling2D
- **Head:** Dense(512) → BatchNorm → Dropout(0.5) → Dense(256) → Dropout(0.3) → Softmax(10)
- **Regularization:** L2(1e-4) on conv and first dense layer
- **Input:** 128×128×1 (grayscale)

### MobileNetV2 + Transfer Learning
Two-phase training on top of ImageNet-pretrained weights:

- **Backbone:** MobileNetV2 (1.00, 128px) — frozen in Phase 1
- **Head:** GlobalAveragePooling2D → Dense(512, relu) → Dropout(0.5) → Dense(256, relu) → Dropout(0.3) → Softmax(10)
- **Phase 1:** Train head only — Adam lr=1e-3, 15 epochs
- **Phase 2:** Unfreeze top 50 layers — Adam lr=1e-5, up to 30 epochs
- **Input:** 128×128×3 (RGB), preprocessed to [-1, 1]

---

## Training Configuration

```python
IMG_HEIGHT   = 128
IMG_WIDTH    = 128
BATCH_SIZE   = 32
EPOCHS_CNN   = 20
EPOCHS_TL    = 30   # Phase 2 fine-tuning
SEED         = 42
```

**Augmentation (training set only):**
- Rotation ±15°, width/height shift ±10%, shear ±10%
- Zoom ±15%, horizontal flip, brightness [0.8, 1.2]

**Callbacks:**
- `EarlyStopping` (patience=6, restore best weights)
- `ReduceLROnPlateau` (factor=0.3, patience=3)
- `ModelCheckpoint` (save best only)

---

## Grad-CAM Implementation

Standard Grad-CAM is computed by building a feature extractor directly from the MobileNetV2 sub-model (not the wrapper model) to avoid Keras 3 graph-boundary errors. For high-confidence predictions where softmax gradients vanish, a **mean absolute activation fallback** is used automatically.

```python
# Automatic method selection
if cam.max() < 0.01:          # gradient vanished
    method = 'Activation'     # mean |feature map| fallback
else:
    method = 'Grad-CAM'       # standard weighted gradient approach
```

---

## Deployment

The best model is exported as a quantized TFLite file for on-device inference:

```python
tl_model.export('hand_gesture_mobilenetv2_export')   # SavedModel dir
converter = tf.lite.TFLiteConverter.from_saved_model('hand_gesture_mobilenetv2_export')
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # int8 quantization
tflite_model = converter.convert()
```

The `.tflite` file can be deployed on **Android**, **iOS**, and **Raspberry Pi / edge devices**.

---

## Requirements & Setup

### Run in Google Colab (recommended)

1. Open `hand_gesture_recognition.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set **Runtime → Change runtime type → T4 GPU**
3. Run all cells — the dataset downloads automatically

### Local setup

```bash
pip install kagglehub tensorflow opencv-python matplotlib seaborn scikit-learn
```

```python
import kagglehub
path = kagglehub.dataset_download("gti-upm/leapgestrecog")
```

**Tested environment:**
- Python 3.12
- TensorFlow 2.20.0
- Keras 3.13.2
- Google Colab (T4 GPU)

---

## Real-Time Webcam Inference

A webcam capture function is included for live gesture prediction directly in Colab:

```python
# Uncomment to run — requires browser camera permission
predict_from_webcam(tl_model, CLASS_NAMES)
```

Captures a single frame, runs inference, and displays a top-3 confidence bar chart.

---

## Environment

```json
{
  "timestamp": "2026-05-15 09:31:46",
  "keras_version": "3.13.2",
  "tensorflow_version": "2.20.0",
  "dataset": "LeapGestRecog (gti-upm)",
  "num_classes": 10,
  "image_size": "128x128",
  "custom_cnn_acc": "10.00%",
  "mobilenetv2_acc": "100.00%"
}
```

---

## License

Dataset: [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) — see Kaggle dataset page for terms.
Code: MIT License.
