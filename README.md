# chicken_detection_and_tracking
# 🐔 Chicken Detection and Tracking using YOLO

A computer vision and deep learning project for **chicken detection and tracking** using the **YOLO (You Only Look Once)** object detection framework.

The model was trained using an annotated dataset and can be used to detect chickens in images and videos. The project also includes a **Streamlit-based user interface** for interacting with the trained YOLO model.

---

## 📌 Project Overview

The purpose of this project is to develop an AI-based computer vision system capable of detecting chickens in images and videos.

The dataset was annotated with bounding boxes around chickens and prepared in YOLO format. The dataset was divided into training, validation, and testing sets and then used to train a YOLO model.

After training, the model was tested on new images and videos to identify chickens and display their locations using bounding boxes and confidence scores.

The project also includes a web-based interface built with **Streamlit**, allowing users to upload an image and run chicken detection using the trained model.

---

## 🎯 Project Objectives

The main objectives of this project are:

* Detect chickens automatically using deep learning.
* Train a YOLO object detection model.
* Use an annotated dataset for model training.
* Perform chicken detection on images.
* Perform chicken detection on videos.
* Display bounding boxes around detected chickens.
* Display confidence scores for detections.
* Build a simple and user-friendly web interface.
* Deploy the trained model as a web application.

---

## 🧠 Technologies Used

* **Python**
* **YOLO**
* **Ultralytics**
* **PyTorch**
* **OpenCV**
* **Roboflow**
* **Google Colab**
* **Streamlit**
* **GitHub**

---

## 📊 Dataset

The project uses an **annotated chicken detection dataset**.

The images were annotated using bounding boxes around chickens. The annotations provide the information required by the YOLO model to learn how to detect chickens.

The dataset was organized into three main parts:

```text
Train
Validation
Test
```

The YOLO dataset structure is:

```text
dataset/
│
├── data.yaml
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

### Annotation

Each chicken in an image is marked using a bounding box.

The annotation contains information about:

* Object class
* Bounding box position
* Bounding box width
* Bounding box height

These annotations are used during YOLO training.

---

# 🚀 Model Training

The model was trained using the **Ultralytics YOLO framework** in **Google Colab**.

During training, YOLO generated model weights including:

```text
best.pt
last.pt
```

The main model used for inference is:

```text
best.pt
```

The trained model was generated at:

```text
/content/runs/detect/train/weights/best.pt
```

The training output structure is:

```text
runs/
└── detect/
    └── train/
        └── weights/
            ├── best.pt
            └── last.pt
```

### Best Model

The `best.pt` file is used as the trained model for performing object detection.

---

# 🔍 Chicken Detection

The trained YOLO model can detect chickens in new images.

Example Python code:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="chicken.jpg",
    conf=0.25
)
```

The model produces:

* Bounding boxes
* Class labels
* Confidence scores

For example:

```text
Chicken — 92%
Chicken — 87%
Chicken — 81%
```

---

# 🎥 Video Detection

The trained model can also be used to detect chickens in videos.

Example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="chicken_video.mp4",
    save=True
)
```

The processed video is automatically saved by YOLO in the prediction output directory.

For example:

```text
runs/
└── detect/
    └── predict/
        └── output_video.mp4
```

---

# 🐔 Chicken Tracking

The project is designed for **chicken detection and tracking**.

The general tracking workflow is:

```text
Input Video
     ↓
Video Frames
     ↓
YOLO Detection
     ↓
Chicken Bounding Boxes
     ↓
Object Tracking
     ↓
Chicken Identification
     ↓
Tracking Across Frames
```

Tracking can be used to follow chickens as they move through consecutive video frames.

---

# 🖥️ Streamlit Web Interface

The project includes a **Streamlit-based user interface**.

The UI provides a simple way for users to interact with the trained YOLO model.

### Current UI functionality

* Upload an image
* Run chicken detection
* Display the original image
* Display the detection result
* Display detected objects
* Display confidence scores

The application uses:

```python
model = YOLO("best.pt")
```

to load the trained model.

### UI Workflow

```text
User
 ↓
Upload Image
 ↓
Streamlit Interface
 ↓
Trained YOLO Model
 ↓
Object Detection
 ↓
Bounding Boxes
 ↓
Confidence Scores
 ↓
Detection Result
```

---

# ⚙️ Installation

Clone the repository using Git:

```bash
git clone https://github.com/abdulwahaboffical210/chicken_detection_and_tracking.git
```

Move into the project directory:

```bash
cd chicken_detection_and_tracking
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

The main Python libraries used in this project are:

```text
streamlit
ultralytics
opencv-python-headless
pillow
numpy
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

After installing the dependencies, run:

```bash
streamlit run app.py
```

Streamlit will start the web application and provide a local URL.

Open the URL in a web browser to use the application.

---

# 🧪 Prediction

The YOLO model uses a confidence threshold to filter detections.

Example:

```python
results = model.predict(
    source="image.jpg",
    conf=0.25
)
```

The value:

```text
conf=0.25
```

means that detections with confidence below 25% are filtered out.

The confidence score represents how confident the model is that a detected object belongs to a particular class.

---

# 📈 Model Evaluation

Object detection models can be evaluated using several performance metrics.

### Precision

Precision measures the proportion of predicted objects that are correct.

```text
Precision =
True Positives /
(True Positives + False Positives)
```

### Recall

Recall measures the proportion of actual objects that were successfully detected.

```text
Recall =
True Positives /
(True Positives + False Negatives)
```

### Mean Average Precision

mAP (Mean Average Precision) is commonly used to evaluate object detection models.

Important YOLO metrics include:

```text
mAP@50
mAP@50-95
Precision
Recall
```

---

# 📂 Project Structure

The recommended GitHub repository structure is:

```text
chicken_detection_and_tracking/
│
├── app.py
├── best.pt
├── requirements.txt
├── README.md
│
├── images/
│   ├── input/
│   └── results/
│
└── notebooks/
    └── YOLO_Training.ipynb
```

### File Description

| File                  | Description                      |
| --------------------- | -------------------------------- |
| `app.py`              | Streamlit web application        |
| `best.pt`             | Trained YOLO model               |
| `requirements.txt`    | Required Python libraries        |
| `README.md`           | Project documentation            |
| `YOLO_Training.ipynb` | Training and prediction notebook |
| `images/`             | Input and output examples        |

---

# 🔧 Complete Project Workflow

The complete project workflow is:

```text
1. Dataset Collection
        ↓
2. Image Annotation
        ↓
3. Dataset Preparation
        ↓
4. Train / Validation / Test Split
        ↓
5. YOLO Dataset Configuration
        ↓
6. Model Training
        ↓
7. Model Validation
        ↓
8. Image Detection
        ↓
9. Video Detection
        ↓
10. Chicken Tracking
        ↓
11. Streamlit UI
        ↓
12. Deployment
```

---

# 🌐 Deployment

The Streamlit application can be deployed as a web application.

The deployment workflow is:

```text
Project Files
     ↓
GitHub Repository
     ↓
Streamlit Deployment Platform
     ↓
Install Requirements
     ↓
Load best.pt
     ↓
Run app.py
     ↓
Public Web Application
```

The GitHub repository for this project is:

**Repository:** `chicken_detection_and_tracking`

**GitHub Username:** `abdulwahaboffical210`

---

# 📚 Learning Outcomes

This project provided practical experience in:

* Computer Vision
* Object Detection
* Object Tracking
* Deep Learning
* YOLO
* Dataset Annotation
* Dataset Preparation
* Image Processing
* Video Processing
* Python
* PyTorch
* OpenCV
* Streamlit
* Google Colab
* GitHub
* Machine Learning Deployment

---

# 💡 Future Improvements

Future versions of the project can include:

* Real-time webcam detection
* Real-time chicken tracking
* Chicken counting
* Individual chicken IDs
* Video upload through Streamlit
* Downloadable processed videos
* Confidence threshold slider
* Detection history
* FPS monitoring
* Improved tracking
* Model performance visualization
* Cloud deployment
* Mobile-friendly interface

---

# 👨‍💻 Author

## Abdul Wahab

Computer Science Student
Aspiring AI Engineer & Data Scientist

### GitHub

**Username:** `abdulwahaboffical210`

**Repository:** `chicken_detection_and_tracking`

---

# ⭐ Acknowledgements

This project uses the following technologies and resources:

* **Ultralytics YOLO**
* **Roboflow**
* **Google Colab**
* **Streamlit**
* **PyTorch**
* **OpenCV**

---

# 📄 License

This project is developed for **educational and academic purposes**.

If the dataset is obtained from an external source, its original license and usage terms should be followed.
