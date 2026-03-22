# 🧠 Brain Tumor Classification using CNN

A Deep Learning project that classifies brain tumors from MRI images into four categories using a custom Convolutional Neural Network (CNN). The model was **trained separately on Google Colab** and the trained model (`brain_tumor_model.keras`) is included in this repository for direct use.

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Model Training on Colab](#model-training-on-colab)
- [Results](#results)
- [Technologies Used](#technologies-used)

---

## 📖 About the Project

This project uses a **CNN-based deep learning model** to classify brain MRI images into four categories:

| Class | Description |
|-------|-------------|
| **Glioma** | A type of tumor that occurs in the brain and spinal cord |
| **Meningioma** | A tumor that arises from the meninges (membranes surrounding the brain) |
| **Pituitary** | A tumor that forms in the pituitary gland |
| **No Tumor** | Normal brain MRI with no tumor detected |

The model was **trained separately on Google Colab** using GPU acceleration and the trained model file (`brain_tumor_model.keras`) was downloaded and included in the `outputs/` folder of this repository.

---

## 📊 Dataset

The dataset contains MRI images organized into **Training** and **Testing** folders, each with 4 subfolders representing the tumor classes:

```
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

- **Image Size**: 224 × 224 pixels (resized during preprocessing)
- **Data Augmentation**: Rotation, shifting, shearing, zooming, horizontal flip
- **Train/Validation Split**: 80/20 split on the training data

---

## 🏗️ Model Architecture

Custom CNN with **4 convolutional blocks** followed by a fully connected classifier:

```
Input (224×224×3)
    │
    ├── Block 1: Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ├── Block 2: Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ├── Block 3: Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ├── Block 4: Conv2D(256) → BatchNorm → MaxPool → Dropout(0.25)
    │
    ├── Flatten
    ├── Dense(512) → BatchNorm → Dropout(0.5)
    └── Dense(4, softmax) → Output
```

- **Optimizer**: Adam (lr = 0.001)
- **Loss**: Categorical Crossentropy
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

---

## 📁 Project Structure

```
Brain_Tumor/
│
├── Backend/                        # Core ML pipeline
│   ├── config.py                   # Configuration (paths, hyperparameters, classes)
│   ├── data_preprocessing.py       # Data generators with augmentation
│   ├── model.py                    # CNN model architecture
│   ├── train.py                    # Training pipeline with callbacks & plots
│   ├── evaluate.py                 # Evaluation (metrics, confusion matrix, sample predictions)
│   └── predict.py                  # Single image prediction
│
├── Frontend/                       # Streamlit Web Application
│   ├── app.py                      # Streamlit UI for brain tumor detection
│   └── style.css                   # Custom CSS styling
│
├── dataset/                        # MRI image dataset
│   ├── Training/                   # Training images (4 classes)
│   └── Testing/                    # Testing images (4 classes)
│
├── outputs/                        # Model & results (generated after training)
│   ├── brain_tumor_model.keras     # Trained CNN model (trained on Colab)
│   ├── accuracy_plot.png           # Training/Validation accuracy plot
│   ├── loss_plot.png               # Training/Validation loss plot
│   ├── confusion_matrix.png        # Confusion matrix heatmap
│   ├── classification_report.txt   # Precision, recall, F1-score report
│   └── sample_predictions.png      # Sample test predictions with labels
│
|
├── Approach and Outputs.pdf        # Project approach documentation
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/shubham8patil/DL_ETE_Project.git
   cd DL_ETE_Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### Run the Streamlit Web App (Frontend)

```bash
streamlit run Frontend/app.py
```

This launches a web interface where you can:
- Upload a brain MRI image (JPG/PNG)
- Click **"Predict Tumor"** to classify it
- View the predicted class and confidence scores as a bar chart

### Run Evaluation (Backend)

```bash
python -m Backend.evaluate
```

### Run Prediction on a Single Image (Backend)

```bash
python -m Backend.predict
```

---

## ☁️ Model Training on Colab

The model was **trained separately on Google Colab** to leverage free GPU acceleration. The trained model was then downloaded as `brain_tumor_model.keras` and placed in the `outputs/` folder.

### To retrain the model on Colab:

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the `Brain_Tumor_Colab.py` file from this repository
3. In a notebook cell, run:
   ```python
   !python Brain_Tumor_Colab.py
   ```
4. The script will automatically:
   - Clone this repository to get the dataset
   - Preprocess the data with augmentation
   - Build and train the CNN model
   - Evaluate on the test set and generate plots
   - Save the trained model as `brain_tumor_model.keras`
5. Download the trained model from `outputs/brain_tumor_model.keras`

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Image Size | 224 × 224 |
| Validation Split | 20% |

---

## 📈 Results

After training, the following outputs are generated in the `outputs/` folder:

- **`accuracy_plot.png`** — Training vs Validation Accuracy over epochs
- **`loss_plot.png`** — Training vs Validation Loss over epochs
- **`confusion_matrix.png`** — Confusion matrix heatmap on test data
- **`classification_report.txt`** — Detailed precision, recall, and F1-score
- **`sample_predictions.png`** — Grid of sample test images with true vs predicted labels

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **TensorFlow / Keras** | Deep Learning framework for CNN |
| **NumPy** | Numerical computations |
| **Matplotlib & Seaborn** | Data visualization and plots |
| **Scikit-learn** | Classification metrics |
| **Streamlit** | Web application frontend |
| **Pillow (PIL)** | Image processing |
| **OpenCV** | Computer vision utilities |
| **Google Colab** | Cloud-based model training with GPU |


**Shubham Patil**
- GitHub: [@shubham8patil](https://github.com/shubham8patil)

