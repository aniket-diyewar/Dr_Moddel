# 🧠 Diabetic Retinopathy Detection (Multi-Phase Deep Learning Model)

## 📌 Overview
This project focuses on automated detection of **Diabetic Retinopathy (DR)** using deep learning on retinal fundus images.

The model is developed in multiple phases, starting from a baseline and progressively improving using preprocessing, augmentation, and optimization techniques.



## 🎯 Objective
- Detect and classify DR severity
- Improve early-stage detection (Mild & Moderate)
- Build a robust and practical AI-based healthcare solution




## 🔄 Project Phases

### Phase 1: Baseline Model
- Dataset: APTOS
- Basic preprocessing
- Initial model training

### Phase 2: Preprocessing Improvement
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Enhanced image quality
- Better feature extraction

### Phase 3: Model Optimization
- Data augmentation (flip, rotation, brightness)
- Improved generalization
- Reduced overfitting

### Phase 4: Multi-Class Classification
- 5-class DR classification:
  - 0 → No DR  
  - 1 → Mild  
  - 2 → Moderate  
  - 3 → Severe  
  - 4 → Proliferative  



## 🧠 Model Details
- Architecture: EfficientNet-B3  
- Transfer Learning used  
- Fine-tuning applied  
- Loss Function: CrossEntropy  


## 📊 Results
- Good performance on No DR and Severe classes  
- Improvement after preprocessing and augmentation  
- Mild & Moderate classification remains challenging  


## 🔍 Explainability
- Grad-CAM used for visualization  
- Highlights important regions in retinal images  
- Helps in model interpretability  


## 📦 Model Weights
Due to GitHub size limitations, model weights are hosted externally.

Download Model: https://drive.google.com/drive/folders/1QT455LkxMqBvZx3ZQ2eLAmD4UlHJa89R?usp=sharing

After downloading, place the .pth file in the project directory.



## ⚙️ Installation

git clone https://github.com/aniket-diyewar/Dr_Moddel.git  
cd Dr_Moddel  
pip install -r requirements.txt  

