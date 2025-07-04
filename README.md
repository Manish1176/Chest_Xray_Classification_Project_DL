# Chest X-ray Classification (Normal vs Abnormal)

This project focuses on developing a deep learning pipeline to classify chest X-ray images as **Normal** or **Abnormal** (indicating possible diseases). The model uses **Transfer Learning** with **ResNet50** and includes image preprocessing, augmentation, fine-tuning.

---

## 🔍 Project Overview

* **Type**: Binary Image Classification
* **Classes**:

  * `0`: Normal (No Finding)
  * `1`: Abnormal (Disease Present)
* **Framework**: TensorFlow / Keras
* **Dataset Source**: Kaggle (Chest X-ray Dataset)

---

## 🚀 Pipeline Summary

### 1. **Dataset Extraction**

* Extracted ZIP file containing:

  * Chest X-ray images (`x_ray_images/`)
  * `ground_truth.csv` file with labels (`Finding Labels`)

### 2. **Data Preprocessing**

* Converted all labels into binary classes: `'No Finding' → 0`, others → `1`
* Preprocessed grayscale images:

  * Resized to `224x224`
  * Histogram equalization
  * Added Gaussian noise
  * Applied random rotation

### 3. **Balanced Sampling**

* Extracted equal samples from both classes (e.g., 5000 each) to handle imbalance
* Split into **Train (80%)** and **Test (20%)**

### 4. **Model Architecture**

* Base: `ResNet50` pretrained on ImageNet (top 30 layers unfrozen)
* Custom classification head:

  * Global Average Pooling
  * Dense(256) + Dropout
  * Dense(128) + Dropout
  * Output: Dense(2, softmax)
* Regularization:

  * L2 (`l2=0.01`)
  * Dropout (`0.5`)

### 5. **Training Strategy**

* Optimizer: Adam (`lr=1e-5`)
* Loss: Categorical Crossentropy
* Callbacks:

  * EarlyStopping (`patience=3`)
  * ModelCheckpoint (`best_model.h5`)
* Class Weights: Auto-computed using `sklearn.utils.class_weight`

### 6. **Evaluation**

* Accuracy, Precision, Recall, F1-score using `classification_report`
* Confusion Matrix and metric plots

### 7. **Prediction on New Image**

* Accepts path input to predict class of a new X-ray
* Shows the uploaded image
* Outputs prediction: `Normal` or `Abnormal`

---

## 📂 Project Structure

```
Chest_Xray_Classification_Project/
│
├── data/                         # Extracted images and CSV
├── best_model.h5                # Saved trained model
├── chest_xray_pipeline.ipynb    # Main notebook
├── requirements.txt             # Dependencies
├── README.md                    # Project summary
```

---

## ✅ Key Features

* Transfer Learning with fine-tuning
* Balanced training with class weights
* Fully compatible with Google Colab

---

## 📦 Requirements

* Python 3.8+
* TensorFlow 2.x
* OpenCV, NumPy, Matplotlib, Seaborn
* scikit-learn, pandas

Install with:

```bash
pip install -r requirements.txt
```

---

## 📈 Sample Results

* **Accuracy**: \~61%
* **Balanced Recall**: \~0.47 for Abnormal class
* **Training**: \~10 epochs with early stopping

---

## 💡 Insights and Conclusions

* The model successfully distinguishes between Normal and Abnormal chest X-rays using a ResNet50-based architecture.

* Basic image preprocessing techniques (grayscale equalization, noise injection, and rotation) helped the model learn radiological patterns.

* Balancing the dataset was essential to prevent model bias toward the majority class.

* Fine-tuning the top layers of ResNet50 improved generalization and recall on minority (Abnormal) class.

* With limited data and moderate compute (Google Colab), the model achieved a good balance between performance and explainability.


---

## 👨‍💻 Author

**Manish**

AI & Data Science Engineer

---

## 🔗 License

This project is for educational/research purposes only.

## Note 

To ensure compatibility with Google Colab GPU and avoid memory issues, we limited training to 5,000 images per class (10,000 total). This subset was balanced to ensure equal representation of Normal and Abnormal cases.
