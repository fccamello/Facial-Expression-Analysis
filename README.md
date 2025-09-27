# 🎭 **Facial Expression Recognition**  

This project explores **facial expression recognition** using deep learning and facial landmarks. The goal is to classify human emotions, **happiness, sadness, and surprise** from facial images.  

📂 **Dataset Source:** [Kaggle – Facial Expressions (Happiness, Sadness, Surprise)](https://www.kaggle.com/datasets/denisleu/facial-expressions-happiness-sadness-surprise)  

---

## 📊 **Dataset Overview**  
- **Number of classes:** 3  
- **Class distribution:**  
  - 😀 **Happy** → 1,435 images  
  - 😢 **Sad** → 1,165 images  
  - 😲 **Surprise** → 1,089 images  

The model was trained on **facial landmarks** extracted using **MediaPipe Face Mesh**. Each face is represented as **468 (x, y, z) coordinates**, flattened into a feature vector for classification.  

---

## ⚙️ **Methodology**  

**1. Preprocessing**  
- Images loaded with **OpenCV**  
- Facial landmarks extracted via **MediaPipe Face Mesh**  
- Flattened into numerical vectors  

**2. Model Architecture**  
- Input Layer → **256 neurons**, ReLU  
- Hidden Layer → **128 neurons**, ReLU  
- Output Layer → **3 neurons**, Softmax  

**3. Training Setup**  
- Optimizer → **Adam**  
- Loss → **Categorical Crossentropy**  
- Epochs → **30**  
- Batch Size → **32**  
- Validation Split → **20%**

---

## 📈 **Results**  

✅ **Test Accuracy:** **80.96%**  

**Classification Report:**  

| Class       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| Happy    | **0.79**  | **0.93** | **0.86** |
| Sad      | 0.74      | 0.71   | 0.73     |
| Surprise | **0.93** | 0.75   | 0.83     |

- **Weighted Avg Accuracy:** **0.81**  
- **Observation:** Strongest recognition for *happy* and *surprise*, improvement needed for *sad*.  

---

## 🚀 **How to Run**  

```bash
# 1. Clone the repository
git clone https://github.com/fccamello/Facial-Expression-Recognition.git
cd Facial-Expression-Analysis

# 2. Run the app (loads pre-trained model and makes predictions)
python app.py
