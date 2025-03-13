# AI-Assisted Skin Disease Diagnosis

## 📌 Project Overview
This project is an AI-powered **Skin Disease Diagnosis System** that detects various skin conditions using deep learning. It consists of:
- **Kaggle-trained CNN model** (TensorFlow/Keras)
- **Flask Backend API** (Handles image processing & prediction)
- **React Frontend** (User-friendly UI for image upload & results)

---

## 🚀 Features
- **Deep Learning-based Skin Disease Detection** 🧠
- **Real-time Image Processing & Prediction** 📷
- **Flask API Integration** 🔗
- **User-friendly React UI** 🎨
- **Risk Level & Recommendation System** ⚠️

---

## 🔥 Model Training (Kaggle)
### 1️⃣ Dataset Preparation
- Used **Skin Disease Dataset** containing labeled images.
- Applied **Image Augmentation** to improve generalization.

### 2️⃣ Preprocessing & Handling Imbalance
- Resized images to **150x150**.
- Normalized pixel values (**/255** scaling).
- Used **SMOTE** (Synthetic Minority Oversampling) for class imbalance.

### 3️⃣ Model Architecture
- **Convolutional Neural Network (CNN)** using TensorFlow/Keras.
- Layers: **Conv2D → MaxPooling → Flatten → Dense → Softmax**.
- Optimizer: **Adam**, Loss Function: **Categorical Crossentropy**.
- Callbacks: **EarlyStopping & ModelCheckpoint**.

### 4️⃣ Model Evaluation
- **Confusion Matrix & Classification Report**.
- Achieved **high accuracy** on test data.
- Saved model as `model.h5` for deployment.

---

## 🛠 Backend Setup (Flask API)
### 1️⃣ Install Dependencies
```bash
pip install flask flask-cors tensorflow numpy pillow
```

### 2️⃣ Run Flask Server
```bash
python app.py
```

### 3️⃣ API Endpoint
| Method | Endpoint  | Description |
|--------|----------|-------------|
| POST   | `/predict` | Upload image & get disease prediction |

---

## 🎨 Frontend Setup (React)
### 1️⃣ Install Dependencies
```bash
npm install axios @mui/material @emotion/react @emotion/styled
```

### 2️⃣ Start React App
```bash
npm start
```

---

## 🖼 Usage Guide
1. **Upload an image** of the affected skin area.
2. Click on **"Predict"** to analyze.
3. Get **diagnosis, confidence score, & risk level**.
4. Follow the **recommendation** based on the result.

---

## 🎯 Future Improvements
✅ Deploy on cloud (AWS/GCP) ☁️  
✅ Improve model accuracy with more data 📊  
✅ Add multilingual support 🌍  
✅ Implement secure authentication 🔐  

---

## 🤝 Contributing
Feel free to contribute by raising issues or making pull requests!

---

## 📜 License
This project is open-source under the **MIT License**.

---

## 📧 Contact
📩 Email: aquibkhan8140@gmail.com  
💼 LinkedIn: [aqib]([https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/aqib-kha9/))

