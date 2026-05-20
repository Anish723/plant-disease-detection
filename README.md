# 🌿 Plant Disease Detection System Using VGG16

An AI-powered web application that detects plant diseases from leaf images using Deep Learning and Computer Vision techniques.

---

## 🚀 Features

- Plant leaf disease detection
- Deep learning-based prediction using VGG16
- Flask web application
- User authentication system
- AI-powered disease classification
- Report generation using PDF
- LangChain + FAISS integration
- Image upload support
- Real-time prediction results

---

## 🛠️ Technologies Used

- Python
- Flask
- PyTorch
- VGG16
- OpenCV
- LangChain
- FAISS
- HTML
- CSS
- JavaScript
- SQLAlchemy

---

## 📂 Project Structure

```bash
plant-disease-detection/
│
├── model/
│   ├── binary_classes.json
│   └── classes.json
│
├── rag/
├── static/
├── templates/
├── utils/
│
├── app.py
├── routes.py
├── models.py
├── auth.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation Guide

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Anish723/plant-disease-detection.git
cd plant-disease-detection
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

### 3️⃣ Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Download Model Files

Since `.pth` model files are large, they are not uploaded to GitHub.

Download model files manually and place them inside:

```bash
model/
```

Required files:

- `best_model.pth`
- `binary_model.pth`

---

### Google Drive Links

- Best Model:
https://drive.google.com/file/d/1PCRFDkUVI-dWbuIVcCTkJnBBESQTyGaH/view

- Binary Model:
https://drive.google.com/file/d/1Qag4L-yMKh91Y6SlBtgz6fc-A8jYo9F1/view

---

## ▶️ Run Project

```bash
python app.py
```

---

## 🌐 Open in Browser

```bash
http://127.0.0.1:5000
```

---

## 📸 Output

- Upload plant leaf image
- AI predicts disease
- Displays prediction result instantly

---

## 👨‍💻 Author

Anish Kumar

---

## 📌 Note

This project is developed for educational and research purposes only.
