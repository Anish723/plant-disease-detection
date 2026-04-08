# Plant Disease Detection System

## About This Project

This project is a simple web application that helps to detect plant diseases from leaf images.
You can upload an image of a plant leaf, and the system will tell if the plant is healthy or has a disease.

It also includes a chatbot, admin panel, and support system.

---

## What This Project Can Do

* Check if the uploaded image is a plant or not
* Detect plant diseases using a deep learning model (VGG16)
* Show how confident the prediction is
* Chatbot to answer questions about plant diseases
* Admin panel to see users and predictions
* Support system with live chat
* Download report as PDF
* User profile update

---

## Technologies Used

* Python (Flask)
* PyTorch (for deep learning model)
* HTML, CSS, Bootstrap (for frontend)
* JavaScript
* SQLite (database)
* Socket.IO (for real-time chat)

---

## Project Structure

app.py
routes.py
models.py
extensions.py

model/

* classes.json
* binary_classes.json
* training_notebook.ipynb

templates/
static/
utils/
rag/

---

## How to Run This Project

1. Clone the project
   git clone https://github.com/Anish723/plant-disease-detection.git

2. Go to project folder
   cd plant-disease-detection

3. Create virtual environment
   python -m venv venv

4. Activate virtual environment
   venv\Scripts\activate

5. Install libraries
   pip install -r requirements.txt
   pip install flask-socketio langchain-community faiss-cpu sentence-transformers reportlab

6. Run the project
   python app.py

7. Open in browser
   http://127.0.0.1:5000

---

## Important Note

Model files (.pth) are not uploaded because they are too large.
You can train your own model using the notebook file.

---

## Future Improvements

* Show disease area on image (heatmap)
* Make mobile app
* Deploy online
* Improve chatbot

---

## Author

Anish Kumar
GitHub: https://github.com/Anish723

---

## Purpose

This project is made for learning and academic use.
