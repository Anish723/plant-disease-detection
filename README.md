# Plant Disease Detection System

## About

This is a web application that detects plant diseases from leaf images using deep learning.
You can upload a leaf image and get the result (healthy or disease).

## Features

* Detect plant or non-plant
* Predict disease using VGG16 model
* Show confidence score
* Chatbot for basic help
* Admin dashboard
* Support chat system
* PDF report download

## Tech Used

* Python (Flask)
* PyTorch
* HTML, CSS, Bootstrap
* SQLite
* Socket.IO

## How to Run

git clone https://github.com/Anish723/plant-disease-detection.git
cd plant-disease-detection

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python app.py

Open: http://127.0.0.1:5000

## Note

Model files (.pth) are not included due to large size.

## Author

Anish Kumar
https://github.com/Anish723
