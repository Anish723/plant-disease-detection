from flask import Blueprint, render_template, request, jsonify, redirect, flash, session, url_for
from flask_login import login_required, current_user
from models import db, Prediction, User

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import os
import json
import uuid
from sqlalchemy import func

# from utils.pdf_report import generate_prediction_pdf
# from flask import send_file

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import time

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import datetime

from extensions import socketio
from flask_socketio import emit, join_room

from flask_login import login_user, logout_user

main = Blueprint("main", __name__)

UPLOAD_FOLDER = "static/uploads"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================

# LOAD DISEASE MODEL

# ===============================

with open("model/classes.json", "r") as f:
    disease_classes = json.load(f)

disease_model = models.vgg16(weights=None)

disease_model.classifier[6] = nn.Linear(4096, len(disease_classes))

disease_model.load_state_dict(
    torch.load("model/best_model.pth", map_location=device)
)

disease_model = disease_model.to(device)
disease_model.eval()

# ===============================
# LOAD BINARY MODEL
# ===============================

with open("model/binary_classes.json", "r") as f:
    binary_classes = json.load(f)

binary_model = models.mobilenet_v2(weights=None)

binary_model.classifier[1] = nn.Linear(binary_model.last_channel, 2)

binary_model.load_state_dict(
    torch.load("model/binary_model.pth", map_location=device)
)

binary_model = binary_model.to(device)
binary_model.eval()

# ===============================
# IMAGE TRANSFORM
# ===============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# LOAD RAG VECTOR DATABASE
# ==============================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.load_local(
    "rag/vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

@main.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if not user:
            flash("User not found.", "danger") 
            return render_template("login.html")
        
        if user.is_locked:
            return "Account is locked. Contact admin."
        
        if user.password == password:
            user.failed_attempts = 0
            db.session.commit()

            login_user(user)
            return redirect(url_for("main.dashboard"))
        
        else:
            user.failed_attempts += 1
            if user.failed_attempts >= 5:
                user.is_locked = True
                db.session.commit()
                return "Account locked due to multiple failed attempts. Contact admin."
            db.session.commit()
            flash(f"Invalid password. Attempt left: {5 - user.failed_attempts}", "danger")
            return render_template("login.html")

    return render_template("login.html")

@main.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            flash("Email already registered.", "danger")
            return render_template("register.html")
        
        new_user = User(
            username=username,
            email=email,
            password=password
        )
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("main.login"))
    
    return render_template("register.html")

@main.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("main.login"))

@main.route("/")
def home():
    return redirect(url_for("main.dashboard"))

# ===============================
# DASHBOARD
# ===============================

@main.route("/dashboard")
@login_required
def dashboard():

    predictions = Prediction.query.filter_by(
        user_id=current_user.id
    ).all()

    return render_template(
        "dashboard.html",
        predictions=predictions
    )

# ===============================
# IMAGE PREDICTION
# ===============================

@main.route("/predict", methods=["POST"])
@login_required
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"})

    import uuid

    # filename = str(uuid.uuid4()) + ".jpg"
    ext = file.filename.rsplit(".", 1)[-1].lower()
    filename = str(uuid.uuid4()) + "." + ext

    os.makedirs("static/uploads", exist_ok=True)

    full_path = os.path.join("static/uploads", filename)

    file.save(full_path)

    try:
        image = Image.open(full_path).convert("RGB")
    except:
        return jsonify({"error": "Invalid image file"})

    db_path = f"uploads/{filename}"

    # ===============================
    # IMAGE PREPROCESSING
    # ===============================


    image_tensor = transform(image).unsqueeze(0).to(device)

    start_time = time.time()

    # ===============================
    # STAGE 1 — PLANT VS NON-PLANT
    # ===============================

    with torch.no_grad():

        binary_output = binary_model(image_tensor)

        binary_probs = torch.softmax(binary_output, dim=1)

        binary_conf, binary_pred = torch.max(binary_probs, 1)

    binary_label = binary_classes[binary_pred.item()]

    binary_conf = binary_conf.item() * 100

    if binary_label.lower() != "plant" or binary_conf < 70:

        return jsonify({
            "error": "Uploaded image is not a plant leaf."
        })

    # ===============================
    # STAGE 2 — DISEASE DETECTION
    # ===============================

    with torch.no_grad():

        outputs = disease_model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        top2_probs, top2_idx = torch.topk(probabilities, 2)

        confidence = top2_probs[0][0].item() * 100
        second_conf = top2_probs[0][1].item() * 100

        predicted = top2_idx[0][0]

        predicted_class = disease_classes[predicted.item()]
    gap = confidence - second_conf


    predicted_class = predicted_class.replace("___", " ")

    prediction_time = round(time.time() - start_time, 2)

    if "healthy" in predicted_class.lower():
        predicted_class = "No disease found. The plant appears healthy."

    elif gap < 10:
        predicted_class = "The system is unsure. This may be a new or uncommon disease."
    
    elif confidence >= 70:
        predicted_class = f"Predicted Disease: {predicted_class}"

    elif 40 <= confidence < 70:
        predicted_class = "The plant seems to have a disease, but the system doesn't have the knowledge about this disease."
    
    else:
        predicted_class = "The plant appears healthy or the disease can not be identified with the given image. Please upload a clearer image."
    

    # ===============================
    # SAVE PREDICTION
    # ===============================

    new_prediction = Prediction(
        image_path=db_path,
        result=predicted_class,
        confidence=round(confidence, 2),
        user_id=current_user.id
    )

    db.session.add(new_prediction)
    db.session.commit()

    # ===============================
    # RETURN RESULT
    # ===============================

    return jsonify({
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "image_path": db_path,
        "time": prediction_time,
        "prediction_id": new_prediction.id
    })

# ===============================
# ADMIN DASHBOARD
# ===============================

@main.route("/admin")
@login_required
def admin_dashboard():

    if not current_user.is_admin:
        return "Access Denied", 403

    total_users = User.query.count()

    total_predictions = Prediction.query.count()

    disease_counts = db.session.query(
        Prediction.result,
        func.count(Prediction.result)
    ).group_by(Prediction.result).all()

    labels = [d[0] for d in disease_counts]

    values = [d[1] for d in disease_counts]

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_predictions=total_predictions,
        labels=labels,
        values=values
    )
# ===============================
# DELETE USER
# ===============================

@main.route("/admin/delete-user/<int:user_id>")
@login_required
def delete_user(user_id):

    if not current_user.is_admin:
        return "Access Denied", 403

    user = User.query.get_or_404(user_id)

    if user.is_admin:
        flash("Cannot delete another admin!", "danger")
        return redirect("/admin/users")

    db.session.delete(user)
    db.session.commit()

    flash("User deleted successfully!", "success")

    return redirect("/admin/users")

# ===============================
# DELETE PREDICTION
# ===============================

@main.route("/admin/delete-prediction/<int:pred_id>")
@login_required
def delete_prediction(pred_id):

    if not current_user.is_admin:
        return "Access Denied", 403

    prediction = Prediction.query.get_or_404(pred_id)

    db.session.delete(prediction)
    db.session.commit()

    flash("Prediction deleted successfully!", "success")

    return redirect("/admin/users")

# ===============================
# UNLOCK USER
# ===============================

@main.route("/admin/unlock-user/<int:user_id>")
@login_required
def unlock_user(user_id):

    if not current_user.is_admin:
        return "Access Denied", 403

    user = User.query.get_or_404(user_id)

    user.is_locked = False
    user.failed_attempts = 0

    db.session.commit()

    flash("User account unlocked.", "success")

    return redirect("/admin/users")

# ===============================
# PROFILE PAGE
# ===============================

@main.route("/profile", methods=["GET","POST"])
@login_required
def profile():

    if request.method == "POST":

        current_user.username = request.form.get("username")
        current_user.phone = request.form.get("phone")
        current_user.gender = request.form.get("gender")

        file = request.files.get("profile_pic")

        if file and file.filename != "":

            os.makedirs("static/profiles", exist_ok=True)

            ext = file.filename.split(".")[-1]
            filename = str(uuid.uuid4()) + "." + ext

            filepath = os.path.join("static/profiles", filename)
            file.save(filepath)

            current_user.profile_image = f"profiles/{filename}"

        db.session.commit()

        flash("Profile updated successfully!", "success")

        return redirect("/profile")

    return render_template("profile.html")

# ===============================
# ADMIN USERS PAGE
# ===============================

@main.route("/admin/users")
@login_required
def admin_users():

    if not current_user.is_admin:
        return "Access Denied", 403

    search_query = request.args.get("search")

    if search_query:
        users = User.query.filter(
            (User.email.contains(search_query)) |
            (User.username.contains(search_query))
        ).all()
    else:
        users = User.query.all()

    return render_template("admin_users.html", users=users)

# ===============================
# CHATBOT
# ===============================

@main.route("/chatbot", methods=["POST"])
@login_required
def chatbot():

    try:

        data = request.get_json()

        message = data.get("message")

        disease = data.get("disease")

        reply = generate_bot_reply(message, disease)

        return jsonify({
            "reply": reply
        })

    except Exception as e:

        print("CHATBOT ERROR:", e)

        return jsonify({
            "reply": "Sorry, I encountered an error while answering your question."
        })

# ===============================
# CHATBOT LOGIC
# ===============================


import random

def generate_bot_reply(message, disease=None):

    message = message.lower()

    greetings = [
        "Hello! 🌱 I'm your Plant Doctor assistant.",
        "Hi there! I'm here to help you understand plant diseases.",
        "Hello! Ask me anything about plant health."
    ]

    treatment_tips = [
        "You should remove infected leaves and apply a recommended fungicide.",
        "Using copper-based fungicides can help control the disease.",
        "Try removing infected plant parts and improve air circulation."
    ]

    prevention_tips = [
        "Maintain good airflow between plants.",
        "Avoid watering leaves directly.",
        "Crop rotation can reduce disease spread."
    ]

    causes = {
        "late blight": "Late blight is caused by the pathogen Phytophthora infestans and spreads quickly in humid conditions.",
        "early blight": "Early blight is caused by the fungus Alternaria solani.",
        "leaf mold": "Leaf mold is caused by the fungus Passalora fulva."
    }

    symptoms = {
        "late blight": "Dark lesions appear on leaves and stems and spread rapidly.",
        "early blight": "Brown spots with concentric rings appear on leaves.",
        "leaf mold": "Yellow spots appear on leaf surfaces while mold develops underneath."
    }

    # greeting
    if "hello" in message or "hi" in message:
        return random.choice(greetings)

    # if user asks cause
    if "cause" in message:
        if disease:
            d = disease.replace("_"," ").lower()
            return causes.get(d, "Plant diseases are usually caused by fungi, bacteria, or viruses.")
        return "Plant diseases are usually caused by fungi, bacteria, or viruses."

    # symptoms
    if "symptom" in message:
        if disease:
            d = disease.replace("_"," ").lower()
            return symptoms.get(d, "Symptoms usually include leaf spots, discoloration, or mold growth.")
        return "Symptoms often include leaf spots or discoloration."

    # treatment
    if "treat" in message or "medicine" in message:
        return random.choice(treatment_tips)

    # prevention
    if "prevent" in message:
        return random.choice(prevention_tips)

    # fertilizer question
    if "fertilizer" in message:
        return "Balanced fertilizers with nitrogen, phosphorus, and potassium help plants stay healthy."

    # if prediction exists
    if disease:
        disease_name = disease.replace("_"," ")
        return f"This plant appears to have {disease_name}. You can ask me about its cause, symptoms, treatment, or prevention."

    # default
    return "I can help explain plant diseases, treatments, and prevention tips. Ask me anything!"

# ===============================
# Report Generation
# ===============================

@main.route("/download-full-report")
@login_required
def download_full_report():

    predictions = Prediction.query.filter_by(user_id=current_user.id).all()

    os.makedirs("static/reports", exist_ok=True)

    pdf_path = f"static/reports/full_report_{current_user.id}.pdf"

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("Full Prediction Report", styles['Title']))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"User: {current_user.username}", styles['Normal']))
    elements.append(Paragraph(f"Email: {current_user.email}", styles['Normal']))
    elements.append(Spacer(1,20))

    for pred in predictions:

        elements.append(Paragraph(f"Disease: {pred.result}", styles['Normal']))
        elements.append(Paragraph(f"Confidence: {pred.confidence}%", styles['Normal']))
        elements.append(Spacer(1,10))

        img_path = os.path.join("static", pred.image_path)

        if os.path.exists(img_path):
            elements.append(PDFImage(img_path, 2*inch, 2*inch))

        elements.append(Spacer(1,20))

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)

    doc.build(elements)

    return redirect("/" + pdf_path)


@main.route("/download-report/<int:pred_id>")
@login_required
def download_report(pred_id):

    prediction = Prediction.query.get_or_404(pred_id)

    if prediction.user_id != current_user.id and not current_user.is_admin:
        return "Access Denied", 403

    pdf_path = f"static/reports/prediction_{pred_id}.pdf"

    os.makedirs("static/reports", exist_ok=True)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("Plant Disease Detection Report", styles['Title']))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"User Name: {current_user.username}", styles['Normal']))
    elements.append(Paragraph(f"Email: {current_user.email}", styles['Normal']))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Disease: {prediction.result}", styles['Normal']))
    elements.append(Paragraph(f"Confidence: {prediction.confidence}%", styles['Normal']))
    elements.append(Paragraph(f"Date: {datetime.datetime.now()}", styles['Normal']))
    elements.append(Spacer(1,20))

    img_path = os.path.join("static", prediction.image_path)

    if os.path.exists(img_path):
        elements.append(PDFImage(img_path, 3*inch, 3*inch))

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)

    doc.build(elements)

    return redirect("/" + pdf_path)

# ===============================
# SUPPORT SYSTEM ROUTES
# ===============================

from models import SupportTicket, SupportMessage


# 1️⃣ User - View Tickets
@main.route("/support")
@login_required
def support():

    tickets = SupportTicket.query.filter_by(user_id=current_user.id).all()

    return render_template("support.html", tickets=tickets)


# 2️⃣ Create Ticket
@main.route("/support/create", methods=["POST"])
@login_required
def create_ticket():

    subject = request.form.get("subject")
    message = request.form.get("message")

    ticket = SupportTicket(
        user_id=current_user.id,
        subject=subject
    )

    db.session.add(ticket)
    db.session.commit()

    msg = SupportMessage(
        ticket_id=ticket.id,
        sender="user",
        message=message
    )

    db.session.add(msg)
    db.session.commit()

    flash("Ticket created!", "success")

    return redirect("/support")


# 3️⃣ Open Chat
from flask import abort

@main.route("/support/<int:ticket_id>")
@login_required
def support_chat(ticket_id):

    ticket = SupportTicket.query.get(ticket_id)
    if not ticket:
        abort(404)

    if ticket.user_id != current_user.id and not current_user.is_admin:
        return "Access Denied", 403   
    

    return render_template("support_chat.html", ticket=ticket)


# 4️⃣ Send Message
@main.route("/support/send/<int:ticket_id>", methods=["POST"])
@login_required
def send_message(ticket_id):
    ticket = SupportTicket.query.get_or_404(ticket_id)

    if ticket.status == "closed":
        return redirect(f"/support/{ticket_id}")
    
    text = request.form.get("message")

    if not text or text.strip() == "":
        return redirect(f"/support/{ticket_id}")

    sender = "admin" if current_user.is_admin else "user"

    msg = SupportMessage(
        ticket_id=ticket_id,
        sender=sender,
        message=text
    )

    db.session.add(msg)
    db.session.commit()

    return redirect(f"/support/{ticket_id}")


# 5️⃣ Admin Panel - View All Tickets
@main.route("/admin/support")
@login_required
def admin_support():

    if not current_user.is_admin:
        return "Access Denied", 403

    tickets = SupportTicket.query.all()

    return render_template("admin_support.html", tickets=tickets)

@main.route("/admin/new-messages")
@login_required
def new_messages():

    if not current_user.is_admin:
        return jsonify({"count": 0})

    count = SupportTicket.query.filter_by(status="open").count()

    return jsonify({"count": count})

@main.route("/support/close/<int:ticket_id>")
@login_required
def close_ticket(ticket_id):

    ticket = SupportTicket.query.get_or_404(ticket_id)

    if ticket.user_id != current_user.id and not current_user.is_admin:
        return "Access Denied", 403

    ticket.status = "closed"
    db.session.commit()

    flash("Ticket closed successfully!", "success")

    return redirect(f"/support/{ticket_id}")


@socketio.on("join")
def on_join(data):
    room = str(data["ticket_id"])
    join_room(room)


@socketio.on("typing")
def handle_typing(data):

    ticket_id = str(data["ticket_id"])

    name = "Admin" if current_user.is_admin else current_user.username

    emit("show_typing", {
        "name": name,
        "user_id": current_user.id
    }, room=ticket_id)


@socketio.on("stop_typing")
def stop_typing(data):

    ticket_id = str(data["ticket_id"])

    emit("hide_typing", {}, room=ticket_id)

@socketio.on("send_message")
def handle_message(data):
    
    if not data.get("message") or data["message"].strip() == "":
        return

    ticket_id = str(data["ticket_id"])
    
    sender = "admin" if current_user.is_admin else "user"
    name = "Admin" if current_user.is_admin else current_user.username

    emit("receive_message", {
        "message": data["message"],
        "sender": sender,
        "time": data["time"],
        "name": name
    }, room=ticket_id)

    msg = SupportMessage(
        ticket_id=int(ticket_id),
        sender=sender,
        message=data["message"]
    )
    db.session.add(msg)
    db.session.commit()