from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

from datetime import datetime
from flask_login import UserMixin
from models import db

class User(db.Model, UserMixin):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    profile_image = db.Column(db.String(200), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    gender = db.Column(db.String(20), nullable=True)

    is_admin = db.Column(db.Boolean, default=False)
    
    failed_attempts = db.Column(db.Integer, default=0)
    is_locked = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    predictions = db.relationship(
        'Prediction',
        backref='user',
        cascade="all, delete",
        lazy=True
    )
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    image_path = db.Column(db.String(300), nullable=False)
    result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id', ondelete="CASCADE"),
        nullable=False
    )

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ===============================
# SUPPORT SYSTEM MODELS
# ===============================

class SupportTicket(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))

    subject = db.Column(db.String(200))
    status = db.Column(db.String(20), default="open")

    created_at = db.Column(db.DateTime, default=db.func.now())

    user = db.relationship("User", backref="tickets")


class SupportMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    ticket_id = db.Column(db.Integer, db.ForeignKey("support_ticket.id"))

    sender = db.Column(db.String(20))  # "user" or "admin"
    message = db.Column(db.Text)

    timestamp = db.Column(db.DateTime, default=db.func.now())

    ticket = db.relationship("SupportTicket", backref="messages")