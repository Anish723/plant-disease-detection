from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from models import db, User
from flask_bcrypt import Bcrypt
bcrypt = Bcrypt()
from flask_login import login_user, logout_user, login_required
import re

auth = Blueprint("auth", __name__)
bcrypt = Bcrypt()


def is_valid_password(password):
    if len(password) < 8 or len(password) > 15:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True


@auth.route("/")
def home():
    return render_template("login.html")


@auth.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if user:

            if user.is_locked:
                flash("Account is locked. Contact admin.", "danger")
                return render_template("login.html")

            if bcrypt.check_password_hash(user.password, password):

                user.failed_attempts = 0
                db.session.commit()

                login_user(user)
                return redirect(url_for("main.dashboard"))

            else:
                user.failed_attempts += 1

                if user.failed_attempts >= 5:
                    user.is_locked = True
                    flash("Account locked due to multiple failed attempts.", "danger")
                else:
                    flash("Incorrect password.", "danger")

                db.session.commit()

        else:
            flash("Email does not exist.", "danger")

    return render_template("login.html")


@auth.route("/signup")
def signup_page():
    return render_template("signup.html")


@auth.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("username")
    email = request.form.get("email").strip().lower()
    password = request.form.get("password")
    confirm_password = request.form.get("confirm_password")

    if password != confirm_password:
        flash("Passwords do not match", "danger")
        return redirect(url_for("auth.signup_page"))

    if not is_valid_password(password):
        flash("Password does not meet requirements", "danger")
        return redirect(url_for("auth.signup_page"))

    if User.query.filter_by(email=email).first():
        flash("Email already registered", "danger")
        return redirect(url_for("auth.signup_page"))

    hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")

    user = User(username=username, email=email, password=hashed_pw)
    db.session.add(user)
    db.session.commit()

    flash("Account created successfully! Please login.", "success")
    return redirect(url_for("auth.home"))


@auth.route("/logout")
@login_required
def logout():

    session.clear()   # 🔥 deletes chatbot data

    logout_user()

    flash("Logged out successfully.", "info")
    return redirect(url_for("auth.home"))