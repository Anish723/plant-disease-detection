from flask import Flask
from models import db, User
from extensions import socketio  # ✅ IMPORT FROM NEW FILE
from flask_login import LoginManager

login_manager = LoginManager()


def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'your_secret_key'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///plant_disease.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    socketio.init_app(app, cors_allowed_origins="*")  # ✅ INIT HERE
    db.init_app(app)
    login_manager.init_app(app)

    login_manager.login_view = 'main.login'

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    from routes import main
    app.register_blueprint(main)
 
    return app


app = create_app()

if __name__ == "__main__":
    socketio.run(app, debug=True)