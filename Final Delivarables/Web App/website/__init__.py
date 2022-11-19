from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'Sparsh_Secret_Key'
    app.config['UPLOAD_FOLDER_SPIRAL'] = "website/uploads/spiral"
    app.config['UPLOAD_FOLDER_WAVE'] = "website/uploads/wave"
    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    return app