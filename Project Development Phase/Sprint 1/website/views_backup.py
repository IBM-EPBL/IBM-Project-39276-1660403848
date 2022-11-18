from flask import Blueprint, render_template, request, flash, jsonify, url_for, redirect
from flask_login import login_required, current_user
from flask import current_app
from .models import Note
from . import db
import os
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename

views = Blueprint('views', __name__)

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    form = UploadFileForm()
    if request.method == 'POST':

        id = request.form.get('upload_image')
        print(id)
        upload_image = request.files['upload_image']   
        # print(upload_image.filename,"\n\n")
        if(upload_image!=''):
            filename = secure_filename(upload_image.filename)
            filename_split = upload_image.filename.split('.')
            img_filename = str(current_user.id) + "." + filename_split[1]
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER_SPIRAL'],img_filename)
            # print(filepath,"\n\n")
            upload_image.save(filepath)
            # return redirect(url_for('views.predict'), path=filepath) 
        return redirect(url_for('views.predict_spiral'))

    return render_template("home.html", user=current_user, form=form)


@views.route('/predict_spiral', methods=['GET','POST'])
@login_required
def predict_spiral():
    form = UploadFileForm()

    return render_template("predict_spiral.html", user=current_user, form=form)

@views.route('/predict_wave', methods=['GET','POST'])
@login_required
def predict_wave():
    form = UploadFileForm()
    return render_template("predict_wave.html", user=current_user, form=form)

@views.route('/information', methods=['GET','POST'])
@login_required
def info():
    form = UploadFileForm()
    return render_template("information.html", user=current_user, form=form)


