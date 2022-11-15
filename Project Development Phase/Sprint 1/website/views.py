from flask import Blueprint, render_template, request, flash, jsonify, url_for, redirect
from flask_login import login_required, current_user
from flask import current_app
from .models import Note
from . import db
import os
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField

views = Blueprint('views', __name__)

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    form = UploadFileForm()
    if request.method == 'POST':
        upload_image = request.files['upload_image']   
        print(upload_image.filename)
        if(upload_image!=''):
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'],upload_image.filename)
            # upload_image.save(filepath)
            # return redirect(url_for('views.predict'), path=filepath) 
        return redirect(url_for('views.predict'))

    return render_template("home.html", user=current_user, form=form)



@views.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    form = UploadFileForm()
    return render_template("predict.html", user=current_user, form=form)


@views.route('/information', methods=['GET','POST'])
@login_required
def info():
    form = UploadFileForm()
    return render_template("information.html", user=current_user, form=form)


