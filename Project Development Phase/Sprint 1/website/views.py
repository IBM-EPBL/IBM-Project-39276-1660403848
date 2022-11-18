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

        choice = request.form.get('choice')

        if(choice=='Spiral'):
            return redirect(url_for('views.spiral'))

        elif(choice=='Wave'):
            return redirect(url_for('views.wave'))
            
        elif(choice=='Voice'):
            return redirect(url_for('views.voice'))
        
        else:
            return redirect(url_for('views.home'))

    return render_template("home.html", user=current_user, form=form)


@views.route('/spiral', methods=['GET','POST'])
@login_required
def spiral():
    form = UploadFileForm()

    if request.method == 'POST':

        id = request.form.get('upload_image')
        print(id)
        upload_image = request.files['upload_image']   
        # print(upload_image.filename,"\n\n")
        if(upload_image!=''):
            filename = secure_filename(upload_image.filename)
            filename_split = upload_image.filename.split('.')
            img_filename = str(current_user.id) + "." + filename_split[-1]
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER_SPIRAL'],img_filename)
            upload_image.save(filepath)
        return redirect(url_for('views.predict_spiral'))

    return render_template("spiral.html", user=current_user, form=form)


@views.route('/predict_spiral', methods=['GET','POST'])
@login_required
def predict_spiral():
    form = UploadFileForm()
    return render_template("predict_spiral.html", user=current_user, form=form)



@views.route('/wave', methods=['GET','POST'])
@login_required
def wave():
    form = UploadFileForm()

    if request.method == 'POST':

        id = request.form.get('upload_image')
        print(id)
        upload_image = request.files['upload_image']   
        if(upload_image!=''):
            filename = secure_filename(upload_image.filename)
            filename_split = upload_image.filename.split('.')
            img_filename = str(current_user.id) + "." + filename_split[-1]
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER_WAVE'],img_filename)
            upload_image.save(filepath)
        return redirect(url_for('views.predict_wave'))

    return render_template("wave.html", user=current_user, form=form)


@views.route('/predict_wave', methods=['GET','POST'])
@login_required
def predict_wave():
    form = UploadFileForm()
    return render_template("predict_wave.html", user=current_user, form=form)


@views.route('/voice', methods=['GET','POST'])
@login_required
def voice():
    form = UploadFileForm()
    if request.method == 'POST':
        inputs = [x for x in request.form.values()]   #Array to access inputs (1-22 is Input Values, 23 is Button)
        print(inputs)
        return redirect(url_for('views.predict_voice'))
    return render_template("voice.html", user=current_user, form=form)


@views.route('/predict_voice', methods=['GET','POST'])
@login_required
def predict_voice():
    form = UploadFileForm()
    return render_template("predict_voice.html", user=current_user, form=form)



@views.route('/information', methods=['GET','POST'])
@login_required
def info():
    form = UploadFileForm()
    return render_template("information.html", user=current_user, form=form)


