import numpy as np
import requests
import cv2
from skimage import feature
from flask import Blueprint, render_template, request, url_for, redirect, session, flash
from flask import current_app
import os
from functools import wraps
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
# from auth import is_logged_in, current_user
import config

def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('auth.login'))
    return wrap

def current_user():
    return session['name'] if 'name' in session else None

views = Blueprint('views', __name__)

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")

@views.route('/', methods=['GET', 'POST'])
@is_logged_in
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

    return render_template("home.html", user=current_user(), form=form)


token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={
                                    "apikey": config.API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

def get_features(img):
    features = feature.hog(img, orientations=9, pixels_per_cell=(
        10, 10), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    return features

def get_image(filepath):
    image = []
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (200, 200))
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = get_features(img)
    image.append(features)
    image = np.array(image, dtype=float)
    return image

@views.route('/spiral', methods=['GET','POST'])
@is_logged_in
def spiral():
    form = UploadFileForm()

    if request.method == 'POST':

        id = request.form.get('upload_image')
        print("id:" ,id)
        upload_image = request.files['upload_image']   
        print(upload_image.filename,"\n\n")

        if(upload_image!=''):
            filename = secure_filename(upload_image.filename)
            filename_split = upload_image.filename.split('.')
            img_filename = str(current_user()) + "." + filename_split[-1]
            filepath = os.path.join(
                current_app.config['UPLOAD_FOLDER_SPIRAL'], img_filename)
            upload_image.save(filepath)

        image = get_image(filepath)
        payload_scoring = {"input_data": [{"values": (image.tolist())}]}
        response_scoring = requests.post('https://jp-tok.ml.cloud.ibm.com/ml/v4/deployments/cd63b4af-d1ca-428f-b0de-c1d878df86cd/predictions?version=2022-11-17',
                                        json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})

        print("Scoring response")
        print(response_scoring.json()) # Can get probability if needed
        if (response_scoring.json()['predictions'][0]['values'][0][0]):
            print('Parkinson')
        else:
            print('Healthy')
        return redirect(url_for('views.predict_spiral'))

    return render_template("spiral.html", user=current_user(), form=form)


@views.route('/predict_spiral', methods=['GET','POST'])
@is_logged_in
def predict_spiral():
    form = UploadFileForm()
    return render_template("predict_spiral.html", user=current_user(), form=form)



@views.route('/wave', methods=['GET','POST'])
@is_logged_in
def wave():
    form = UploadFileForm()

    if request.method == 'POST':

        id = request.form.get('upload_image')
        print(id)
        upload_image = request.files['upload_image']   
        if(upload_image!=''):
            filename = secure_filename(upload_image.filename)
            filename_split = upload_image.filename.split('.')
            img_filename = str(current_user()) + "." + filename_split[-1]
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER_WAVE'],img_filename)
            upload_image.save(filepath)

            image = get_image(filepath)

            payload_scoring = {"input_data": [{"values": (image.tolist())}]}
            response_scoring = requests.post('https://jp-tok.ml.cloud.ibm.com/ml/v4/deployments/8d7f5128-75d6-4b69-a2b8-7e3d5df0e68c/predictions?version=2022-11-17', json=payload_scoring,
                                            headers={'Authorization': 'Bearer ' + mltoken})
            print("Scoring response")
            print(response_scoring.json()) # Can get probability if needed
            if (response_scoring.json()['predictions'][0]['values'][0][0]):
                print('Parkinson')
            else:
                print('Healthy')
        return redirect(url_for('views.predict_wave'))

    return render_template("wave.html", user=current_user(), form=form)


@views.route('/predict_wave', methods=['GET','POST'])
@is_logged_in
def predict_wave():
    form = UploadFileForm()
    return render_template("predict_wave.html", user=current_user(), form=form)


@views.route('/voice', methods=['GET','POST'])
@is_logged_in
def voice():
    form = UploadFileForm()
    if request.method == 'POST':
        inputs = [x for x in request.form.values()]   #Array to access inputs (1-22 is Input Values, 23 is Button)
        print(inputs)

        payload_scoring = {"input_data": [{"fields": [['name', 
        'MDVP:Fo(Hz)', 
        'MDVP:Fhi(Hz)',
        'MDVP:Flo(Hz)',
        'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)',
        'MDVP:RAP',
        'MDVP:PPQ',
        'Jitter:DDP',
        'MDVP:Shimmer',
        'MDVP:Shimmer(dB)',
        'Shimmer:APQ3',
        'Shimmer:APQ5',
        'MDVP:APQ',
        'Shimmer:DDA',
        'NHR',
        'HNR',
        'RPDE',
        'DFA',
        'spread1',
        'spread2',
        'D2',
        'PPE']], "values": [['phon_R01_S01_1', 119.99200, 157.30200, 74.99700, 0.00784, 0.00007, 0.00370, 0.00554, 0.01109, 0.04374,
        0.42600, 0.02182, 0.03130, 0.02971, 0.06545, 0.02211, 21.03300, 0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654]]}]}

        response_scoring = requests.post('https://jp-tok.ml.cloud.ibm.com/ml/v4/deployments/parkinson_voice/predictions?version=2022-11-17', json=payload_scoring,
        headers={'Authorization': 'Bearer ' + mltoken})
        print("Scoring response")
        print(response_scoring.json()) # Can get probability if needed
        
        if(response_scoring.json()['predictions'][0]['values'][0][0]):
            print('Parkinson')
        else:
            print('Healthy')

        return redirect(url_for('views.predict_voice'))
    return render_template("voice.html", user=current_user(), form=form)


@views.route('/predict_voice', methods=['GET','POST'])
@is_logged_in
def predict_voice():
    form = UploadFileForm()
    return render_template("predict_voice.html", user=current_user(), form=form)



@views.route('/information', methods=['GET','POST'])
@is_logged_in
def info():
    form = UploadFileForm()
    return render_template("information.html", user=current_user(), form=form)


