from flask import Flask, render_template, flash, redirect, url_for, session, request, logging, Blueprint
from wtforms import Form, StringField, TextAreaField, PasswordField, validators, SelectField, IntegerField
import ibm_db
from functools import wraps

auth = Blueprint('auth', __name__)
#IBM DB2 Connection
try:
    conn = ibm_db.connect("DATABASE=bludb;HOSTNAME=2d46b6b4-cbf6-40eb-bbce-6251e6ba0300.bs2io90l08kqb1od8lcg.databases.appdomain.cloud;PORT=32328;SECURITY=SSL;SSLServerCertificate=DigiCertGlobalRootCA.crt;UID=gzs19461;PWD=Fr7LWoNyM0URum9L", "", "")
except:
    print("Unable to connect: ", ibm_db.conn_error())
#Home Page


@auth.route('/')
def index():
    return render_template('home.html')

#Register Form Class


class RegisterForm(Form):
    name = StringField('Name', [validators.Length(min=1, max=50)])
    username = StringField('Username', [validators.Length(min=1, max=25)])
    email = StringField('Email', [validators.length(min=6, max=50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords do not match')
    ])
    confirm = PasswordField('Confirm Password')
#user register


@auth.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    # print(form.validate())
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        password = str(form.password.data)

        sql = "SELECT * FROM users WHERE email=?"
        prep_stmt = ibm_db.prepare(conn, sql)
        ibm_db.bind_param(prep_stmt, 1, email)
        ibm_db.execute(prep_stmt)
        account = ibm_db.fetch_assoc(prep_stmt)
        print(account)
        if account:
            error = "Account already exists! Log in to continue !"
        else:
            insert_sql = "INSERT INTO users (email,username,password,name) values(?,?,?,?)"
            prep_stmt = ibm_db.prepare(conn, insert_sql)
            ibm_db.bind_param(prep_stmt, 1, email)
            ibm_db.bind_param(prep_stmt, 2, username)
            ibm_db.bind_param(prep_stmt, 3, password)
            ibm_db.bind_param(prep_stmt, 4, name)
            #ibm_db.bind_param(prep_stmt, 4, password)
            ibm_db.execute(prep_stmt)
            flash(" Registration successful. Log in to continue !")

        #when registration is successful redirect to home
        return redirect(url_for('auth.login'))
    flash("Registration failed. Please try again !")
    return render_template('register.html', form=form)

#User login


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        error = None
        account = None
        #Get form fields
        email = request.form['email']
        password = request.form['password']
        # print(email, password)

        sql = "SELECT * FROM users WHERE email=? AND password=?"
        stmt = ibm_db.prepare(conn, sql)
        ibm_db.bind_param(stmt, 1, email)
        ibm_db.bind_param(stmt, 2, password)
        ibm_db.execute(stmt)
        account = ibm_db.fetch_assoc(stmt)
        print(account)
    if account:
        session['logged_in'] = True
        session['username'] = account['USERNAME']
        session['email'] = email
        session['name'] = account['NAME']
        flash("Logged in successfully", "success")
        return redirect(url_for('views.home'))
    else:
        error = "Incorrect username / password"
        return render_template('login.html', error=error)


#Is Logged In
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap

def current_user():
    return session['name'] if 'name' in session else None


@auth.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash("Logged out successfully", "success")
    return redirect(url_for('login'))
