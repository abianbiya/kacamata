import os
from flask import Flask, escape, request, render_template, flash, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from classifier import classifier
from mainmodel import DataStore
import numpy as np
import base64
from math import *

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CLASS_FOLDER'] = 'static/img/class/'
app.config['CLAZZ_FOLDER'] = 'static/img/clazz/'
# app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# jenis kelamin 1 = pria, 2 = wanita
# jenis kacamata 1 = baca, 2 = sung

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/')
def base():
    return 'base'

@app.route('/home')
def hello():
    return render_template('home.html')

@app.route('/upload/<jk>/<km>')
def upload(jk, km):
    page = request.path.split('/')
    if page[2] == '1' and page[3] == '1':
        pg = 'bp'
    elif page[2] == '2' and page[3] == '1':
        pg = 'bw'
    elif page[2] == '1' and page[3] == '2':
        pg = 'gp'
    elif page[2] == '2' and page[3] == '2':
        pg = 'gw'

    return render_template('upload.html', jk=jk, km=km, page=pg)

@app.route('/webcam/<jk>/<km>')
def webcam(jk, km):
    page = request.path.split('/')
    if page[2] == '1' and page[3] == '1':
        pg = 'bp'
    elif page[2] == '2' and page[3] == '1':
        pg = 'bw'
    elif page[2] == '1' and page[3] == '2':
        pg = 'gp'
    elif page[2] == '2' and page[3] == '2':
        pg = 'gw'

    return render_template('webcam.html', jk=jk, km=km, page=pg)

@app.route('/post/upload/<jk>/<km>',  methods=['GET', 'POST'])
def post_upload(jk, km):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            # print(request.form['file'])
            # exit(0)
            file = request.form['file']
            starter = file.find(',')
            image_data = file[starter+1:]
            imgdata = base64.b64decode(image_data)
            filename = 'uploads/some_image.jpg'  # I assume you have a way of picking unique filenames
            with open(filename, 'wb') as f:
                f.write(imgdata)
            
            claz = classifier(jk, km)
            hasil, kiri, kanan = claz.predict(filename)
            code=200
            if hasil == 98:
                code=98
                return jsonify(
                    code=code
                )
            elif hasil == 99:
                code = 99
                return jsonify(
                    code=code
                )
                
            kiri = np.array(kiri)[0]
            kanan = np.array(kanan)[0]
            degree = (degrees(atan((int(kanan[0]) - int(kiri[0])) / (int(kanan[1]) - int(kiri[1])))))
            if degree < 0:
                degree = (0 - 90 - degree)
            else:
                degree = 90 - degree

            if km == '1':
                kelas='baca-0'+str(hasil)
                classpath = str(os.path.join(app.config['CLASS_FOLDER'], str(hasil)+'.png'))
            else:
                kelas='sung-0'+str(hasil)
                classpath = str(os.path.join(app.config['CLAZZ_FOLDER'], str(hasil)+'.png'))    

            return jsonify(
                code=code,
                kelas=kelas,
                uploaded=filename,
                classified=classpath,
                kiriX=int(kiri[0]),
                kiriY=int(kiri[1]),
                kananX=int(kanan[0]),
                kananY=int(kanan[1]),
                degree=degree
            )

        if 'file' in request.files:
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                pathh = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(pathh)
                claz = classifier(jk, km)
                hasil, kiri, kanan = claz.predict(pathh)
                code=200
                if hasil == 98:
                    code=98
                    return jsonify(
                        code=code
                    )
                elif hasil == 99:
                    code = 99
                    return jsonify(
                        code=code
                    )
                
                kiri = np.array(kiri)[0]
                kanan = np.array(kanan)[0]
                if int(kanan[1]) - int(kiri[1]) != 0:
                    degree = (degrees(atan((int(kanan[0]) - int(kiri[0])) / (int(kanan[1]) - int(kiri[1])))))
                else:
                    degree = 90

                if degree < 0:
                    degree = (0 - 90 - degree)
                else:
                    degree = 90 - degree

                if km == '1':
                    kelas='baca-0'+str(hasil)
                    classpath = str(os.path.join(app.config['CLASS_FOLDER'], str(hasil)+'.png'))
                else:
                    kelas='sung-0'+str(hasil)
                    classpath = str(os.path.join(app.config['CLAZZ_FOLDER'], str(hasil)+'.png'))    

                return jsonify(
                    code=code,
                    kelas=kelas,
                    uploaded=pathh,
                    classified=classpath,
                    kiriX=int(kiri[0]),
                    kiriY=int(kiri[1]),
                    kananX=int(kanan[0]),
                    kananY=int(kanan[1]),
                    degree=degree
                )
    return request.files

@app.route('/<iss>/<namakelas>')
def like(iss, namakelas):
    data = DataStore()
    # if(km == '1'):
    #     km = 'baca'
    # else:
    #     km = 'sung'

    nama_kelas = namakelas
    # nama_kelas = km+'-0'+kelas
    if iss == 'like':
        data.like(nama_kelas, True)
    else:
        data.like(nama_kelas, False)

    return redirect(request.referrer or url_for('hello'))

@app.route('/data')
def monitoring():
    data = DataStore()
    kacas, matas = data.all_kacamata(True)
    likes = []
    namas = []
    dislikes = []
    for i in kacas:
        likes.append(i['like'])
        namas.append(i['nama'])
        dislikes.append(i['dislike'])
        
    likez = []
    namaz = []
    dislikez = []
    for i in matas:
        likez.append(i['like'])
        namaz.append(i['nama'])
        dislikez.append(i['dislike'])
   
    return render_template('monitoring.html', likes=likes, namas=namas, likez=likez, namaz=namaz, dislikes=dislikes, dislikez=dislikez)

@app.route('/reset')
def enol():
        
    # ekse = None
    # kelas = None
    # kacamata = None
    # db = None

    # def __init__(self):
    #     self.db = TinyDB('db/database.json')
    #     self.ekse = self.db.table('eksekusi')
    #     self.kelas = self.db.table('kelas')
    #     self.kacamata = Query()
    # self.initialize()

    ds = DataStore()
    ds.initialize()
    
    return redirect("/data", code=302)
    # return (monitoring)

