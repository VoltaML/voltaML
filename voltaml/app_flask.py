import os
#import magic
import urllib.request
from flask import Flask, flash, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename

import sys
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser

import torch
from voltaml.compile import VoltaGPUCompiler
from voltaml.inference import gpu_performance
import torchvision
from voltaml.models.common import DetectMultiBackend
from test_dali import dali_infer
import shutil
import warnings
warnings.filterwarnings('ignore')

UPLOAD_FOLDER = 'test_data/uploads'

app = Flask(__name__)
app.secret_key = "segmind one"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024


ALLOWED_EXTENSIONS_IMAGES = set(['png', 'jpg', 'jpeg'])
ALLOWED_EXTENSIONS_WEIGHTS = set(['pt'])

def allowed_file_images(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMAGES

def allowed_file_weights(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_WEIGHTS

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=False)
os.makedirs(app.config['UPLOAD_FOLDER']+'/images', exist_ok=False)
os.makedirs(app.config['UPLOAD_FOLDER']+'/weights', exist_ok=False)

@app.route('/')
def upload_form():
    # return "<h1 style='color:blue'> Volta ML!</h1>"
    return render_template('upload.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    # print(request.files)

    if request.method == 'POST':
        # check if the post request has the files part
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file_images(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER']+'/images', filename))

        flash('File(s) successfully uploaded')
        # return redirect('/')
        return jsonify({'status':'Files uploaded successfully.'})

@app.route('/upload-weights', methods=['POST','GET'])
def upload_weights():
    if request.method == 'POST':
        # print(request.files)
        # check if the post request has the files part
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        print(files)
        for file in files:
            if file and allowed_file_weights(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER']+'/weights', filename))
        flash('File(s) successfully uploaded')
        # return {'Files uploaded successfully.'}
        return jsonify({'status':'Weights uploaded successfully.'})

@app.route('/build-engine', methods=['POST','GET'])
def build_engine():
    args_json = request.get_json()
    # print(args_json)
    b_sz = args_json['batch_size']
    input_sh = args_json['input_shape']
    input_shape = (b_sz,input_sh[0],input_sh[1],input_sh[2])
    precision = args_json['precision']
    model_name = args_json['filename']
    filename = args_json['filename'].split('.')[0]
    
    model_path = app.config['UPLOAD_FOLDER']+'/weights/'+model_name
    
    compiled_model_dir = app.config['UPLOAD_FOLDER']+'/weights/%s_bs%s_%s.engine'%(filename,b_sz,precision) ## Compiled model directory
    is_yolo = True
    input_name = 'images'
    output_name = 'output'
    simplify = True
    
    model = DetectMultiBackend(model_path)
    
    compiler = VoltaGPUCompiler(
    model=model,
    output_dir=compiled_model_dir,
    input_shape=input_shape,
    precision=precision,
    input_name=input_name,
    output_name=output_name,
    simplify=simplify
    )

    compiled_model = compiler.compile()
    
    return jsonify({'status':'Engine build successfully.'})

@app.route('/infer', methods=['POST','GET'])
def infer():
    args_json = request.get_json()
    # print(args_json)
    b_sz = args_json['batch_size']
    input_sh = args_json['input_shape']
    input_shape = (b_sz,input_sh[0],input_sh[1],input_sh[2])
    precision = args_json['precision']
    model_name = args_json['filename']
    filename = args_json['filename'].split('.')[0]
    use_engine = args_json['use_engine']
    
    if use_engine:
        model_path = app.config['UPLOAD_FOLDER']+'/weights/%s_bs%s_%s.engine'%(filename,b_sz,precision) ## Compiled model directory
    else:
        model_path = app.config['UPLOAD_FOLDER']+'/weights/'+model_name
    
    print(app.config['UPLOAD_FOLDER'])
    t, tt_s, tt_ms, tot_det, tot_images = dali_infer(model_path, b_sz)
    
    out_json = {'Total Images :':tot_images,
               'Total time in sec': tt_s,
               'Total No of Detections':tot_det}
    return jsonify(out_json)
           
@app.route('/reset_dir', methods=['POST','GET'])
def reset_dir():
    args_json = request.get_json()
    reset = args_json['reset']
    
    if reset:
        try:
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
        except:
            pass
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=False)
        os.makedirs(app.config['UPLOAD_FOLDER']+'/images', exist_ok=False)
        os.makedirs(app.config['UPLOAD_FOLDER']+'/weights', exist_ok=False)
        
    return jsonify({'status': 'Files reset successful.'})

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000, debug=False)