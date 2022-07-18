import requests
import json
import os
from glob import glob
def upload_images(url, images_path):
    images_url = url + '/upload-image'
    img_list = glob(images_path+'/*.png')
    multiple_files = []
    for file in img_list:
        filename = os.path.basename(file)
        multiple_files.append(('files[]', (filename, open(file,'rb'), 'image/png')))   
    r = requests.post(images_url, files=multiple_files)
    return r.json()
    
def upload_weights(url, file_path):
    weights_url = url + '/upload-weights'
    filename = os.path.basename(file_path)
    files = [('files[]', (filename, open(file_path, 'rb')))]
    r = requests.post(weights_url, files=files)
    return r.json()

def build_engine(url, filename, batch_size, input_shape, precision='fp16'):
    build_url = url + '/build-engine'
    json_dict = json.dumps({'filename':filename,
                'batch_size':batch_size,
                'input_shape':input_shape,
                'precision':precision,
                })
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post(build_url, data=json_dict, headers=headers)
    return r.json()

def batch_infer(url, filename, batch_size, input_shape, precision='fp16', use_engine=True):
    infer_url = url + '/infer'
    json_dict = json.dumps({'filename':filename,
                'batch_size':batch_size,
                'input_shape':input_shape,
                'precision':precision,
                'use_engine':use_engine
                })
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post(infer_url, data=json_dict, headers=headers)
    return r.json()

def reset_dir(url):
    reset_url = url + '/reset_dir'
    json_dict = json.dumps({'reset':True})
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post(reset_url, data=json_dict, headers=headers)
    return r.json()
    