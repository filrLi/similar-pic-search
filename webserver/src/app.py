import logging
import os
import os.path as path
import shutil

import numpy as np
from numpy import linalg as LA

import tensorflow as tf
from common.config import DATA_PATH, DEFAULT_TABLE
from common.const import UPLOAD_PATH, default_cache_dir, input_shape
from diskcache import Cache
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_restful import reqparse
from indexer.index import (create_index, create_table, delete_table,
                           insert_vectors, milvus_client, search_vectors)
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing import image
from preprocessor.xception import extract_feat
from service.count import do_count
from service.delete import do_delete
from service.search import do_search, query_name_from_ids
from service.theardpool import thread_runner
from service.train import do_train
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from werkzeug.utils import secure_filename

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
global sess
sess = tf.Session(config=config)
set_session(sess)

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['jpg', 'png'])
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
app.config['JSON_SORT_KEYS'] = False
CORS(app)

model = None


def load_model():
    global graph
    graph = tf.get_default_graph()

    global model
    model = Xception(weights='imagenet',
                     input_shape=input_shape,
                     pooling='max',
                     include_top=False)


@app.route('/api/v1/train', methods=['POST'])
def do_train_api():
    args = reqparse.RequestParser(). \
        add_argument('Table', type=str). \
        add_argument('File', type=str). \
        parse_args()
    table_name = args['Table']
    file_path = args['File']
    try:
        thread_runner(1, do_train, table_name, file_path)
        filenames = os.listdir(file_path)
        if not os.path.exists(DATA_PATH):
            os.mkdir(DATA_PATH)
        for filename in filenames:
            shutil.copy(file_path + '/' + filename, DATA_PATH)
        return "Start"
    except Exception as e:
        return "Error with {}".format(e)


@app.route('/api/v1/delete', methods=['POST'])
def do_delete_api():
    args = reqparse.RequestParser(). \
        add_argument('Table', type=str). \
        parse_args()
    table_name = args['Table']
    print("delete table.")
    status = do_delete(table_name)
    try:
        shutil.rmtree(DATA_PATH)
    except:
        print("cannot remove", DATA_PATH)
    return "{}".format(status)


@app.route('/api/v1/count', methods=['POST'])
def do_count_api():
    args = reqparse.RequestParser(). \
        add_argument('Table', type=str). \
        parse_args()
    table_name = args['Table']
    rows = do_count(table_name)
    return "{}".format(rows)


@app.route('/api/v1/process')
def thread_status_api():
    cache = Cache(default_cache_dir)
    return "current: {}, total: {}".format(cache['current'], cache['total'])


@app.route('/data/<image_name>')
def image_path(image_name):
    file_name = DATA_PATH + '/' + image_name
    if path.exists(file_name):
        return send_file(file_name)
    return "file not exist"


@app.route('/api/v1/search', methods=['POST'])
def do_search_api():
    args = reqparse.RequestParser(). \
        add_argument("Table", type=str). \
        add_argument("Num", type=int, default=1). \
        parse_args()

    table_name = args['Table']
    if not table_name:
        table_name = DEFAULT_TABLE
    top_k = args['Num']
    file = request.files.get('file', "")
    if not file:
        return "no file data", 400
    if not file.name:
        return "need file name", 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        res_id, res_distance = do_search(
            table_name, file_path, top_k, model, graph, sess)
        if isinstance(res_id, str):
            return res_id
        res_img = [request.url_root + "data/" + x for x in res_id]
        res = dict(zip(res_img, res_distance))
        res = sorted(res.items(), key=lambda item: item[1])
        return jsonify(res), 200
    return "not found", 400


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0")
