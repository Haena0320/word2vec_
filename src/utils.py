import logging
import os
import json
import glob
import gzip
import pickle

class DictObj(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def __getitem__(self, key):
        return getattr(self, key)

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Dictobj(value) if isinstance(value, dict) else value

def load_config(conf):
    with open(os.path.join('config', '{}.json'.format(conf)), 'r') as f:
        config =json.load(f)
    return DictObj(config)

def get_logger(level="info", name="log"):
    lg = logging.getLogger(name) # log instance 설정
    
    if level =="debug":
        lg.setLevel(logging.DEBUG) # log level 설정
    else:
        lg.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler() # handler instance설정
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] :: %(message)s')

    stream_handler.setFormatter(formatter) # formatter 지정
    lg.addHandler(stream_handler) # handler 추가

    lg.info('Logger Module Initialized, Set log level {}'.format(level))
    return lg

def get_data(path_raw):
    files = glob.glob(path_raw+"*")
    return files

def data_loader(data_path):
    with gzip.open(data_path, "rb") as f:
        data = pickle.load(f)
    return data

def data_save(data, data_path):
    with gzip.open(data_path, 'wb') as f:
        pickle.dump(data, f)

def get_train_data(txt_path):
    with open(txt_path, encoding="utf-8") as f:
        data = f.read()
    return data











