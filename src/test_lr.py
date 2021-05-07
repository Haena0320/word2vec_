import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import time
import random

from src.prepro import make_target
from src.utils import *
from src.model import *


class Trainer:

    def __init__(self, config, args, writer):

        self.cbow = args.cbow
        self.neg = args.neg

        self.word2id = data_loader(config.path_word2id)
        self.frequency = data_loader(config.path_frequency)
        self.data_list = glob.glob(config.path_temp2 + "*")
        self.vocab_size = len(self.frequency.keys())
        self.window_size = args.window_size
        self.embedding_dim = args.dimensionality
        self.threshold = args.sample_threshold
        self.max_learning_rate = args.learning_rate

        self.epochs = args.epochs
        self.writer = writer
        self.total_sentence_num = 30301028
        self.sentence_cnt = 0

        if args.neg == 0:
            self.tree = data_loader(config.path_tree_list)
            self.hh_code = data_loader(config.path_h_code)

    def write_log(self, log, step):
        self.writer.add_scalar('train/log', log, step)
        self.writer.flush()

    def update_learning_rate(self, total_step, max_learning_rate, cnt):
        learning_rate = 1 - cnt / (total_step * self.epochs)
        if learning_rate <= 0.0001:
            learning_rate = 0.0001
        learning_rate *= max_learning_rate
        return learning_rate

    def init_train(self):
        pass


    def fit(self, args_log):
        f_cnt = 0
        lent = 0
        for i in tqdm(self.data_list):
            print('data file : {}'.format(i))
            training_data = data_loader(i)
            lent += len(training_data)
            f_cnt += 1
            if f_cnt > len(self.data_list):
                break
            for sentence in tqdm(training_data):
                if len(sentence) < 2:
                    continue
                learning_rate = self.update_learning_rate(self.total_sentence_num, self.max_learning_rate,self.sentence_cnt)
                self.sentence_cnt += 1
                self.write_log(learning_rate, self.sentence_cnt)
            del training_data

        print(lent)
        return None

