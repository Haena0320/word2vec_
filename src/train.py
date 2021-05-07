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

    def __init__(self, config, args, writer1, writer2):

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

        #self.sentence_cnt = 27546389 # skip-neg : 90  # skip-hs : 63
        self.sentence_cnt = 19282472
        self.total_sentence_num = 30301028

        self.epochs = args.epochs
        self.writer1 = writer1
        self.writer2 = writer2

        if args.neg == 0:
            self.tree = data_loader(config.path_tree_list)
            self.hh_code = data_loader(config.path_h_code)
            

    def write_log(self, log, lr, step):
        self.writer1.add_scalar('train/log', log, step)
        self.writer2.add_scalar("train/lr", lr, step)

    def update_learning_rate(self, total_step, max_learning_rate, cnt):
        learning_rate = 1 - cnt / (total_step * self.epochs)
        if learning_rate <= 0.001:
            learning_rate = 0.001
        learning_rate *= max_learning_rate
        return learning_rate

    def init_train(self):
        print('init train..')
        total_count = sum(self.frequency.values())
        freqs = {word: count / total_count for word, count in self.frequency.items()}
        self.ran = {self.word2id[word]: (np.sqrt(freqs[word] / self.threshold) + 1) * (self.threshold / freqs[word]) for word in self.frequency.keys()}

        #self.W_in = 0.01 * np.random.randn(self.vocab_size, self.embedding_dim)  # (vocab_size, embedding_dim)
        with gzip.open("/hdd1/user15/workspace/word2vec/log/skip-hs/lr0.025/word2vec_63.pkl") as f:
            self.W_in = pickle.load(f)
        self.W_out = np.zeros((self.vocab_size, self.embedding_dim), dtype=np.float64)
        if self.neg > 0:
            self.sampler = UnigramTable(frequency=self.frequency, sample_size=5)


    def random_train_word(self, sentence):
        return [word for word in sentence if self.ran[word] > random.random()]

    def fit(self, args_log):
        f_cnt = 0
        for i in tqdm(self.data_list):
            print('data file : {}'.format(i))
            training_data = data_loader(i)
    
            f_cnt += 1
            if f_cnt > len(self.data_list):
                break
            for sentence in tqdm(training_data):
                if len(sentence) < 2:
                    continue
                target_words = self.random_train_word(sentence)
                learning_rate = self.update_learning_rate(self.total_sentence_num, self.max_learning_rate, self.sentence_cnt)

                if self.cbow:
                    contexts, targets = make_target(target_words=target_words, window=self.window_size, cbow=True)
                    if len(contexts) < 3:
                        continue
                    contexts_vec = [np.mean(self.W_in[context], axis=0) for context in contexts]
                    total_loss = 0

                    if self.neg > 0:
                        for idx, target in enumerate(targets):
                            context = contexts[idx]
                            n1 = contexts_vec[idx]
                            ns_indices = self.sampler.get_negative_sample(target)
                            classifiers = [(target, 1)] + [(sample, 0) for sample in ns_indices]

                            for s, label in classifiers:
                                z = np.matmul(n1, self.W_out[s].T)
                                p = sigmoid(z)
                                g = p - label
                                if label == 1:
                                    total_loss -= np.log(p + 1e-7)
                                    b = learning_rate * np.dot(g, self.W_out[s])

                                else:
                                    total_loss -= np.log(1 - p + 1e-7)
                                    b += learning_rate * np.dot(g, self.W_out[s])

                                self.W_out[s] -= learning_rate * np.dot(g, n1)

                            for c in context:
                                self.W_in[c] -= b

                    else:  # hs
                        for idx, target in enumerate(targets):
                            context = contexts[idx]
                            n1 = contexts_vec[idx]
                            classifiers = list(zip(self.tree[target], self.hh_code[target]))
                            b = np.zeros((self.embedding_dim,))
                            for s, label in classifiers:
                                z = np.dot(n1, self.W_out[s])
                                p = sigmoid(z)
                                g = p - label
                                if label == 1:
                                    total_loss -= np.log(p + 1e-7)
                                    b += learning_rate * np.dot(g, self.W_out[s])

                                else:
                                    total_loss -= np.log(1 - p + 1e-7)
                                    b += learning_rate * np.dot(g, self.W_out[s])
                                self.W_out[s] -= learning_rate * np.dot(g, n1)
                            b /= len(context)
                            for c in context:
                                self.W_in[c] -= b

                    self.sentence_cnt += 1
                    self.write_log(total_loss / len(contexts), learning_rate, self.sentence_cnt)
                else: # skip
                    contexts, targets = make_target(target_words=target_words, window=self.window_size,cbow=False)
                    if len(contexts) < 3:
                        continue
                    total_loss = 0

                    if self.neg > 0:
                        for idx, context_mini in enumerate(contexts):
                            target = targets[idx]
                            ns_indices = self.sampler.get_negative_sample(target, len(context_mini))
                            # positive 
                            s = np.matmul(self.W_in[context_mini], self.W_out[target].T)
                            s = sigmoid(s)
                            g = s - 1

                            dcontext_mini = learning_rate * np.outer(g, self.W_out[target])
                            dtarget = learning_rate * np.matmul(g.T, self.W_in[context_mini])
                            total_loss -= np.sum(np.log(s + 1e-7))
                            self.W_out[target] -= dtarget
                            # negative

                            for ns in ns_indices:
                                s = np.matmul(self.W_in[context_mini], self.W_out[ns].T)  # (4, 10), (5, 10).T
                                s = sigmoid(s)
                                g = s - 0  # 4,5

                                dcontext_mini += learning_rate * np.matmul(g, self.W_out[ns])
                                dns = learning_rate * np.matmul(g.T, self.W_in[context_mini])
                                total_loss -= np.sum(np.log(1 - s + 1e-7))
                                self.W_out[ns] -= dns
                            self.W_in[context_mini] -= dcontext_mini

                    else:  # hs
                        for idx, context_mini in enumerate(contexts):
                            target = targets[idx]
                            tree = self.tree[target]
                            hh_code = self.hh_code[target]
                            classifier = list(zip(tree, hh_code))
                            dcontext_mini = np.zeros_like(self.W_in[context_mini], dtype="float32")

                            for node, label in classifier:
                                s = np.matmul(self.W_in[context_mini], self.W_out[node].T)
                                s = sigmoid(s)
                                g = s - label
                                if label > 0:
                                    total_loss -= np.sum(np.log(s + 1e-7))
                                else:
                                    total_loss -= np.sum(np.log(1 - s + 1e-7))
                                dcontext_mini += learning_rate * np.outer(g, self.W_out[node])
                                self.W_out[node] -= learning_rate * np.matmul(g.T, self.W_in[context_mini])

                            self.W_in[context_mini] -= dcontext_mini

                    self.sentence_cnt += 1
                    self.write_log(total_loss/len(contexts),learning_rate, self.sentence_cnt)
            del training_data
            if f_cnt % 9 == 0:
                with gzip.open(os.path.join(args_log, 'word2vec_{}.pkl'.format(f_cnt)),"wb") as f:
                    pickle.dump(self.W_in, f)

        print("training finished...")
        return None

