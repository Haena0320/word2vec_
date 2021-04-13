import sys, os
sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
from collections import Counter
import pickle
import re

from src.utils import *

class DataReader:
    def __init__(self, config, args):
        self.word2id = dict()
        self.token_cnt = 0
        self.sentence_cnt = 0
        self.frequency = dict()
        self.token_word = []
        self.config = config
        self.input_file_list = get_data(config.path_raw)
        self.path_temp = config.path_temp

        self.min_cnt = args.min_cnt

    def read_data(self, input_file):
        with open(input_file, "r", encoding="UTF-8") as fd:
            for line in fd:
                yield line

    def tokenizer(self):
        id, file_num = 0, 0
        for i in tqdm(self.input_file_list):
            a = self.read_data(i)
            file_num += 1
            while True:

                try:
                    line = next(a)
                    line = line.replace(".\n", "</s>")
                    line = line.split(" ")
                    sentence = []
                    for word in line:
                        self.frequency[word] = self.frequency.get(word, 0) + 1
                        if word not in self.word2id:
                            self.word2id[word] = id
                            id += 1
                        sentence.append(self.word2id[word])

                    self.token_word.append(sentence)
                except:
                    data_save(self.token_word, self.path_temp+"file_list" + str(file_num) + ".pk")
                    self.token_word = []
                    break

            if file_num > len(self.input_file_list):
                break

        print('reduce vocab under min count...')
        self.frequency = list(self.frequency.items())
        self.frequency.sort(key=lambda x: -x[1])

        new_frequency = tuple(filter(lambda x: x[1] >= self.min_cnt, self.frequency))  # 5 개 미만 제거
        new_word2id = dict()
        self.under_mincnt = list(set(self.frequency) - set(new_frequency))

        new_frequency = dict((x, y) for x, y in tuple(new_frequency))
        self.under_mincnt = dict((x, y) for x, y in tuple(self.under_mincnt))  # word : frequency

        self.word2id = {v: k for k, v in self.word2id.items()}
        data_save(self.word2id, self.config.path_id2word_original)

        del self.frequency, self.word2id

        cnt = 0
        for i in new_frequency.keys():
            if i not in self.under_mincnt.keys():
                new_word2id[i] = cnt
                cnt += 1

        print("Total embeddings: " + str(len(new_frequency.keys())))
        print('saving....')

        data_save(new_frequency, self.config.path_frequency)
        data_save(new_word2id, self.config.path_word2id)

        print("All finished...")
        return None

    def make_token(self):
        id2word_original= data_loader(self.config.path_id2word_original)
        word2id = data_loader(self.config.path_word2id)
        for i in tqdm(range(1, len(self.input_file_list) + 1)):
            new_data = []
            data = data_loader(self.path_temp+"file_list" + str(i) + ".pk")

            for sentence in data:
                new_sentence = []
                if len(sentence) > 1000:
                    sentence = sentence[:1000]
                for word in sentence:
                    word = id2word_original[word]
                    if word not in self.under_mincnt.keys():  # words
                        new_sentence.append(word2id[word])
                    else:
                        continue
                new_data.append(new_sentence)
            data_save(new_data, self.config.path_temp2 +"file_list"+str(i)+".pk")
            del data, new_data
        return None

import random
def make_target(target_words=None, window=5, cbow=True):
    contexts = []
    targets = []
    for word in target_words:
        context = []
        if cbow:
            for a in range(2 * window + 1):
                if a != window:
                    current = target_words.index(word) - window + a
                    if (current < 0) or (current >= len(target_words)):
                        continue
                    else:
                        context.append(target_words[current])
                else:
                    continue

        else:
            window = random.randint(3, window)
            for a in range(2 * window + 1):
                if a != window:
                    current = target_words.index(word) - window + a
                    if (current < 0) or (current >= len(target_words)):
                        continue
                    else:
                        context.append(target_words[current])
                else:
                    continue
        contexts.append(context)
        targets.append(word)
    return contexts, targets









