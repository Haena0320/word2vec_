import argparse
from distutils.util import strtobool as _bool
import pickle
import numpy as np
from tqdm.auto import tqdm
import gzip
import os
import sys
sys.path.append(os.getcwd())
from src.functions import *

############################### Init Net ##################################
# load analogy data
with open("data/analogy/questions-words.txt", 'r') as fr:
    loaded = fr.readlines()
count = 0
semantic = []
syntactic = []
for line in loaded:
    if line[0] == ':':
        count += 1
        continue
    elif line == '\n':
        continue
    if count < 6:
        semantic.append(line.split())
    else:
        syntactic.append(line.split())

# load word vectors
file="/hdd1/user15/workspace/word2vec/log/cbow-hs/lr0.025/word2vec_99.pkl"


with gzip.open(file, 'rb') as fr:
    word_vectors = pickle.load(fr)


# load dictionary
with gzip.open("data/1-billion-corpus/preprocessed/etc/word2id.gzip", 'rb') as fr:
    word_to_id  = pickle.load(fr)

id_to_word = {v: k for k, v in word_to_id.items()}
vocabulary = list(word_to_id.keys())

# Check whether my word vectors contain all words in questions
valid_sem = checkValid(semantic, vocabulary)
valid_syn = checkValid(syntactic, vocabulary)
print("valid semantic: %d/%d" %(len(valid_sem), len(semantic)))
print("valid syntactic: %d/%d" %(len(valid_syn), len(syntactic)))


############################### Start evaluate ##################################
batch_size = 20
batch1 = len(valid_syn)//batch_size
batch2 = len(valid_sem)//batch_size
syn_counts = 0
sem_counts = 0
for i in range(batch_size):
    print('data: {}/{}'.format(i+1, batch_size))
    batch_syn = valid_syn[i*batch1: (i+1)*batch1]
    batch_sem = valid_sem[i*batch2: (i+1)*batch2]

    # syntactic
    a1, b1, c1, d1 = convert2vec(batch_syn, word_vectors, word_to_id)
    predict_syn = b1 - a1 + c1
    similarity_syn = cos_similarity(predict_syn, word_vectors)
    syn_max_top4, syn_sim_top4, syn_count = count_in_top4(similarity_syn, id_to_word, batch_syn)
    syn_counts += syn_count

    #semantic
    a2, b2, c2, d2 = convert2vec(batch_sem, word_vectors, word_to_id)
    predict_sem = b2 - a2 + c2
    similarity_sem = cos_similarity(predict_sem, word_vectors)
    sem_max_top4, sem_sim_top4, sem_count = count_in_top4(similarity_sem, id_to_word, batch_sem)
    sem_counts += sem_count

syn_acc = syn_counts/len(valid_syn) * 100
sem_acc = sem_counts/len(valid_sem) * 100
print("syntactic accuracy: ", syn_acc)
print("semantic accuracy: ", sem_acc)
print("total accuracy: ", (syn_acc*len(valid_syn) + sem_acc*len(valid_sem))/
                                                             (len(valid_syn)+len(valid_sem)))
