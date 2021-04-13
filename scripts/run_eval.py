import sys, os

sys.path.append(os.getcwd())
import argparse
from src.util import *

import torch

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="skip-gram-negative")
parser.add_argument("--mode", type=str, default="word")
parser.add_argument('--config', type=str, default='default') # hyperparams

parser.add_argument('--log', type=str, default="analogy")
parser.add_argument('--gpu', type=str, default=None)

parser.add_argument("--epoch", type=int, default=1)


args = parser.parse_args()
config = load_config(args.config)

assert args.mode in ['word', 'phrase']

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else "cpu")

assert args.model in ['skip-gram-negative', 'skip-gram-hierarchical', "cbow-negative", "cbow-hierarchical"]

oj = os.path.join

word2vec_load = oj(config.path_word2vec, args.model)
word2vec = data_loader(word2vec_laod)

if config.mode == "path_analogy_word":
    questions = analogy_parsing(config.path_analogy_word, config.path_word2id)

else:
    questions = analogy_parsing(config.path_analogy_phrase, config.path_word2id)

result_path= oj(config.path_analogy_result, args.model, args.mode, str(args.epoch))

result = accuracy_scoring(questions,word2vec)
data_save(result, result_path)

lg.info("Done")

