import sys, os
sys.path.append(os.getcwd())

import argparse
from src.utils import *
from src.prepro import DataReader as prepro # model Preprocessor
from src.tree import *

parser = argparse.ArgumentParser()
parser.add_argument('--prepro', type=str, default="..")
parser.add_argument('--hierarchical', type=int, default=0)
parser.add_argument('--mode', type=str, default="word")
parser.add_argument('--min_cnt', type=int, default="5")

parser.add_argument('--config', type=str, default='default') # hyperparams
args = parser.parse_args()
config = load_config(args.config)

preprocessor = prepro(config, args)
preprocessor.tokenizer()
preprocessor.make_token()

if args.mode == "phrase":
    # preprocessor.make_phrase()
    pass

#make tree
#tree(config.path_frequency, config.path_tree)


