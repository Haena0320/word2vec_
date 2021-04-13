import sys, os
sys.path.append(os.getcwd())

import argparse
from torch.utils.tensorboard import SummaryWriter
import gzip
import pickle

from src.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="skip-neg")
parser.add_argument('--config', type=str, default='default') # hyperparams
parser.add_argument('--log', type=str, default='loss')
parser.add_argument('--epochs', type=int, default=1)

parser.add_argument('--cbow', type=int, default=1) # skip 0, cbow 1
parser.add_argument('--neg', type=int, default=1) # neg : 1, hs : 0
parser.add_argument('--window_size', type=int, default=5)
parser.add_argument('--dimensionality', type=int, default=300)
parser.add_argument('--sample_threshold', type=int, default=1e-3)

parser.add_argument('--learning_rate', type=float, default=0.025)
args = parser.parse_args()
config = load_config(args.config)

assert args.model in ['skip-neg', 'skip-hs', "cbow-neg", "cbow-hs"]

lg = get_logger()
oj = os.path.join


from src.model import *
from src.train import Trainer as trainer

args.log = 'log/{}'.format(args.model)
lr_loc = oj(args.log, "lr"+str(args.learning_rate))
tb_loc = oj(lr_loc, "tb")

if not os.path.exists(args.log): # make log save directory
    os.mkdir(args.log)
    os.mkdir(lr_loc)
    os.mkdir(tb_loc)

writer = SummaryWriter(tb_loc) # log writer
trainer = trainer(config, args, writer)
trainer.init_train()
print("Total Epoch : {}".format(args.epochs))

for epoch in range(args.epochs):
    trainer.fit(lr_loc)








