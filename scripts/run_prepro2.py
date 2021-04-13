import sys, os
sys.path.append(os.getcwd())

from tqdm import tqdm
import argparse
import glob

from src.utils import load_config, data_loader
from src.prepro import make_target
from src.model import UnigramSampler_CB

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="skip-gram-negative")
parser.add_argument('--mode', type=str, default="word")
parser.add_argument('--min_cnt', type=int, default=5)
parser.add_argument('--win_size', type=int, default=5)
parser.add_argument('--sample_size', type=int, default=5)
parser.add_argument('--power', type=int, default=0.75)
parser.add_argument('--config', type=str, default='default') # hyperparams
args = parser.parse_args()

# load_ word frequency dictionary || corpus || negative sampler ||
# Load word frequency dictionary e.g. the 12322 home 123


config = load_config(args.config)
print("data load...")
frequency = data_loader(config.path_frequency)
word2id = data_loader(config.path_word2id)
id_freq = {word2id[k]:v for k, v in frequency.items()}
del word2id, frequency

data_list = os.path.join(config.path_trainingdata, "*")
data_list= glob.glob(data_list)

print("data write...")
cnt = 0
for i in data_list:
    cnt +=1
    data = data_loader(i)
    sampler = UnigramSampler_CB(id_freq, args.power, args.sample_size)
    f = open(config.path_cbow_preprocessed + 'cbow_winSize-{}_{}_NoNeg.txt'.format(args.win_size, cnt), 'w')

    for line in tqdm(data):
        if len(line) <= args.win_size * 2:
            continue

        for idx in range(args.win_size, len(line) - args.win_size):
            c_idx = line[idx]
            contexts_idx = list()
            for _idx in range(idx-args.win_size, idx):
                con_idx = line[_idx]
                contexts_idx.append(con_idx)

            for _idx in range(idx+1, idx+args.win_size+1):
                con_idx = line[_idx]
                contexts_idx.append(con_idx)
            ns = sampler.get_negative_sample_bundle(contexts_idx)
            #print(contexts_idx)
            #print(c_idx)
            #print(ns)
            for con_idx in contexts_idx:
                f.write(str(con_idx) + ' ')
            f.write('\t')
            f.write(str(c_idx))
            f.write('\t')
            for sample in ns:
                f.write(str(sample) + ' ')
            f.write('\n')
    f.close()
print("Done ! ")

from tqdm import tqdm

def split_dataset(path, length=469138928, file_num=100, save_path=path_etc):
    # length : 469138928
    if length is None:
        with open(path, "r") as f:
            length = 0
            while (1):
                a = f.readline()
                if a == "":
                    break
                else:
                    length += 1

                if length % 1000000 == 0:
                    print(length)
        print(length)

    file_name = [save_path + "split_%d.txt" % i for i in range(file_num)]

    num_of_line = length // file_num
    remain = length % file_num
    with open(path, "r") as f:

        for idx, file in enumerate(file_name):
            g = open(file, "w")

            if idx == file_num - 1:
                num_of_line += remain

            for i in tqdm(range(num_of_line), "split %d file" % idx):
                a = f.readline()
                g.write(a)

            g.close()


#if __name__ == "__main__":
#    split_dataset(path_etc+"cbow_winSize-5_NoNeg.txt")




