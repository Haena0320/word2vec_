import numpy as np
from numpy import dot
import gzip
from tqdm import tqdm
import pickle
from numpy.linalg import norm

from src.utils import *


def analogy_parsing(path_analogy_raw, path_word2id, path_analogy_preprocessed):
    txt = txt_loader(path_analogy_raw)
    word2id = data_loader(path_word2id)

    ret = dict()
    cat = None
    n = len(txt)
    ix = 0
    end = txt.find("\n", ix)

    while ix < n:
        line = txt[ix:end].strip()
        ix = end + 1
        end = txt.find('\n', ix)

        if line[0] == ":":
            cat = line[1:].strip()
            ret[cat] = []

        elif line[0] == "#":
            pass
        else:
            try:
                [a, b, c, d] = [w for w in line.split()]
                ret[cat].append([word2id[a], word2id[b], word2id[c], word2id[d]])
            except:
                continue

    data_save(ret, path_analogy_preprocessed)
    return None


def cosine_similarity(word2vec, word_vec):
    """
    :param word2vec: (vocab_size,embedding_dim)
    :param word_vec: (N, embedding_dim) -> N : num of test words
    :return: (vocab_size, N)
    """
    assert len(word_vec.shape) == 2

    word2vec /= norm(word2vec, axis=1, keepdims=True)
    word_vec /= norm(word_vec, axis=1, keepdims=True)
    return np.matmul(word2vec, word_vec.T)


def accuracy_scoring(path_word2id, ret_path, word2vec, result_path):
    word2id = data_loader(path_word2id)
    id2word = {v: k for k, v in word2id.items()}
    ret = data_loader(ret_path)
    word2vec /= norm(word2vec)  # normalizing

    result = dict()
    for task, question in tqdm(ret.items()):
        correct = 0
        question = np.array(question)
        q_target = word2vec[list(question[:, 1])] - word2vec[list(question[:, 0])] + word2vec[list(question[:, 2])]
        q_target /= norm(q_target, axis=1, keepdims=True)
        target = question[:, 3]
        similarity = cosine_similarity(word2vec, np.array(
            q_target))  # input: target vectors (N, embedding_dims), output: (N, top)
        similarity = similarity.T

        for i in tqdm(range(len(question))):
            most_sim_idx = similarity[i].argsort()[-5:]  # top 5
            if target[i] in most_sim_idx:
                correct += 1
                # ls = [id2word[q] for q in question[i]]
                # ls2 = [id2word[q] for q in most_sim_idx]
                # print(ls)
                # print(ls2)

        result[task] = (correct, len(question), correct * 100 / len(question))

    print(result)

    cnt, sementic_c, sementic_q, syntactic_c, syntactic_q = 0, 0, 0, 0, 0
    for i in result.values():
        if cnt < 5:
            sementic_c += i[0]
            sementic_q += i[1]
        else:
            syntactic_c += i[0]
            syntactic_q += i[1]
        cnt += 1

    print("sementic accuracy: {}%".format(sementic_c * 100 / sementic_q))
    print("syntactic accuracy: {}%".format(syntactic_c * 100 / syntactic_q))

    data_save(result, result_path)
    return result



