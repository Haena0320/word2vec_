import pickle
import gzip

from src.utils import *

# traversal method 부터 시작 (tree node 생성)
class Node:
    def __init__(self, left=None, right=None, value=None):
        self.left = left
        self.right = right
        self.value = value

    def children_node(self):
        return (self.left, self.right)


"""트리 생성 & 트리 노드 인덱스 지정 """


def build_tree(frequency):  # frequency : (id : frequency), 딕셔너리 타입
    # 튜플로 변환
    frequency = sorted(list(frequency.items()), key=lambda x: x[1])
    value = 0
    while len(frequency) > 1:
        w1, f1 = frequency[0]
        w2, f2 = frequency[1]
        frequency = frequency[2:]

        node = Node(w1, w2, value)
        value += 1  # 각 innder node 당 인덱스 할당
        if value % 1000000 == 0:
            print("%d M Node made.." % (value))
        frequency.append((node, f1 + f2))
        tree = sorted(frequency, key=lambda x: x[1])

    return tree


""" 각 단어에 대한 허프만 코드 생성"""


def build_HuffmanCode(data, code=""):
    if type(data) == str:  # 데이터가 leaf node 이면
        return {data: code}
    h_code = dict()
    left, right = data.children_node()
    h_code.update(build_HuffmanCode(left, True, code + '0'))
    h_code.update(build_HuffmanCode(right, False, code + "1"))
    return h_code


""" 노드 리스트 생성"""  # input : word, h_code, output: path node list
# root에서 모든 leaf로 가는 path 프린트해주는 함수
total_list = []


# root에서 모든 leaf로 가는 path 프린트해주는 함수
def nodePaths(root):
    path = []
    nodePathRec(root, path, 0)


# root에서 leaf 로 가는데 도와주는 함수
def nodePathRec(root, path, pathLen):
    if root is None:
        return None
    if (len(path) > pathLen):
        path[pathLen] = root.value
    else:
        try:
            path.append(root.value)
        except:
            pass
    pathLen += 1

    if type(root) == str:
        # 현재 노드(root) 가 leaf 노드이면 리스트 출력
        node_Array(root, path, pathLen)
    else:
        nodePathRec(root.left, path, pathLen)
        nodePathRec(root.right, path, pathLen)


# root-to-leaf path is stored

def node_Array(root, ints, len):
    n_list = []
    n_list.append(root)
    for i in ints[0:len]:
        n_list.append(i)
    total_list.append(n_list)


def tree(frequency_path, save_path):
    frequency = data_loader(frequency_path)
    # hierarchical tree build
    print("build tree..")
    tree = build_tree(frequency)
    print("build node index list..")
    nodePaths(tree[0][0])
    print("data save ..")
    data_save(total_list, save_path)
    return None



######################################

import gzip
import pickle
with gzip.open("/hdd1/user15/workspace/word2vec/data/1-billion-corpus/preprocessed/etc/frequency.gzip", 'rb') as f:
    frequency = pickle.load(f)
    
with gzip.open("/hdd1/user15/workspace/word2vec/data/1-billion-corpus/preprocessed/etc/word2id.gzip", 'rb') as f:
    word2id = pickle.load(f)

id_frequency = {word2id[k]:v for k, v in frequency.items()}


path, code = encode_huffman(id_frequency)

with gzip.open("/hdd1/user15/workspace/word2vec/data/1-billion-corpus/preprocessed/etc/tree_list.gzip", 'wb') as f:
    pickle.dump(path, f)

with gzip.open("/hdd1/user15/workspace/word2vec/data/1-billion-corpus/preprocessed/etc/hh_code.gzip", 'wb') as f:
    pickle.dump(code, f)


def encode_huffman(frequency):
    vocab_size = len(frequency.items())
    count = [v for k, v in frequency.items()] + [1e15] * (vocab_size - 1)
    print(len(count))
    parent = [0]*(2*vocab_size-2)
    binary = [0]*(2*vocab_size-2)
    
    p1 = vocab_size-1
    p2 = vocab_size
    for i in range(vocab_size-1):
        if p1 >= 0:
            if count[p1] < count[p2]:
                min1 = p1
                p1 -= 1

            else:
                min1 = p2
                p2 += 1

        else:
            min1 = p2
            p2 += 1

        if p2 >= 0:
            if count[p1] < count[p2]:
                min2 = p1
                p1 -=1
            else:
                min2 = p2
                p2 += 1

        else:
            min2 = p2
            p2 += 1

        count[vocab_size + i] = count[min1] + count[min2]
        parent[min1] = vocab_size + i
        parent[min2] = vocab_size + i
        binary[min2] = 1

    # Assign binary code and path pointers to each vocab word
    root_idx = 2 * vocab_size - 2
    total_path = []
    total_code = []
    for i, token in enumerate(range(len(id_frequency.items()))):
        path = []  # List of indices from the leaf to the root
        code = []  # Binary Huffman encoding from the leaf to the root

        node_idx = i
        while node_idx < root_idx:
            if node_idx >= vocab_size: path.append(node_idx)
            code.append(binary[node_idx])
            node_idx = parent[node_idx]
        path.append(root_idx)

        # These are path and code from the root to the leaf
        total_path.append([j - vocab_size for j in path[::-1]])
        total_code.append(code[::-1])
    return total_path, total_code


