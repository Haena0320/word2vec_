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