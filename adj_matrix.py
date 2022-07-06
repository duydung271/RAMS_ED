# import torch
from trankit import Pipeline
from eddataset import load_sentence_data
from tqdm import tqdm
import pickle
import numpy as np

p = Pipeline(lang='english', gpu=True, cache_dir='./cache')

def get_adj_matrix(words):
    dtree = p.posdep(words, is_sent=True)
    tokens = dtree['tokens']
    n = len(words)

    adj_matrix = np.zeros((n,n), dtype=np.int32)

    for token in tokens:
        u,v = token['id'] - 1 , token['head'] - 1
        adj_matrix[u][v] = adj_matrix[v][u] = 1

    return adj_matrix

if __name__ == '__main__':
    name = 'test'
    path = f'data/{name}.jsonlines'
    sentence_data = load_sentence_data(path)
    adj_matrixs = []

    for i, item in enumerate(tqdm(sentence_data, desc='Converting to adj matrixs')):
        words, labels = item
        adj_matrix = get_adj_matrix(words)
        adj_matrixs.append(adj_matrix)
    
    out_path = f'data/adj_matrixs/{name}.mat'
    with open(out_path, 'wb') as f:
        pickle.dump(adj_matrixs, f)

    
    # adj_matrixs = []
    # with open(out_path, 'rb') as f:
    #     adj_matrixs = pickle.load(f)
    # print(adj_matrixs[0].shape)
