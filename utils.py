
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_add
from sknetwork.clustering import Louvain
from utils import *

def read_triplets(file_path, entity2id, relation2id):
    triplets = []
    entities=[]
    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))
    return np.array(triplets)


def load_data(file_path):

    print("load data from {}".format(file_path))
    if file_path=='./data/FB15k-237' or file_path=='./data/WN18RR' or file_path=='./data/wn18' or file_path=='./data/NELL995' or file_path=='./data/FB15K' :
        print('here')
        with open(os.path.join(file_path, 'entities.dict')) as f:
            entity2id = dict()

            for line in f:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        with open(os.path.join(file_path, 'relations.dict')) as f:
            relation2id = dict()

            for line in f:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)
    else:
        with open(os.path.join(file_path, 'entities.txt')) as f: 
            entity2id = dict()
            n_ent=0
            for line in f:
                entity = line.strip()
                entity2id[entity] = n_ent
                n_ent+=1
        
        with open(os.path.join(file_path, 'relations.txt')) as f:
            relation2id = dict()
            n_rel=0    
            for line in f:
                relation = line.strip()
                relation2id[relation] = n_rel
                n_rel += 1

    train_triplets= read_triplets(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
    valid_triplets = read_triplets(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
    test_triplets = read_triplets(os.path.join(file_path, 'test.txt'), entity2id, relation2id)

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))
    
    if file_path=='./data/FB15k-237' or file_path=='./data/WN18RR' or file_path=='./data/wn18' or file_path=='./data/NELL995' or file_path=='./data/FB15K' :
        return entity2id, relation2id, train_triplets, valid_triplets, test_triplets
    else:
        return entity2id, relation2id, train_triplets, valid_triplets, test_triplets

def build_adjacency(num_nodes, triplets):
    A = np.zeros((num_nodes,num_nodes))
    A_prime=np.zeros((num_nodes,num_nodes))
    for i in range(0,len(triplets)):
       
        A[triplets[i,0],triplets[i,2]]+=1
        A[triplets[i,2],triplets[i,0]]+=1
        A_prime[triplets[i,0],triplets[i,2]]=1
        A_prime[triplets[i,2],triplets[i,0]]=1
    return A, A_prime

def cnt_nonzero(num_nodes):
    count=0
    for i in range(0,len(num_nodes)):
        if num_nodes[i]!=0:
            count = count+1
    return count
    
