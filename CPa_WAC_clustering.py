import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
import random
import collections
import scipy.stats
from sknetwork.clustering import Louvain
from utils import load_data, build_adjacency, cnt_nonzero
import argparse
import os


def load_kgr_data(agrs):
    print(f"Loading KGR Data for Dataset: {args.dataset}")
    entity2id, relation2id, train_triplets, valid_triplets, test_triplets = load_data('./data/'+ agrs.dataset)
    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets)))
    Adjacency_prime, Adjacency = build_adjacency(len(entity2id),train_triplets)
    return entity2id, relation2id, train_triplets, valid_triplets, test_triplets, all_triplets, Adjacency_prime, Adjacency 





def combine_noisy_lable(labels, Adjacency, delta):
    
    unique_classes = []
    class_counts = []
    below_limit_classes = [] 
    for i in range(1,np.max(labels)+1):
        correction = np.where(labels==i)
        length = len(correction[0])
        unique_classes.append(length)
        class_counts.append(i)
        if length < delta:
            below_limit_classes.append(i)
        
    unique_classes = np.array(unique_classes)
    class_counts=np.array(class_counts)
    
    below_limit_classes = np.array(below_limit_classes)
    
    
    if len(below_limit_classes) != 0:
        for i in range(0,len(below_limit_classes)):
            for j in range(0,len(labels)):
                if labels[j] == below_limit_classes[i]:
                    labels[j] = 0 
                
    labels = labels+1
    
    unique_classes_correction = []
    class_counts_correction = []
    for i in range(1,np.max(labels)+1):
        correction = np.where(labels==i)
        length = len(correction[0])
        unique_classes_correction.append(length)
        class_counts_correction.append(i)
    unique_classes_correction = np.array(unique_classes_correction)
    class_counts_correction=np.array(class_counts_correction)

    
    
    print("       Getting Clustering with high intermediate links")
    # highest number of links
    Adjacency_correction = []
    for i in range(0, len(Adjacency)):
        Adj = np.multiply(Adjacency[i],labels)
        Adjacency_correction.append(Adj)
    Adjacency_correction = np.array(Adjacency_correction)

    highest=[]
    for i in range(0,len(Adjacency_correction)):
        fining=np.where(Adjacency_correction[i]>0)
        fining=np.array(fining[0])

        cluster_nearest = []
        if len(fining)>1:
            for j in range(0,len(fining)):
                if Adjacency_correction[i][fining[j]]!=labels[i]:
                    cluster_nearest.append(labels[fining[j]])
                elif Adjacency_correction[i][fining[j]]==labels[i]:
                    cluster_nearest.append(0)
        elif len(fining)==1:
            cluster_nearest.append(0)
        cluster_nearest=np.array(cluster_nearest)

        clust_high=[]
        alpha = cnt_nonzero(cluster_nearest)
        if alpha>0:
            for k in range(0,len(cluster_nearest)):
                if cluster_nearest[k]>0:
                    clust_high.append(cluster_nearest[k])

        else:
            clust_high.append(0)
            clust_high=np.array(clust_high)

        highest.append(scipy.stats.mode(clust_high, nan_policy='omit')[0][0])
        
    
    print("       Combining Clusters with High links")
    #Cluster Adjacency
    all_cluster=[]
    for i in range(1,np.max(labels)+1):
        clus_adj=np.where(labels==i)
        max_count=[]
        cluster=[]
        for j in range(0,len(clus_adj[0])):
            highest_count=highest[clus_adj[0][j]]
            if highest_count!=0:
                max_count.append(highest_count)
        max_count=np.array(max_count)
        elements_count = collections.Counter(max_count)
        cluster.append([i,scipy.stats.mode(max_count, nan_policy='omit',keepdims=False)[0]])
        all_cluster.append(cluster) 

    all_cluster = np.array(all_cluster)

    
    return labels, unique_classes_correction, class_counts_correction, all_cluster
    
    
    
def combine_clusters_below_threshold(class_counts_correction, all_cluster, labels, Adjacency, threshold, sigma):
    
    for i in range(2,len(all_cluster)):
        if class_counts_correction[i] < threshold:
            correction=np.where(labels == all_cluster[i][0][0])
            for j in range(0,len(correction[0])):
                labels[correction[0][j]] = all_cluster[i][0][1]


    unique_classes_correction = []
    class_counts_correction = []
    for i in range(1,np.max(labels)+1):
        correction = np.where(labels==i)
        length = len(correction[0])
        unique_classes_correction.append(length)
        class_counts_correction.append(i)
    unique_classes_correction = np.array(unique_classes_correction)
    class_counts_correction=np.array(class_counts_correction)
    
    
    cluster_maximus=np.where(class_counts_correction > sigma)
    cluster_maximus=cluster_maximus[0]+1
    cluster_maximus=cluster_maximus.astype('f')
    
    
    print("       Getting Clustering with high intermediate links")
    # highest number of links
    Adjacency_correction = []
    for i in range(0, len(Adjacency)):
        Adj = np.multiply(Adjacency[i],labels)
        Adjacency_correction.append(Adj)
    Adjacency_correction = np.array(Adjacency_correction)

    highest=[]
    for i in range(0,len(Adjacency_correction)):
        fining=np.where(Adjacency_correction[i]>0)
        fining=np.array(fining[0])

        cluster_nearest = []
        if len(fining)>1:
            for j in range(0,len(fining)):
                if Adjacency_correction[i][fining[j]]!=labels[i] and all(Adjacency_correction[i][fining[j]]!=cluster_maximus[m] for m in range(0,len(cluster_maximus))):
                    cluster_nearest.append(labels[fining[j]])
                    
        elif len(fining)==1:
            cluster_nearest.append(0)
        cluster_nearest=np.array(cluster_nearest)

        clust_high=[]
        alpha = cnt_nonzero(cluster_nearest)
        if alpha>0:
            for k in range(0,len(cluster_nearest)):
                if cluster_nearest[k]>0:
                    clust_high.append(cluster_nearest[k])

        else:
            clust_high.append(0)
            clust_high=np.array(clust_high)

        highest.append(scipy.stats.mode(clust_high, nan_policy='omit')[0][0])
        
    
    print("       Combining Clusters with High links")
    #Cluster Adjacency
    all_cluster=[]
    for i in range(1,np.max(labels)+1):
        clus_adj=np.where(labels==i)
        max_count=[]
        cluster=[]
        for j in range(0,len(clus_adj[0])):
            highest_count=highest[clus_adj[0][j]]
            if highest_count!=0:
                max_count.append(highest_count)
        max_count=np.array(max_count)
        elements_count = collections.Counter(max_count)
        cluster.append([i,scipy.stats.mode(max_count, nan_policy='omit',keepdims=False)[0]])
        all_cluster.append(cluster) 

    all_cluster = np.array(all_cluster)

    
    return labels, unique_classes_correction, class_counts_correction, all_cluster
    
def calculate_head_tail_same(data_triplets):
    count=0
    for i in range(0,len(data_triplets)):
        if labels[data_triplets[i,0]]==labels[data_triplets[i,2]]:
            count+=1
    average = count/len(data_triplets)
    return average

def save_cluster(labels, data_triplets, cluster_path, cluster_set="traincluster" ):
    print(f'Saving for {cluster_set}')
    splitted_triple=[]
    triple_extra=[]
    for j in range(2, np.max(labels)+1):
        data_split=[]
        for i in range(0,len(data_triplets)):
            if labels[data_triplets[i,0]]==j and labels[data_triplets[i,2]]==j:
                    data_split.append(data_triplets[i])
        data_split=np.array(data_split)
        path = cluster_path+str(j)+f'{cluster_set}.txt'
        np.savetxt(path,data_split, fmt ='%.0f')
        print(f'     Saved for {cluster_set} having number of clusters as {j} at {path}')
            
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for clustering')
    parser.add_argument('--dataset', dest='dataset',type=str, default='WN18RR', help='Dataset to use, default: WN18RR')
    parser.add_argument('--beta', dest='beta',type=int, default=200)
    parser.add_argument('--delta', dest='delta',type=int, default=40)
    parser.add_argument('--gamma', dest='gamma',type=int, default=1)
    parser.add_argument('--sigma', dest='sigma',type=int, default=18000)
    parser.add_argument('--cluster_folder', dest='cluster folder',type=str, default='cluster_c2')
    
    args = parser.parse_known_args()[0]
    print(args)
    
    cluster_path = './data/'+args.dataset+f'/{args.cluster_folder}/'
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)
    
    
    entity2id, relation2id, train_triplets, valid_triplets, test_triplets, all_triplets, Adjacency_prime, Adjacency = load_kgr_data(args)
    
    
    print("Fitting Louvain Clustering")
    louvain = Louvain()
    labels = louvain.fit_predict(Adjacency_prime)
    labels_unique, counts = np.unique(labels, return_counts=True)
    labels = labels + 1
    
    
    
    print(f"Applying First Set of correction: Combining all cluster having size < (delta) {args.delta}")
    labels_correction_1, unique_classes_correction_1, class_counts_correction_1, all_cluster_1 = combine_noisy_lable(labels, Adjacency, args.delta)
    
    
    
    
    print(f"Applying Second Set of correction: Combining cluster having size < (gamma*beta*delta) {args.gamma*args.beta*args.delta}")
    threshold = args.gamma*args.beta*args.delta
    labels_correction_2, unique_classes_correction_2, class_counts_correction_2, all_cluster_2 = combine_clusters_below_threshold(class_counts_correction_1, all_cluster_1, labels_correction_1, Adjacency, threshold, args.sigma)
    
    
    
    print(f"Applying Third Set of correction: Combining cluster having size < (2*gamma*beta*delta) {2*args.gamma*args.beta*args.delta}")
    threshold = 2*args.gamma*args.beta*args.delta
    labels_correction_3, unique_classes_correction_3, class_counts_correction_3, all_cluster_3 = combine_clusters_below_threshold(class_counts_correction_2, all_cluster_2, labels_correction_2, Adjacency, threshold, args.sigma)
    
    
    
    
    print(f"Applying Fourth Set of correction: Combining cluster having size < (3*gamma*beta*delta) {3*args.gamma*args.beta*args.delta}")
    threshold = 3*args.gamma*args.beta*args.delta
    labels_correction_4, unique_classes_correction_4, class_counts_correction_4, all_cluster_4 = combine_clusters_below_threshold(class_counts_correction_3, all_cluster_3, labels_correction_3, Adjacency, threshold, args.sigma)
    
    
    
    
    print(f"Applying Fifth Set of correction: Combining cluster having size < (4*gamma*beta*delta) {4*args.gamma*args.beta*args.delta}")
    threshold = 4*args.gamma*args.beta*args.delta
    labels_correction_5, unique_classes_correction_5, class_counts_correction_5, all_cluster_5 = combine_clusters_below_threshold(class_counts_correction_4, all_cluster_4, labels_correction_4, Adjacency, threshold, args.sigma)
    
    
    
    pot = cnt_nonzero(class_counts_correction_5)
    while pot<np.max(labels_correction_5):
        rec=np.where(class_counts_correction_5==0)[0]
        if len(rec)==0:
            break
        for i in range(0,len(labels_correction_5)):
            if labels_correction_5[i]==np.max(labels_correction_5):
                labels_correction_5[i]=rec[0]+1
                class_counts_correction_5[rec[0]]=class_counts_correction_5[-1:]

    label_path = cluster_path + 'labels.txt'
    print(f"Saving Labels at {label_path}")
    np.savetxt(label_path,labels, fmt ='%.0f')
    
    train_average = calculate_head_tail_same(train_triplets)
    valid_average = calculate_head_tail_same(valid_triplets)
    test_average = calculate_head_tail_same(test_triplets)
    
    print(f"Percentage of head and tail in same cluster for train set is : {train_average*100}%")
    print(f"Percentage of head and tail in same cluster for valid set is : {valid_average*100}%")
    print(f"Percentage of head and tail in same cluster for test set is : {test_average*100}%")
    
    
    
    save_cluster(labels_correction_5, train_triplets, cluster_path, cluster_set="traincluster" )
    save_cluster(labels_correction_5, valid_triplets, cluster_path, cluster_set="validcluster" )
    save_cluster(labels_correction_5, test_triplets, cluster_path, cluster_set="testcluster" )
    
    
# python CPa_WAC_clustering.py --dataset WN18RR --beta 200 --delta 40 --gamma 1 --sigma 18000 --cluster_folder cluster_c2
    
    

    
    
    


        
    

    




