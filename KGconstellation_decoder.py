import sys
import xlsxwriter
import pandas as pd
from helper import *
from data_loader import *
import joblib
import numpy
import pickle
import traceback
from helper import *
from model.Conv_new import WACConv
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd
import numpy as np
import random
import gc



## Loss Function for Global Decoder
class BaseKG(torch.nn.Module):
    def __init__(self, params):
        super(BaseKG, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()
   

    def loss(self, pred, true_label):
        
        return self.bceloss(pred,true_label)

## Global Decoder Initialization
class MergingBase(BaseKG):
    def __init__(self, num_rel,num_ent, params):
        super(MergingBase, self).__init__(params)
        self.embed_dim=params.embed_dim
        self.device = torch.device('cuda')

        self.init_embed1=nn.Parameter(params.main_ent.to(torch.float32))
        self.init_rel1=nn.Parameter(params.main_rel.to(torch.float32))
        
        self.We=get_param((params.main_ent.shape[1], self.embed_dim))
        
        self.Wr=get_param((params.main_rel.shape[1], self.embed_dim))


        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        

        
        

    def forward_base(self, sub, rel, drop1, drop2,downsample):
        
        rel_embed1 =  self.init_rel1 if downsample==0 else self.init_rel1.mm(self.Wr)

        ent_embed1=self.init_embed1 if downsample==0 else self.init_embed1.mm(self.We)

        ent_embed1 = drop1(ent_embed1)

        final_ent2 = ent_embed1

        final_rel2 = rel_embed1

        sub_emb2 = torch.index_select(final_ent2, 0, sub)
        
        rel_emb2 = torch.index_select(final_rel2, 0, rel)

        return sub_emb2, rel_emb2, final_ent2,final_rel2
    

## Global Decoder Model    
class D1_MLP(MergingBase):
    def __init__(self, params=None):
        super(self.__class__, self).__init__(params.num_rel,params.num_ent, params)
        self.p = params
        self.embed_dim = self.p.embed_dim
        
        self.downsample=params.downsample
        self.bn0 = torch.nn.BatchNorm1d(1)
        self.bn1 = torch.nn.BatchNorm1d(self.p.embed_dim*self.p.cluster_total*2)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim*self.p.cluster_total) if self.downsample==0 else torch.nn.BatchNorm1d(self.p.embed_dim)
        
        
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        
        
        ## Two dense layers
        self.dimensions=self.p.cluster_total*self.embed_dim if self.downsample==0 else self.embed_dim
        self.fc1 = torch.nn.Linear(self.dimensions, 2*self.p.cluster_total*self.embed_dim)
        self.fc2 = torch.nn.Linear(2*self.p.cluster_total*self.embed_dim, self.dimensions)


    def forward(self, sub, rel, neg_ents=None):
        sub_emb2, rel_emb2, all_ent,final_rel= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop,self.downsample)
        
        sub_emb2 = sub_emb2.view(-1, 1, self.embed_dim*self.p.cluster_total) if self.downsample==0 else sub_emb2.view(-1, 1, self.embed_dim)

        rel_emb2 = rel_emb2.view(-1, 1, self.embed_dim*self.p.cluster_total) if self.downsample==0 else rel_emb2.view(-1, 1, self.embed_dim)
        
        
        stk_inp=torch.mul(sub_emb2, rel_emb2)
       
        x=stk_inp
        
        x = x.view(-1, (x.shape[1]*x.shape[2]))
       
        x=self.hidden_drop(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x=F.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x=F.relu(x)
        x=torch.mm(x,all_ent.transpose(1, 0))

        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)


        return score,all_ent,final_rel

######## Runner ########
    
class Runner(object):
    
    ## Data loading to train decoder
    def load_data(self):
        

        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train','test','valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split),encoding="utf8"):
                sub, rel, obj = map(str.lower, line.strip().split())
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)
        
        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}
 
        self.p.num_ent = len(self.ent2id)
        print('number of entities:',self.p.num_ent)
        
        self.p.num_rel = len(self.rel2id) // 2
        print('number of relations:',self.p.num_rel)
        self.p.embed_dim =  self.p.embed_dim

        self.data = ddict(list)
        
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            aggregator_mapping=[]
            aggregator_dict=[]
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split),encoding="utf8"):
                sub, rel, obj = map(str.lower, line.strip().split())
                aggregator_mapping.append([sub,rel,obj])
                sub2, rel2, obj2=sub, rel, obj
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]

                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+ self.p.num_rel )].add(sub)
                

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)
        
        for (sub, rel), obj in self.sr2o.items():
                self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
        
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)
        

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.test_batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.test_batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.test_batch_size,shuffle=False),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.test_batch_size,shuffle=False),
        }

    
    ## Parameters initialization
    def __init__(self, params):

        self.p = params
        

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')
        

        self.load_data()
        self.model = self.add_model()
        self.optimizer = self.add_optimizer(self.model.parameters())
        
    ## Loading Decoder Model
    def add_model(self):
        
        model = D1_MLP(params=self.p)
        
        
        model.to(self.device)
        return model

    def add_optimizer(self, parameters):
   
        return torch.optim.AdamW(parameters, lr=0.005, weight_decay=0.001)
    
    ## Reading Batches
    def read_batch(self, batch, split):
    
    
        if split == 'train':

            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label, None, None
            
        else:
            triple, label = [_.to(self.device) for _ in batch]
            
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    ## Validation and testing
    def evaluate(self, split, epoch):
     
        
        left_results = self.predict(split=split, mode='tail_batch')
        
        right_results = self.predict2(split=split, mode='head_batch')
        
        results = get_combined_results(left_results, right_results)
        res_mrr = '\n\tMRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_mrr'],
                                                                              results['right_mrr'],
                                                                              results['mrr'])
        res_mr = '\tMR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_mr'],
                                                                          results['right_mr'],
                                                                          results['mr'])
        res_hit1 = '\tHit-1: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_hits@1'],
                                                                               results['right_hits@1'],
                                                                               results['hits@1'])
        res_hit3 = '\tHit-3: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_hits@3'],
                                                                               results['right_hits@3'],
                                                                               results['hits@3'])
        res_hit10 = '\tHit-10: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(results['left_hits@10'],
                                                                               results['right_hits@10'],
                                                                               results['hits@10'])
        

        return results
    
    ## Head Entity Prediction
    def predict(self, split='valid', mode='tail_batch'):
      
      
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            
            
        
            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)

                pred, all_ent,rel_aggregator = self.model.forward(sub, rel)

                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]

                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]

                ranks = ranks.float()
                
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)

                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)
                
        
        return results
    
    
    ## Tail Entity Prediction
    def predict2(self, split='valid', mode='head_batch'):
      
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            
            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)

                pred, all_ent,rel_aggregator = self.model.forward(sub, rel)

                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]

                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]
   
                ranks = ranks.float()
                
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)

                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)
               
        
        return results

    
    ## Run Epoch
    def run_epoch(self, epoch, val_mrr=0):
       
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])
        
        
        sub_aggregator=[]
        
        for step, batch in enumerate(train_iter):
            
            self.optimizer.zero_grad()
            sub, rel, obj, label, neg_ent, sub_samp = self.read_batch(batch, 'train')
            
            pred, all_ent,rel_emb = self.model.forward(sub, rel, neg_ent)
            
            loss = self.model.loss(pred, label)
            

            loss.backward()
            
            losses.append(loss.item())
            self.optimizer.step()
        sub_aggregator=np.array((all_ent.cpu().detach()))
        rel_aggregator=np.array((rel_emb.cpu().detach()))
        
        loss = np.mean(losses)
        return loss,sub_aggregator,rel_aggregator

    ## Global Decoder training for Batches
    def fit(self):
       
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.

        val_results = {}
        val_results['mrr'] = 0
        for epoch in range(self.p.max_epochs):
            train_loss,sub_aggregator,rel_aggregator = self.run_epoch(epoch, val_mrr)
            if ((epoch + 1) % 1 == 0):
                val_results = self.evaluate('valid', epoch)
                print('train loss',train_loss, 'epoch:',epoch+1,'val_mrr:',val_results['mrr'],'val_hit@10:',val_results['hits@10'])

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                                                             
                self.best_epoch = epoch
                self.best_sub_aggregator=sub_aggregator
                self.best_rel_aggregator=rel_aggregator


        test_results = self.evaluate('test', self.best_epoch)
        print('test_mrr:',test_results['mrr'],'test_hit@1:',test_results['hits@1'],'test_hit@3:',test_results['hits@3'],'test_hit@5:',test_results['hits@5'],'test_hit@10:',test_results['hits@10'])
        





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Parser For Arguments')

    parser.add_argument('--dataset', dest='dataset',type=str, default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('--opn', dest='opn', type=str, default='W_mult', help='Composition Operation to be used in D1')
    parser.add_argument('--batch', dest='batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('--test_batch', dest='test_batch_size', default=1024, type=int,
                        help='Batch size of valid and test data')

    parser.add_argument('--gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    
    parser.add_argument('--epoch', dest='max_epochs', type=int, default=2, help='Number of epochs')
    
    
    parser.add_argument('--l2', type=float, default=0.001, help='L2 Regularization for Optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', dest='lbl_smooth', type=float, default=0.0, help='Label Smoothing')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of processes to construct batches')
    parser.add_argument('--seed', dest='seed', default=41504, type=int, help='Seed for randomization')

    parser.add_argument('--bias', dest='bias', action='store_true', help='Whether to use bias in the model')
    parser.add_argument('--num_nodes', dest='num_nodes',default=40943,type=int , help='Number of nodes in graph')
    parser.add_argument('--num_relation', dest='num_relation',default=11,type=int , help='Number of relations in graph')
    parser.add_argument('--embed_dim', dest='embed_dim', default=200, type=int,
                        help='Embedding dimension to give as input to score function')

    parser.add_argument('--drop', dest='dropout', default=0.3, type=float, help='Dropout to use')
    parser.add_argument('--hid_drop', dest='hid_drop', default=0.3, type=float, help='D1: Hidden dropout')
    parser.add_argument('--hid_drop2', dest='hid_drop2', default=0.2, type=float, help='D1: Hidden dropout2')
    
    parser.add_argument('--feat_drop', dest='feat_drop', default=0.2, type=float, help='D1: Feature Dropout')

    parser.add_argument('--num_filt', dest='num_filt', default=200, type=int,
                        help='D1: Number of filters in convolution')
    parser.add_argument('--ker_sz', dest='ker_sz', default=41, type=int, help='D1: Kernel size to use')
    parser.add_argument('--downsample', dest='downsample', default=0, type=int, help='Reduce the embedding dimesnion')
    parser.add_argument('--cluster_folder', dest='cluster_folder', default='cluster_c2', type=str, help='Folder of cluster ex: cluster_c2')
    parser.add_argument('--cluster_total', dest='cluster_total', default=2, type=int, help='Total number of clusters')
        
    args = parser.parse_known_args()[0]
    print(args)
    


    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
        
    encoder_op_path = './data/'+args.dataset+f'/{args.cluster_folder}/encoder_output/'
    
    cluster_ent = []
    cluster_rel = []
    for cluster_num in range(2,args.cluster_total+2):
        cluster_ent_tmp = torch.load( encoder_op_path+f'{cluster_num}_cluster_ent.pt')
        cluster_rel_tmp = torch.load(encoder_op_path+f'{cluster_num}_cluster_rel.pt')
        cluster_ent.append(cluster_ent_tmp)
        cluster_rel.append(cluster_rel_tmp)
        
    cluster_path = './data/'+args.dataset+f'/{args.cluster_folder}/'
    label_path = cluster_path + 'labels.txt'
    label_prime=np.loadtxt(label_path, dtype=int)
    main_ent = cluster_ent[0].clone()
    main_rel = cluster_rel[0].clone()

    ## Concatenation of entity and relation embeddings from different clusters
    for i in range(1,len(cluster_ent)):
        main_ent=torch.cat((main_ent, cluster_ent[i]), 1)
        main_rel=torch.cat((main_rel, cluster_rel[i]),1)
        

    args.main_ent = main_ent
    args.main_rel = main_rel
    
    start_time = time.time()
    model = Runner(args)
    
    model.fit()
    end_time = time.time()
    
    ## Recorded Time to train Global Decoder
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)


    
    
    

