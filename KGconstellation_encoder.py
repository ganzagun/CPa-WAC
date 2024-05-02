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



## Loss Function
class KGModel(torch.nn.Module):
    def __init__(self, params):
        super(KGModel, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()
        
    def loss(self, pred, true_label):
        
        return self.bceloss(pred,true_label)

## Knowledge graph Embedding Model Initialization
class ConstellationBase(KGModel):
    def __init__(self, edge_index, edge_type, num_rel,num_ent, params=None):
        super(ConstellationBase, self).__init__(params)

        self.edge_index = edge_index
        
        self.edge_type = edge_type
        
        self.p.gcn_dim = self.p.embed_dim 

        self.device = self.edge_index.device
        
        ##Initializing Parameters of Entities and Relations
        self.init_ent_global, self.init_rel_global = self.initialize_parameter(self.p.num_nodes,self.p.num_relation, self.p.embed_dim)
        
        entities_mapping, rel_mapping,rel_length=Runner.mapping(self)
        entities_mapping=torch.from_numpy(entities_mapping[0]).type(torch.int64)
        rel_mapping=torch.from_numpy(rel_mapping[0]).type(torch.int64)

        self.init_embed=nn.Parameter(self.init_ent_global[entities_mapping])
        self.init_rel=nn.Parameter(self.init_rel_global[rel_mapping])
        self.init_rel2=self.init_rel_global[rel_mapping]

        ## Initializing Graph convolutional encoder
        self.conv1 = WACConv(self.edge_index, self.edge_type,  self.p.embed_dim, self.p.embed_dim, num_rel, num_ent,
                               act=self.act, params=self.p)
       
        ## Inducing Bias
        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

        self.rel_drop = nn.Dropout(0.1)
        
##Function to initialize Parameters of Entities and Relations         
    def initialize_parameter(self,num_nodes, num_relation,embed_dim):
        torch.manual_seed(args.seed)
        ent_global=get_param((num_nodes, embed_dim))
        ent_global=F.relu(ent_global)
    
        rel_global=get_param((2*num_relation, embed_dim))
        rel_global=F.relu(rel_global)
       

        return ent_global, rel_global
        

    def forward_base(self, sub, rel, drop1, drop2):
        init_rel =  self.init_rel
        
        ## Graph convolutional encoder
        ent_embed1, rel_embed1 = self.conv1(x=self.init_embed, rel_embed=init_rel)
        
        ent_embed1 = drop1(ent_embed1)

        final_ent = ent_embed1
        final_rel = rel_embed1
        
        sub_emb = torch.index_select(final_ent, 0, sub)
        rel_emb = torch.index_select(final_rel, 0, rel)
        
        return sub_emb, rel_emb, final_ent,final_rel

## Proposed Knowledge graph Embedding Model
class D1_conv(ConstellationBase):
    def __init__(self, edge_index, edge_type,edge_type_rev, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type,  params.num_rel,params.num_ent, params)
        self.embed_dim = self.p.embed_dim
        
        self.bn0 = torch.nn.BatchNorm1d(1)
        self.bn1 = torch.nn.BatchNorm1d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        
        ## Initializing Conv1D decoder
        self.m_conv1 = torch.nn.Conv1d(1, out_channels=self.p.num_filt, kernel_size=self.p.ker_sz,
                                       stride=1, padding=0,bias=True)
        
        ## Flattening and Dense layer
        self.flat_sz=self.p.num_filt*(self.p.embed_dim-self.p.ker_sz+1)
        
        self.fc1 = torch.nn.Linear(self.flat_sz, self.p.embed_dim,bias=True)

    def forward(self, sub, rel, neg_ents=None):
        sub_emb, rel_emb, all_ent,final_rel = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
        
        sub_emb = sub_emb.view(-1, 1, self.embed_dim)
        
        
        rel_emb = rel_emb.view(-1, 1, self.embed_dim)
        
        
        ## Elementwise multiplication of entities and relations
        stk_inp=torch.mul(sub_emb, rel_emb)
        
        ## Conv1D decoder
        x = self.m_conv1(stk_inp)
       
        x = self.bn1(x)
        x = F.relu(x)

        x = x.view(-1, (x.shape[1]*x.shape[2]))
        
        x = self.fc1(x)

        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = torch.mm(x, all_ent.transpose(1, 0))


        score = torch.sigmoid(x)

        return score,x,all_ent,final_rel

    
    
######### Runnnnnerrrrrrr #########

class Runner(object):
    
    ## Mapping Function for entities and relations generated by training each cluster separately
    def mapping(self):
        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in [str(self.p.cluster_num)+'traincluster', str(self.p.cluster_num)+'testcluster', str(self.p.cluster_num)+'validcluster']:
            for line in open('./data/{}/{}/{}.txt'.format(self.p.dataset, self.p.cluster_folder , split),encoding="utf8"):
                sub, rel, obj = map(str.lower, line.strip().split())
                ent_set.add(sub)
               
                rel_set.add(rel)
                ent_set.add(obj)
        
        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}
        
        change = dict.items(self.ent2id)
        change2= dict.items(self.rel2id)
        
        
        entities_mapping = list(change)
        entities_mapping=np.array(entities_mapping)
        entities_mapping=entities_mapping.astype(int)
    
        
        relation_mapping= list(change2)
        relation_mapping=np.array(relation_mapping)
        relation_mapping=relation_mapping.astype(int)
        
        rel_mapping=[]
        j=0
        for i in range(0,len(relation_mapping)*2):
            if i<len(relation_mapping):
                rel_mapping.append([relation_mapping[i,0],i])
            else:
                rel_mapping.append([relation_mapping[j,0]+self.p.num_relation,i])
                j=j+1
        rel_mapping=np.array(rel_mapping)
        rel_length=len(rel_mapping)
        
        
        
        return entities_mapping.T, rel_mapping.T,rel_length
        
    ## Data Loading and Data Preparation
    def load_data(self):
        

        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in [str(self.p.cluster_num)+'traincluster', str(self.p.cluster_num)+'testcluster', str(self.p.cluster_num)+'validcluster']:
            for line in open('./data/{}/{}/{}.txt'.format(self.p.dataset, self.p.cluster_folder , split),encoding="utf8"):
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

        for split in [str(self.p.cluster_num)+'traincluster', str(self.p.cluster_num)+'testcluster', str(self.p.cluster_num)+'validcluster']:
            aggregator_mapping=[]
            aggregator_dict=[]
            for line in open('./data/{}/{}/{}.txt'.format(self.p.dataset, self.p.cluster_folder , split),encoding="utf8"):
                sub, rel, obj = map(str.lower, line.strip().split())
                aggregator_mapping.append([sub,rel,obj])
                sub2, rel2, obj2=sub, rel, obj
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]

                self.data[split].append((sub, rel, obj))

                if split == str(self.p.cluster_num)+'traincluster':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in [str(self.p.cluster_num)+'testcluster', str(self.p.cluster_num)+'validcluster']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+ self.p.num_rel )].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)
        
        for (sub, rel), obj in self.sr2o.items():
                self.triples[str(self.p.cluster_num)+'traincluster'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
        
        for split in [str(self.p.cluster_num)+'testcluster', str(self.p.cluster_num)+'validcluster']:
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
            'train': get_data_loader(TrainDataset, str(self.p.cluster_num)+'traincluster', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, str(self.p.cluster_num)+'validcluster_head', self.p.test_batch_size),
            'valid_tail': get_data_loader(TestDataset, str(self.p.cluster_num)+'validcluster_tail', self.p.test_batch_size),
            'test_head': get_data_loader(TestDataset, str(self.p.cluster_num)+'testcluster_head', self.p.test_batch_size,shuffle=False),
            'test_tail': get_data_loader(TestDataset, str(self.p.cluster_num)+'testcluster_tail', self.p.test_batch_size,shuffle=False),
        }

        self.edge_index, self.edge_type, self.edge_type_rev = self.construct_adj()
        
    ## Adjacency Matrix construction for each Cluster
    def construct_adj(self):
     
        edge_index, edge_type, edge_type_rev = [], [], []

        for sub, rel, obj in self.data[str(self.p.cluster_num)+'traincluster']:
            edge_index.append((sub, obj))
            edge_type.append(rel)
            edge_type_rev.append(rel + self.p.num_rel)

        
        for sub, rel, obj in self.data[str(self.p.cluster_num)+'traincluster']:
            edge_index.append((obj, sub))
            edge_type.append(rel+ self.p.num_rel)
            edge_type_rev.append(rel)
        
        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)
        edge_type_rev = torch.LongTensor(edge_type_rev).to(self.device)

        return edge_index, edge_type, edge_type_rev
    
    ## Initializing model
    def __init__(self, params):

        self.p = params

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')
        

        self.load_data()

        self.model = D1_conv(self.edge_index, self.edge_type, self.edge_type_rev, params=self.p)
        

        self.model.to(self.device)
        self.optimizer = self.add_optimizer(self.model.parameters())
        
    ## Functions to save layers, weights or the entire model    
    def save_specific_layers(self, layer_names, filename):
        layers_to_save = {}
        for name in layer_names:
            layer = getattr(self.model, name)
            layers_to_save[name] = layer.state_dict()

        model_filename = f'{filename}_model.pt'
        torch.save(layers_to_save, model_filename)
        
    def save_model(self, filename):  
        model_filename = f'{filename}_model.pt'
        torch.save(self.model.state_dict(), model_filename)

    def add_optimizer(self, parameters):
   
        
        return torch.optim.AdamW(parameters, lr=self.p.lr, weight_decay=self.p.l2)

    ## Reading batch data
    def read_batch(self, batch, split):
    
    
        if split == 'train':

            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label, None, None
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
    
    ## Evaluation of specific clusters
    def evaluate(self, split, epoch):
     
        
        left_results,ranks,test_left_ranks,sub1,values1 = self.predict(split=split, mode='tail_batch')
        ranks1=ranks
        right_results,ranks,test_right_ranks,sub2,values2 = self.predict2(split=split, mode='head_batch')
        ranks2=ranks
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
        

        return results,ranks1,ranks2,test_left_ranks,test_right_ranks,sub1,sub2,values1,values2
    
    ## Prediction of Head Entity
    def predict(self, split='valid', mode='tail_batch'):
      
      
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            aggregator1=[]
            change, change2,rel_length=self.mapping()
        
            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                pred,values, all_ent,rel_aggregator = self.model.forward(sub, rel)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]

                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]
                ranks2=torch.argsort(pred, dim=1, descending=True)
                
                for i in range(0,(ranks2.cpu()).shape[0]):
                    lw=numpy.where(ranks2[i].cpu()==obj[i].cpu())
                    
                    lato=np.where(change[1]==(sub[i].cpu()).numpy())
                    lato=lato[0]
                    sub2=change[0,lato]
                    
                    
                    lace=np.where(change2[1]==(rel[i].cpu()).numpy())
                    lace=lace[0]
                    rel2=change2[0,lace]
                    
                    lato2=np.where(change[1]==(obj[i].cpu()).numpy())
                    lato2=lato2[0]
                    obj2=change[0,lato2]
                    
                    rong1=np.where(change[1]==(ranks2[i,0].cpu()).numpy())
                    rong1=rong1[0]
                    ronku1=change[0,rong1]
                    
                    rong2=np.where(change[1]==(ranks2[i,1].cpu()).numpy())
                    rong2=rong2[0]
                    ronku2=change[0,rong2]
                    
                    rong3=np.where(change[1]==(ranks2[i,2].cpu()).numpy())
                    rong3=rong3[0]
                    ronku3=change[0,rong3]
                    
                    
                    aggregator1.append([sub2[0],rel2[0],obj2[0],ronku1[0],ronku2[0],ronku3[0],lw[0]])


                ranks = ranks.float()
                
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)

                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)
                
        aggregator1=np.array(aggregator1)
        return results,ranks2,aggregator1,sub.shape,values
    
    
    ## Prediction of Tail Entity
    def predict2(self, split='valid', mode='head_batch'):
      
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            aggregator2=[]
            change, change2,rel_length=self.mapping()
            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                pred,values, all_ent,rel_aggregator = self.model.forward(sub, rel)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]
                
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]
                ranks2=torch.argsort(pred, dim=1, descending=True)
                for i in range(0,(ranks2.cpu()).shape[0]):
                    lw=numpy.where(ranks2[i].cpu()==obj[i].cpu())
                    
                    lato=np.where(change[1]==(sub[i].cpu()).numpy())
                    lato=lato[0]
                    sub2=change[0,lato]
                    
                    lace=np.where(change2[1]==(rel[i].cpu()).numpy())
                    lace=lace[0]
                    rel2=change2[0,lace]
                    
                    lato2=np.where(change[1]==(obj[i].cpu()).numpy())
                    lato2=lato2[0]
                    obj2=change[0,lato2]
                    
                    rong1=np.where(change[1]==(ranks2[i,0].cpu()).numpy())
                    rong1=rong1[0]
                    ronku1=change[0,rong1]
                    
                    rong2=np.where(change[1]==(ranks2[i,1].cpu()).numpy())
                    rong2=rong2[0]
                    ronku2=change[0,rong2]
                    
                    rong3=np.where(change[1]==(ranks2[i,2].cpu()).numpy())
                    rong3=rong3[0]
                    ronku3=change[0,rong3]
                    
                    
                    aggregator2.append([sub2[0],rel2[0],obj2[0],ronku1[0],ronku2[0],ronku3[0],lw[0]])
                      
                
                ranks = ranks.float()
                
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)
               
        aggregator2=np.array(aggregator2)
        return results,ranks2,aggregator2,sub.shape,values
    
    
    ## Model running and Evaluation for each Cluster
    def run_epoch(self, epoch, val_mrr=0):
       
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])
        
        
        sub_aggregator=[]
        for step, batch in enumerate(train_iter):
            
            self.optimizer.zero_grad()
            sub, rel, obj, label, neg_ent, sub_samp = self.read_batch(batch, 'train')

            pred,values, all_ent,rel_emb = self.model.forward(sub, rel, neg_ent)
            loss = self.model.loss(pred, label)
            

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
        sub_aggregator=np.array((all_ent.cpu().detach()))
        rel_aggregator=np.array((rel_emb.cpu().detach()))
        
        loss = np.mean(losses)
        return loss,values,sub_aggregator,rel_aggregator

    def fit(self):
       
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        proper_sub_aggregator=np.zeros((self.p.num_nodes,self.p.embed_dim))
        
        change,change2,rel_length=self.mapping()
        
        proper_rel_aggregator=np.zeros((2*self.p.num_relation,self.p.embed_dim))
        
        val_results = {}
        val_results['mrr'] = 0
        for epoch in range(self.p.max_epochs):
            train_loss,values,sub_aggregator,rel_aggregator = self.run_epoch(epoch, val_mrr)
            
            if ((epoch + 1) % 100 == 0):
                
                
                val_results,val_left_ranks,val_right_ranks,aggregator1,aggregator2,sub1,sub2,val_values1,val_values2 = self.evaluate('valid', epoch)
                print('train loss',train_loss, 'epoch:',epoch+1,'val_left_results:',val_results['left_mrr'],'val_right_results:',val_results['right_mrr'],'val_mrr:',val_results['mrr'],'val_hit@10:',val_results['hits@10'])
                
            if val_results['mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                                                             
                self.best_epoch = epoch
                self.best_sub_aggregator=sub_aggregator
                self.best_rel_aggregator=rel_aggregator

        ## Mapping of entities and relations according to order
        for map in range(0,rel_length):
            
            proper_rel_aggregator[change2[0,map]]=rel_aggregator[change2[1,map]]
        
        
        proper_rel_aggregator=torch.from_numpy(proper_rel_aggregator)

        
        for map in range(0,len(sub_aggregator)):
            proper_sub_aggregator[change[0,map]]=sub_aggregator[change[1,map]]
        
        proper_sub_aggregator=torch.from_numpy(proper_sub_aggregator)

        
        test_results,test_left_ranks,test_right_ranks,aggregator1,aggregator2,sub1,sub2,values1,values2 = self.evaluate('test', self.best_epoch)
        print('test_left_results:',test_results['left_mrr'],'test_right_results:',test_results['right_mrr'],'test_mrr:',test_results['mrr'],'test_hit@1:',test_results['hits@1'],'test_hit@3:',test_results['hits@3'],'test_hit@5:',test_results['hits@5'],'test_hit@10:',test_results['hits@10'])
        

        return proper_sub_aggregator, proper_rel_aggregator
        

        
        
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
    parser.add_argument('--num_relation', dest='num_relation',default=18,type=int , help='Number of relations in graph')
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
    parser.add_argument('--cluster_num', dest='cluster_num', default=2, type=int, help='cluster_num of cluster to train on')
    
    
    
    
    
    
    args = parser.parse_known_args()[0]


    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    import os

    dir_path = './data/wn18/clusters'
    count = 0
    
    ## Number of CLusters to be Trained on
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    count=int(count/3)
    cluster_ent = []
    cluster_rel=[]
    print('File count:', count)
    
    start_time = time.time()
    print('cluster',args.cluster_num)
    model = Runner(args)
    proper_sub_aggregator,proper_rel_aggregator=model.fit()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time) 
    print('end of training for cluster',x)
    
    
    encoder_op_path = './data/'+args.dataset+f'/{args.cluster_folder}/encoder_output/'
    if not os.path.exists(encoder_op_path):
        os.makedirs(encoder_op_path)
        
    print(f'Saving file at {encoder_op_path}')
    torch.save(cluster_ent, encoder_op_path+f'{args.cluster_num}_cluster_ent.pt')
    torch.save(cluster_rel, encoder_op_path+f'{args.cluster_num}_cluster_rel.pt')
    print(f'Embedding saved {args.cluster_num}')


        
# python KGconstellation_dencoder.py --dataset WN18RR --cluster_folder cluster_c2  --cluster_num 2 
