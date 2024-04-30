# -*- coding: utf-8 -*-
# @Time    : 2021/1/25 6:45 PM
# @Author  : liuxiyang
from helper import *
from torch_scatter import scatter
from torch_scatter.scatter import scatter_sum
import torch
import torch.nn as nn
import inspect
import numpy as np
device = torch.device("cuda:0" )
from typing import Optional, Tuple
def scatter_(name, src, index, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if name == 'add':
        name = 'sum'
    assert name in ['sum', 'mean', 'max']

    out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
    
    return out[0] if isinstance(out, tuple) else out



class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    """

    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()
        
        self.message_args = inspect.getargspec(self.message)[0][1:]
        
        self.update_args = inspect.getargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        r"""The initial call to start propagating messages.
        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
        :obj:`"max"`), the edge indices, and all additional data which is
        needed to construct messages and to update node embeddings."""

        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':  # If arguments ends with _i then include indic
                tmp = kwargs[arg[:-2]]  # Take the front part of the variable | Mostly it will be 'x',
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])  # Lookup for head entities in edges
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]  # tmp = kwargs['x']
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])  # Lookup for tail entities in edges
            else:
                message_args.append(kwargs[arg])  # Take things from kwargs

        update_args = [kwargs[arg] for arg in self.update_args]  # Take update args from kwargs

        out = self.message(*message_args)
        
        out = scatter_(aggr, out, edge_index[0], dim_size=size)

        out = self.update(out, *update_args)

        return out



class WACConv(MessagePassing):
    def __init__(self, edge_index, edge_type, in_channels, out_channels, num_rels,num_ent, act=lambda x: x, params=None
                ):
        super(self.__class__, self).__init__()
        
        
        self.edge_index = edge_index
        self.edge_type = edge_type
        
        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ent =num_ent
        self.num_rels = num_rels
        self.act = act
        self.device = None
        
        
        self.attn_fc = nn.Linear(out_channels, 1, bias=False)

        self.w1_loop = get_param((in_channels, out_channels))
        

        
        self.w1_out = get_param((in_channels, out_channels))
        
        self.w_rel = get_param((out_channels, out_channels))

        
        
        self.loop_rel = get_param((1, in_channels))

        self.drop = torch.nn.Dropout(self.p.dropout)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        

        num_edges = self.edge_index.size(1) // 2 
        
        if self.device is None:
            self.device = self.edge_index.device
        self.in_index, self.out_index = self.edge_index[:, :num_edges], self.edge_index[:, num_edges:]
        
 
        self.in_type, self.out_type = self.edge_type[:num_edges], self.edge_type[num_edges:]
        
        
        
        self.loop_index = torch.stack([torch.arange(self.p.num_ent), torch.arange(self.p.num_ent)]).to(self.device)
        self.loop_type = torch.full((self.p.num_ent,),  2*self.num_rels, dtype=torch.long).to(self.device)
        

        num_ent = self.p.num_ent
        self.in_norm =  self.compute_norm(self.in_index, num_ent)
        
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.rel_weight1 = get_param(( 2*self.num_rels+1, in_channels))
        
        self.ent_weight1 = get_param((self.num_ent, in_channels))
        self.ent_weight2 = get_param((self.num_ent, in_channels))


    ## Graph convolutions
    def forward(self, x, rel_embed):
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
         
        
        selfloop_res1 = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,rel_embed=rel_embed,  rel_weight=self.rel_weight1, ent_weight=self.ent_weight1,ent_weight_loop=self.ent_weight2, edge_norm=None, mode='loop', w_str='w1_{}')
        
        
        original_res1 = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed,
                                  rel_weight=self.rel_weight1,ent_weight=self.ent_weight1, ent_weight_loop=self.ent_weight2, edge_norm=self.out_norm, mode='out', w_str='w1_{}')

        
        ## Attention aggregation
        out2 = self.agg_multi_head(self.drop(original_res1), self.out_index, 1)

            
        if self.p.bias:
            
            out=0.75*selfloop_res1+ self.bias +0.25*( out2)
        else:
            out=0.75*selfloop_res1 +0.25*( out2)

        relation1= rel_embed.mm(self.w_rel)

        out = self.bn(out)
        
        entity1 = out
        

        return entity1, relation1[:-1]

    ## Function for added Attention
    def agg_multi_head(self, in_res,index, head_no):
        
        edge_index = index
        all_message = in_res
        if head_no==1:
            a = self.attn_fc(all_message)
            a=a*all_message
        else:
            b = self.attn_fc(all_message)
            a=b*all_message
            
        return a
    

    ## Relation Embedding transformation
    def rel_transform1(self, ent_embed, rel_embed,ent_weight,rel_weight, opn=None):
        if opn is None:
            opn = self.p.opn
        
        if opn == 'W_mult':
            trans_embed=ent_embed * rel_embed * ent_weight + ent_embed * rel_weight
        
        else:
            raise NotImplementedError

        return trans_embed
    
    ## Self (loop) Relation Embedding transformation
    def rel_transform2(self, ent_embed,ent_weight, rel_embed, opn=None):
        if opn is None:
            opn = self.p.opn
        
        if opn == 'W_mult':
            
            trans_embed = ent_embed*ent_weight* rel_embed 

        else:
            raise NotImplementedError

        return trans_embed
    
    ## Gelu activation
    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))


    def message(self, x_j, edge_type, rel_embed,rel_weight, ent_weight,ent_weight_loop,edge_norm, mode, w_str):
        
        weight = getattr(self, w_str.format(mode))

        rel_emb = torch.index_select(rel_embed, 0, edge_type)

        if mode=='loop':
            
            xj_rel = self.rel_transform2(x_j, ent_weight_loop, rel_emb)

        else:
            rel_weight=torch.index_select(rel_weight, 0, edge_type)
            ent_weight=torch.index_select(ent_weight, 0, self.in_index[0])
            xj_rel = self.rel_transform1(x_j, rel_emb, ent_weight,rel_weight)
        
        out = torch.mm(xj_rel, weight)
        
        assert not torch.isnan(out).any()
        
        
        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        
        aggr_out = self.gelu(aggr_out)
        
        return aggr_out
    
    ## Edge Normalization
    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float().unsqueeze(1)
        deg=scatter_sum( edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)  
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  

        return norm

