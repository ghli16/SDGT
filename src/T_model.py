from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import time
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn.functional as F
import networkx as nx
from utils import *
from torch.utils.data.dataloader import DataLoader
from GCN import GCN
class MultiheadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        set_seed()
        assert in_dim % num_heads == 0
        self.in_dim = in_dim
        self.hidden_dim = out_dim
        self.num_heads = num_heads
        self.depth = in_dim // num_heads
        self.out_dim = out_dim
        self.query_linear = nn.Linear(in_dim, in_dim)
        self.key_linear = nn.Linear(in_dim, in_dim)
        self.value_linear = nn.Linear(in_dim, in_dim)

        self.output_linear = nn.Linear(in_dim, out_dim)

    def res_para(self):
        set_seed()
        self.query_linear.reset_parameters()
        self.key_linear.reset_parameters()
        self.value_linear.reset_parameters()
        self.output_linear.reset_parameters()

    def split_heads(self, x, batch_size):
        # reshape input to [batch_size, num_heads, seq_len, depth]
        set_seed()
        x_szie = x.size()[:-1] + (self.num_heads, self.depth)
        x = x.reshape(x_szie)
        # transpose to [batch_size, num_heads, depth, seq_len]
        return x.transpose(-1, -2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)


        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))


        if mask is not None:
            mask = mask.unsqueeze(1)  # add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=1)
        attention_output = torch.matmul(attention_weights, V)

        output_size = attention_output.size()[:-2]+ (query.size(1),)
        attention_output = attention_output.transpose(-1, -2).reshape((output_size))


        attention_output = self.output_linear(attention_output)

        return torch.sigmoid(attention_output)



class GraphTransformerLayer(nn.Module):
    def __init__(self, node_features, gcn1_features, gcn2_features, gcn3_features, in_dim, hidden_dim, fout_dim, num_heads, dropout, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False, ):
        super().__init__()
        set_seed()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.fout_dim = fout_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attention = MultiheadAttention(in_dim, hidden_dim, num_heads)
        self.residual_layer1 = nn.Linear(node_features, fout_dim)  #残差
        self.O = nn.Linear(hidden_dim, fout_dim)
        self.node_features = node_features
        self.gcn1_features = gcn1_features
        self.gcn2_features = gcn2_features
        self.gcn3_features = gcn3_features

        self.gcn_Q = GCN(self.node_features, self.gcn1_features, self.gcn2_features, self.gcn3_features)
        self.gcn_K = GCN(self.node_features, self.gcn1_features, self.gcn2_features, self.gcn3_features)
        self.gcn_V = GCN(self.node_features, self.gcn1_features, self.gcn2_features, self.gcn3_features)
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(fout_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(fout_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(fout_dim, fout_dim * 2)
        self.FFN_layer2 = nn.Linear(fout_dim * 2, fout_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(fout_dim)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(fout_dim)
    def res_para(self):
        set_seed()
        self.residual_layer1.reset_parameters()
        self.O.reset_parameters()
        self.attention.res_para()
        self.layer_norm1.reset_parameters()
        self.layer_norm2.reset_parameters()
        # self.batch_norm1.reset_parameters()
        # self.batch_norm2.reset_parameters()
        self.FFN_layer1.reset_parameters()
        self.FFN_layer2.reset_parameters()


    def forward(self, data, datap):
        pos = datap.x
        fea = data.x
        # print("pos:{}".format(pos.size()))
        # print("fea:{}".format(fea.size()))

        x = torch.cat((fea, pos), dim=-1)
        q = self.gcn_Q(data, x)
        k = self.gcn_K(data, x)
        v = self.gcn_V(data, x)
        h_in1 = self.residual_layer1(x)


        attn_out = self.attention(q, k, v)

        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = F.leaky_relu(self.O(attn_out))
        if self.residual:
            attn_out = h_in1 + attn_out
        if self.layer_norm:
            attn_out = self.layer_norm1(attn_out)

        if self.batch_norm:
            attn_out = self.batch_norm1(attn_out)

        h_in2 = attn_out

        # FFN
        attn_out = self.FFN_layer1(attn_out)
        attn_out = F.leaky_relu(attn_out)
        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = self.FFN_layer2(attn_out)
        attn_out = F.leaky_relu(attn_out)

        if self.residual:
            attn_out = h_in2 + attn_out

        if self.layer_norm:
            attn_out = self.layer_norm2(attn_out)

        if self.batch_norm:
            attn_out = self.batch_norm2(attn_out)
        return attn_out

class GTM_net(nn.Module):
    def __init__(self, gcn2_features_res, out_dim_res, fout_dim_res, head_num_res, dropout_res, node_features_res, gcn1_features_res, gcn3_features_res,
                 L_res, output_t_res, layer_norm, batch_norm, residual):
        super().__init__()

        set_seed()
        in_dim = gcn2_features_res
        out_dim = out_dim_res
        fout_dim = fout_dim_res
        head_num =head_num_res
        dropout = dropout_res
        node_features =node_features_res
        gcn1_features =gcn1_features_res
        gcn2_features = gcn2_features_res
        gcn3_features = gcn3_features_res
        self.L = L_res
        self.output =output_t_res
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.layers = nn.ModuleList([GraphTransformerLayer(node_features, gcn1_features, gcn2_features, gcn3_features,
                                                           in_dim, out_dim, fout_dim, head_num,dropout,
                                                            self.layer_norm, self.batch_norm, self.residual) for _ in range(self.L - 1)])
        self.layers.append(
            GraphTransformerLayer(node_features, gcn1_features, gcn2_features, gcn3_features,
                                  in_dim, out_dim, fout_dim, head_num, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.FN = nn.Linear(fout_dim, self.output)

    def res_para(self):
        for conv in self.layers:
            conv.res_para()
        self.FN.reset_parameters()
    def forward(self,data, datap):

        for conv in self.layers:
            h = conv(data, datap)
        h = F.leaky_relu((self.FN(h)))
        return h
