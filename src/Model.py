import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from T_model import GTM_net
from self_attention_pooling import SAGPool
from torch_geometric.nn import GCNConv
import torch_geometric.nn as pyg_nn
class main_model(nn.Module):
    def __init__(self, args):
        super(main_model, self).__init__()
        # set_seed()
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)

        self.GT_res = GTM_net(args.gcn2_features_res, args.out_dim_res, args.fout_dim_res, args.head_num_res, args.dropout_res,
                              args.node_features_res, args.gcn1_features_res, args.gcn3_features_res, args.L_res,
                              args.output_t_res, args.layer_norm, args.batch_norm, args.residual)
        self.GT_atom = GTM_net(args.gcn2_features_atom, args.out_dim_atom, args.fout_dim_atom, args.head_num_atom, args.dropout_atom,
                              args.node_features_atom, args.gcn1_features_atom, args.gcn3_features_atom, args.L_atom,
                              args.output_t_atom, args.layer_norm, args.batch_norm, args.residual)

        self.MLP1 = nn.Linear(args.mlpin_dim, args.mlp1_dim)
        self.MLP2 = nn.Linear(args.mlp1_dim, args.mlp2_dim)
        self.MLP3 = nn.Linear(args.mlp2_dim, args.mlp3_dim)
        # self.MLP4 = nn.Linear(args.mlp3_dim, args.mlp4_dim)
        self.pool = pyg_nn.global_mean_pool
        self.max_pool = pyg_nn.global_max_pool
        self.st_pool = SAGPool(args.nhid, ratio=args.pooling_ratio)
        self.st_pool_atom = SAGPool(args.nhid1, ratio=args.pooling_ratio)

    def res_para(self):
        self.GT_res.res_para()
        self.MLP1.reset_parameters()
        self.MLP2.reset_parameters()
        self.MLP3.reset_parameters()
        # self.MLP2.reset_parameters()
    def forward(self,data_res, data_atom, data_atomIF, res_pos, atom_pos):
        edge_index, batch = data_res.edge_index, data_res.batch
        atom_fae = self.GT_atom(data_atom, atom_pos)
        atom2res_feat = average_matrix_by_list(atom_fae, data_atomIF.x)
        res_fea = self.GT_res(data_res, res_pos)
        atom2res_feat = atom2res_feat.to(res_fea.device)
        protein_fea = torch.cat((res_fea, atom2res_feat), dim=1)
        fea1 = self.st_pool(protein_fea, edge_index, None,  batch)
        feature = F.leaky_relu(self.MLP1(fea1))
        feature = self.drop1(feature)
        feature = F.leaky_relu(self.MLP2(feature))
        feature = self.drop2(feature)
        feature = F.leaky_relu(self.MLP3(feature))
        output = F.softmax(feature, dim=1)
        return output, feature


