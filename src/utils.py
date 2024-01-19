import argparse
import pandas as pd
import networkx as nx
import numpy as np
import random
import numpy as np
import torch
from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader
import os
import csv
import time
parser = argparse.ArgumentParser()


parser.add_argument('--epochs', default=50, type=int, help='The training epochs')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--pooling_ratio', default=0.7, type=float, help='pooling_ratio')
parser.add_argument('--nhid', default=80, type=int, help='channel')
parser.add_argument('--nhid1', default=16, type=int, help='channel')
parser.add_argument('--node_features_res', default=1300, type=int, help='initial feature dimension in GCN.')
parser.add_argument('--gcn1_features_res', default=512, type=int, help='encode1 sequences features.')
parser.add_argument('--gcn2_features_res', default=256, type=int, help='encode2 sequences features.')
parser.add_argument('--gcn3_features_res', default=128, type=int, help='decode1 sequences features.')
parser.add_argument('--residual', type=bool, default=True, help='Residual.')
parser.add_argument('--layer_norm', type=bool, default=True, help='Layer_norm.')
parser.add_argument('--batch_norm', type=bool, default=False, help='Batch_norm.')
parser.add_argument('--L_res', type=int, default=3, help='TransformerLayer.')
parser.add_argument('--head_num_res', default=8, type=int, help='head number')
parser.add_argument('--out_dim_res', type=int, default=256, help='out_feature.')
parser.add_argument('--fout_dim_res', type=int, default=128, help='f-out_feature.')
parser.add_argument('--output_t_res', type=int, default=64, help='finally_out_feature.')
parser.add_argument('--dropout_res', type=float, default=0.2, help='dropout.')

parser.add_argument('--node_features_atom', default=70, type=int, help='initial feature dimension in GCN.')
parser.add_argument('--gcn1_features_atom', default=64, type=int, help='encode1 sequences features.')
parser.add_argument('--gcn2_features_atom', default=32, type=int, help='encode2 sequences features.')
parser.add_argument('--gcn3_features_atom', default=32, type=int, help='decode1 sequences features.')
parser.add_argument('--L_atom', type=int, default=3, help='TransformerLayer.')
parser.add_argument('--head_num_atom', default=2, type=int, help='head number')
parser.add_argument('--out_dim_atom', type=int, default=32, help='out_feature.')
parser.add_argument('--fout_dim_atom', type=int, default=16, help='f-out_feature.')
parser.add_argument('--output_t_atom', type=int, default=16, help='finally_out_feature.')
parser.add_argument('--dropout_atom', type=float, default=0.2, help='dropout.')

parser.add_argument('--mlpin_dim', type=float, default=80, help='mlpini_dim.')
parser.add_argument('--mlp1_dim', type=float, default=64, help='mlpmid_dim.')
parser.add_argument('--mlp2_dim', type=float, default=32, help='mlpini_dim.')
parser.add_argument('--mlp3_dim', type=float, default=7, help='mlpini_dim.')
# parser.add_argument('--mlp4_dim', type=float, default=1, help='mlpdin_dim.')
args = parser.parse_args()

def set_seed():
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed((args.seed))
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'



def get_data(path):
    matrix_list = []

    with open(path, 'r') as file:
        lines = file.readlines()
        num_rows = None
        matrix = []
        for line in lines:
            line = line.strip()
            if line:
                if num_rows is None:
                    num_rows = int(line)
                else:
                    row = list(map(float, line.split(',')))
                    matrix.append(row)
                    if len(matrix) == num_rows:
                        matrix_list.append(np.array(matrix))
                        matrix = []
                        num_rows = None
    return matrix_list
def get_atomInF():
    matrix_list = []
    csv_filename = "../Dataset/Atom_list_InF/atomInF_list.csv"
    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            # 将字符串转换回整数
            row = [int(cell) for cell in row]
            matrix_list.append(row)
    return matrix_list
def res_feature():
    path = "../Dataset/afterExtracFeature_12class/afterExtrac_12class_fea.csv"
    list = get_data(path)
    merged_list = list
    return merged_list
def atom_feature():
    path = "../Dataset/afterExtracFeature_12class/atom_fea.csv"
    list = get_data(path)
    merged_list = list
    return merged_list
def res_graph():
    path = "../Dataset/Graph_12class/graph_12class_res.csv"
    matrix_list = []
    with open(path, 'r') as file:
        lines = file.readlines()
        num_rows = None
        matrix = []
        for line in lines:
            line = line.strip()
            if line:
                if num_rows is None:
                    num_rows = int(line)
                else:
                    row = list(map(float, line.split(',')))
                    matrix.append(row)
                    if len(matrix) == num_rows:
                        matrix_list.append(np.array(matrix))
                        matrix = []
                        num_rows = None
    return matrix_list
def atom_graph():
    path = "../Dataset/Graph_12class/graph_12class_atom.csv"
    matrix_list = []
    with open(path, 'r') as file:
        lines = file.readlines()
        num_rows = None
        matrix = []
        for line in lines:
            line = line.strip()
            if line:
                if num_rows is None:
                    num_rows = int(line)
                else:
                    row = list(map(float, line.split(',')))
                    matrix.append(row)
                    if len(matrix) == num_rows:
                        matrix_list.append(np.array(matrix))
                        matrix = []
                        num_rows = None
    for i in range(len(matrix_list)):
        matrix_list[i] = matrix_list[i] - 1
    return matrix_list
def res_position():
    path = "../Dataset/Position/res_position.csv"
    list = get_data(path)
    merged_list = list
    return merged_list
def atom_position():
    path = "../Dataset/Position/atom_position.csv"
    list = get_data(path)
    merged_list = list
    return merged_list
def data_chuli():
    start_time = time.time()  # 记录开始时间

    graph_res = res_graph()
    graph_res = graph_res[:470] + graph_res[505:1034] + graph_res[1035:1036] + graph_res[1037:1045] + graph_res[1047:1053] + graph_res[1054:1057] + graph_res[1058:1060] + graph_res[1061:1112] + graph_res[1113:1114] + graph_res[1115:2633] + graph_res[2667:2995] + graph_res[2996:3369] + graph_res[3370:3531]
    graph_atom = atom_graph()
    graph_atom = graph_atom[:470] + graph_atom[505:1034] + graph_atom[1035:1036] + graph_atom[1037:1045] + graph_atom[1047:1053] + graph_atom[1054:1057] + graph_atom[1058:1060] + graph_atom[1061:1112] + graph_atom[1113:1114] + graph_atom[1115:2633] + graph_atom[2667:2995] + graph_atom[2996:3369] + graph_atom[3370:3531]
    res_fea = res_feature()
    res_fea = res_fea[:470] + res_fea[505:1034] + res_fea[1035:1036] + res_fea[1037:1045] + res_fea[1047:1053] + res_fea[1054:1057] + res_fea[1058:1060] + res_fea[1061:1112] + res_fea[1113:1114] + res_fea[1115:2633] + res_fea[2667:2995] + res_fea[2996:3369] + res_fea[3370:3531]
    atom_fea = atom_feature()
    atom_fea = atom_fea[:470] + atom_fea[505:1034] + atom_fea[1035:1036] + atom_fea[1037:1045] + atom_fea[1047:1053] + atom_fea[1054:1057] + atom_fea[1058:1060] + atom_fea[1061:1112] + atom_fea[1113:1114] + atom_fea[1115:2633] + atom_fea[2667:2995] + atom_fea[2996:3369] + atom_fea[3370:3531]
    res_pos = res_position()
    res_pos = res_pos[:470] + res_pos[505:1034] + res_pos[1035:1036] + res_pos[1037:1045] + res_pos[1047:1053] + res_pos[1054:1057] + res_pos[1058:1060] + res_pos[1061:1112] + res_pos[1113:1114] + res_pos[1115:2633] + res_pos[2667:2995] + res_pos[2996:3369] + res_pos[3370:3531]
    atom_pos = atom_position()
    atom_pos = atom_pos[:470] + atom_pos[505:1034] + atom_pos[1035:1036] + atom_pos[1037:1045] + atom_pos[1047:1053] + atom_pos[1054:1057] + atom_pos[1058:1060] + atom_pos[1061:1112] + atom_pos[1113:1114] + atom_pos[1115:2633] + atom_pos[2667:2995] + atom_pos[2996:3369] + atom_pos[3370:3531]
    atom_list_InF = get_atomInF()
    # atom_list_InF = atom_list_InF[:3622]
    atom_list_InF = atom_list_InF[:470] + atom_list_InF[505:1034] + atom_list_InF[1035:1036] + atom_list_InF[1037:1045] + atom_list_InF[1047:1053] + atom_list_InF[1054:1057] + atom_list_InF[1058:1060] + atom_list_InF[1061:1112] + atom_list_InF[1113:1114] + atom_list_InF[1115:2633] + atom_list_InF[2667:2995] + atom_list_InF[2996:3369] + atom_list_InF[3370:3531]
    print("ATOMLEN:{}   RES:{}".format(len(graph_atom), len(graph_res)))

    graph_labels = [0] * 470  + [1] * 1450 + [2] * 492 + [3] * 177  + [4] * 571 + [5] * 79 + [6] * 212


    data_res = [Data(x=torch.tensor(res_fea[i], dtype=torch.float),
                     edge_index=torch.tensor(np.column_stack(np.where(graph_res[i])), dtype=torch.long).t().contiguous(),
                      y=torch.tensor(graph_labels[i], dtype=torch.long))
                 for i in range(len(graph_res))]
    data_atom = [Data(x=torch.tensor(atom_fea[i], dtype=torch.float),
                      edge_index=torch.tensor((graph_atom[i]), dtype=torch.long))
                 for i in range(len(graph_res))]
    atomInF_list = [Data(x=torch.tensor(atom_list_InF[i], dtype=torch.int).reshape((len(atom_list_InF[i]), 1)))
                   for i in range(len(atom_list_InF))]

    pos_res = [Data(x=torch.tensor(res_pos[i], dtype=torch.float))
                  for i in range(len(res_pos))]

    pos_atom = [Data(x=torch.tensor(atom_pos[i], dtype=torch.float))
               for i in range(len(atom_pos))]
    set_seed()
    random.shuffle(data_res)
    set_seed()
    random.shuffle(data_atom)
    set_seed()
    random.shuffle(atomInF_list)
    set_seed()
    random.shuffle(pos_res)
    set_seed()
    random.shuffle(pos_atom)

    end_time = time.time()
    elapsed_time =(end_time - start_time)/60
    print(f"Data creation took {elapsed_time:.2f} minutes")
    return data_res, data_atom, atomInF_list, pos_res, pos_atom

def average_matrix_by_list(matrix, flattened_list):
    lst = flattened_list
    result_matrix = []
    start = 0
    for end in range(1, len(lst)):
        if lst[end] != lst[start]:
            sublist = matrix[start:end]
            average_row = [sum(col) / len(col) for col in zip(*sublist)]
            result_matrix.append(average_row)
            start = end

    sublist = matrix[start:]
    average_row = [sum(col) / len(col) for col in zip(*sublist)]
    result_matrix.append(average_row)
    result_matrix = torch.tensor(result_matrix)
    return result_matrix
