import random
import gc
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from utils import *
from Model import main_model
from sklearn.metrics import accuracy_score
import json
import matplotlib.pyplot as plt
import time
from Data_pro import Data_loader
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
        classification_report

def train():
    set_seed()
    train_res, valid_res, test_res, \
    train_atom, valid_atom, test_atom, \
    train_atomIF, valid_atomIF, test_atomIF, \
    train_pos_res, valid_pos_res, test_pos_res, \
    train_pos_atom, valid_pos_atom, test_pos_atom = Data_loader()
    model = main_model(args).cuda()
    best_val_loss = float('inf')
    optimizer = optim.AdamW(model.parameters(), args.lr)
    for epoch in range(args.epochs):
        model.train()
        loss_all = 0
        combined_loader = zip(train_res, train_atom, train_atomIF, train_pos_res, train_pos_atom )
        for i, (data_res, data_atom, data_atomIF, res_pos, atom_pos) in enumerate(combined_loader):
            # model.train()
            optimizer.zero_grad()

            data_res.cuda()
            data_atom.cuda()
            data_atomIF.cuda()
            res_pos.cuda()
            atom_pos.cuda()
            # model(data)
            out, _ = model(data_res, data_atom, data_atomIF, res_pos, atom_pos)

            loss_fun = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fun(out, data_res.y)
            loss.backward()
            optimizer.step()
            loss_all += loss
        loss_train = loss_all/len(train_res)

        y_pred_all = []
        real_all = []

        model.eval()
        loss_all1 = 0
        combined_loader = zip(valid_res, valid_atom, valid_atomIF, valid_pos_res, valid_pos_atom)
        with torch.no_grad():
            for i, (data_res, data_atom, data_atomIF, res_pos, atom_pos) in enumerate(combined_loader):
                data_res.cuda()
                data_atom.cuda()
                data_atomIF.cuda()
                res_pos.cuda()
                atom_pos.cuda()
                out, _ = model(data_res, data_atom, data_atomIF, res_pos, atom_pos)
                y_pred_all.append(out)
                real_all.append(data_res.y)
                loss_fun1 = torch.nn.CrossEntropyLoss(reduction='mean')
                loss1 = loss_fun1(out, data_res.y)
                loss_all1 += loss1
        loss_valid = loss_all1/len(valid_res)
        y_pred_all = torch.cat(y_pred_all, dim=0).cpu()
        real_all = torch.cat(real_all, dim=0).cpu().reshape(-1).numpy()
        predicted_classes = torch.argmax(y_pred_all, dim=1).numpy()
        print("real_all:{}---predicted_classes:{}".format(real_all, predicted_classes))
        accuracy = accuracy_score(real_all, predicted_classes)
        precision2 = precision_score(real_all, predicted_classes, average='weighted', zero_division=1.0)
        recall2 = recall_score(real_all, predicted_classes, average='weighted')
        f1_2 = f1_score(real_all, predicted_classes, average='weighted')
        print('epoch  {}: train_Loss: {},  valid_loss: {}'.format(epoch + 1, loss_train, loss_valid))
        if loss_valid < best_val_loss:
            best_val_loss = loss_valid
            torch.save(model.state_dict(), '../model_para/modelpara_meanpool.pt')
        print("valid_performance(weighted):accuracy:{}  precision:{}  recall:{}   f1:{} ".format(accuracy, precision2, recall2, f1_2))
    model = main_model(args).cuda()
    patch = '../model_para/modelpara_07.pt'
    cpkt = torch.load(patch)
    model.load_state_dict(cpkt)
    y_pred_all = []
    real_all = []
    fea_all = []
    model.eval()
    combined_loader = zip(test_res, test_atom, test_atomIF, test_pos_res, test_pos_atom)
    with torch.no_grad():
        for i, (data_res, data_atom, data_atomIF, res_pos, atom_pos) in enumerate(combined_loader):
            data_res.cuda()
            data_atom.cuda()
            data_atomIF.cuda()
            res_pos.cuda()
            atom_pos.cuda()
            out, fea = model(data_res, data_atom, data_atomIF, res_pos, atom_pos)
            y_pred_all.append(out)
            real_all.append(data_res.y)
            fea_all.append(fea)
        y_pred_all = torch.cat(y_pred_all, dim=0).cpu()
        real_all = torch.cat(real_all, dim=0).cpu().reshape(-1).numpy()
        predicted_classes = torch.argmax(y_pred_all, dim=1).numpy()
        print("real_all:{}---predicted_classes:{}".format(real_all, predicted_classes))
        accuracy = accuracy_score(real_all, predicted_classes)
        precision2 = precision_score(real_all, predicted_classes, average='weighted', zero_division=1.0)
        recall2 = recall_score(real_all, predicted_classes, average='weighted', zero_division=1.0)
        f1_2 = f1_score(real_all, predicted_classes, average='weighted', zero_division=1.0)
        #metric_tmp = get_metrics(real_all, y_pred_all)
        class_report = classification_report(real_all, predicted_classes)

        print("-------------test dataset-----------------")

        print("test_performance(weighted):accuracy:{}   precision:{}  recall:{}   f1:{} ".format(accuracy, precision2, recall2, f1_2))
        print("Classification Report:")
        print(class_report)


if __name__=="__main__" :
    start_time = time.time()
    set_seed()
    train()
    end_time = time.time()
    final_time = (end_time - start_time) / 60
    print(f"train  took {final_time:.2f} minutes")




