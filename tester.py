import time
import math
import pandas as pd
import torch
import os
import json
from torch import nn
from model import TransformerModel
from dataloader import load_batch_data,load_json_data

def test(batch_size):
    ntokens = 7551 # 词表大小
    emsize = 200 # 嵌入层维度
    nhid = 200 # nn.TransformerEncoder 中前馈神经网络的维度
    nlayers = 2 # 编码器中 nn.TransformerEncoderLayer 层数
    nhead = 2 # 多头注意力机制中“头”的数目
    dropout = 0.2 # dropout
    max_len=1024
    nclass=14
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 先实例化一个模型
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers,nclass, max_len,dropout).to(device)
    # 模型加载训练好的参数
    checkpoint = torch.load('models/best_model.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    test_id_list=load_json_data("data/test_id.json")
    results=[]
    step=0
    for batch_data in load_batch_data(batch_size,test_id_list,"test",7550):
        step+=1
        if batch_data["input"].size(1)>max_len:
            inputs=batch_data["input"][:,:max_len].T.to(device)
        else:
            inputs=batch_data["input"].T.to(device)
        output=model(inputs)
        idxs=output[-1].argsort(dim=1,descending=True)[:,:1]
        pred=idxs.T.tolist()
        results+=pred[0]
        if step%100==0 and step>0:
            print("predict {:5d}/{:5d}" ,format(step,len(test_id_list)//batch_size+1))
    return results
def test_for_p():
    ntokens = 7551 # 词表大小
    emsize = 200 # 嵌入层维度
    nhid = 200 # nn.TransformerEncoder 中前馈神经网络的维度
    nlayers = 2 # 编码器中 nn.TransformerEncoderLayer 层数
    nhead = 2 # 多头注意力机制中“头”的数目
    dropout = 0.2 # dropout
    max_len=1024
    nclass=14
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 先实例化一个模型
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers,nclass, max_len,dropout).to(device)
    # 模型加载训练好的参数
    checkpoint = torch.load('models/best_model3.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    valid_id_list=load_json_data("data/valid_id.json")
    step=0
    result=[]
    for batch_data in load_batch_data(64,valid_id_list,"train",7550):
        step+=1
        if batch_data["input"].size(1)>max_len:
            inputs=batch_data["input"][:,:max_len].T.to(device)
        else:
            inputs=batch_data["input"].T.to(device)
        output=model(inputs)
        torch.tensor().tolist()
        probility=output[-1].tolist()
        label=torch.zeros(inputs.size(1),nclass)
        row_idx=torch.arange(inputs.size(1))
        label[row_idx,batch_data["target"]]=1
        label=label.tolist()
        for i,prob in probility:
            result.append({"predict":prob,"label":label[i]})
        print("\rsaved:".format(step))
    return result


if __name__ == "__main__":
    #result=test(128)
    #result=pd.Series(result)
    result=test_for_p()
    with open("result/r.json",mode="w+") as f:
        result=json.dumps(result,indent=4)
        f.write(result)
    f.close()
    
