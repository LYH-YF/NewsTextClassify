import time
import math
import pandas as pd
import torch
import os
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
    max_len=20
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
    for batch_data in load_batch_data(batch_size,test_id_list,"test",7551):
        step+=1
        if batch_data["input"].size(1)>max_len:
            inputs=batch_data["input"][:,:max_len].T.to(device)
        else:
            inputs=batch_data["input"].T.to(device)
        inputs[inputs>7550]=7550
        output=model(inputs)
        idxs=output[-1].argsort(dim=1,descending=True)[:,:1]
        pred=idxs.T.tolist()
        results+=pred[0]
        if step%100==0 and step>0:
            print("predict {:5d}/{:5d}" ,format(step,len(test_id_list)//batch_size+1))
    return results
if __name__ == "__main__":
    result=test(64)
    result=pd.Series(result)
    result.to_csv("result/submit.csv")
    #x=torch.tensor([1,2,3,4,5,6,7,8,9])
    #x[x>5]=1
    #print(x>5)
