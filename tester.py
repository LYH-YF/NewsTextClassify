import time
import math
import torch
import os
from torch import nn
from model import TransformerModel
from dataloader import load_batch_data,load_json_data

ntokens = 7551 # 词表大小
emsize = 200 # 嵌入层维度
nhid = 200 # nn.TransformerEncoder 中前馈神经网络的维度
nlayers = 2 # 编码器中 nn.TransformerEncoderLayer 层数
nhead = 2 # 多头注意力机制中“头”的数目
dropout = 0.2 # dropout
# 先实例化一个模型
model = TransformerModel(ntoken=7551, ninp=200, nhead=2, nhid=200, nlayers=2,nclass=14, dropout=0.2).to(device)
# 模型加载训练好的参数
# checkpoint = torch.load('datasets/models/best_model.pth.tar')
checkpoint = torch.load('temp/models/best_model.pth.tar')
model.load_state_dict(checkpoint['state_dict'])