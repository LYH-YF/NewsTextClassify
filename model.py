from pandas.io import json
import torch
from torch import nn
import math
from torch.nn import TransformerEncoder,TransformerEncoderLayer
import random
class PositionalEncoding(nn.Module):
    '''
    给原始序列添加位置编码
    '''
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 首先初始化为0
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # sine 和 cosine 来生成位置信息
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 词经过嵌入层后，再加上位置信息
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers,nclass,max_len, dropout=0.5):
        super(TransformerModel, self).__init__()
        '''
        ntoken: 词表大小，用于构建嵌入层
        ninp: 模型维度
        nhead: 多头注意力机制中 head 数目
        nhid: 前馈神经网络的维度
        nlayers: TransformerEncoderLayer叠加层数
        '''
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout,max_len) # 位置编码
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout) # EncoderLayer
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) # Encoder
        self.encoder = nn.Embedding(ntoken, ninp) # 嵌入层
        self.ninp = ninp # 模型维度
        
        # decoder 用于将隐藏层的表示转化成词表中 token 的概率分布
        self.decoder = nn.Linear(ninp, nclass)

    def forward(self, src):
        # 生成 mask ，保证模型只能看到当前位置之前的信息
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = subsequent_mask(len(src)).to(device)
            #mask=subsequent_mask(src.size(1)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src) # 位置编码
        output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=None)
        output = self.decoder(output)
        return output
if __name__ == "__main__":
    id_list=range(200000)
    valid_id=random.sample(id_list,k=10000)
    for i in valid_id:
        id_list.remove(i)
    print(len(id_list))
    import json
    with open("data/train_id.json",mode="a+") as f:
        id_list=json.dumps(id_list,indent=4)
        f.write(id_list)
    f.close()
    with open("data/valid_id.sjon") as f:
        valid_id=json.dumps(valid_id)
        f.write(valid_id)
    f.close()
    # 获取当前设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ntokens = 7550 # 词表大小
    emsize = 200 # 嵌入层维度
    nhid = 200 # nn.TransformerEncoder 中前馈神经网络的维度
    nlayers = 2 # 编码器中 nn.TransformerEncoderLayer 层数
    nhead = 2 # 多头注意力机制中“头”的数目
    dropout = 0.2 # dropout
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    print(model)
