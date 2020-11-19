'''
Transformer,
TextCNN,
Bert,
'''
from dataloader import load_json_data
import torch
from torch import nn
import math
import logging
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sublayer import *


def subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    return mask


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, nclass, max_len,
                 dropout):
        '''
        ntoken: 词表大小，用于构建嵌入层
        ninp: 模型维度
        nhead: 多头注意力机制中 head 数目
        nhid: 前馈神经网络的维度
        nlayers: TransformerEncoderLayer叠加层数
        '''
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len)  # 位置编码
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid,
                                                 dropout)  # EncoderLayer
        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            nlayers,
        )  # Encoder
        self.encoder = nn.Embedding(ntoken, ninp)  # 嵌入层
        self.ninp = ninp  # 模型维度

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
        src = self.pos_encoder(src)  # 位置编码
        output = self.transformer_encoder(src,
                                          mask=self.src_mask,
                                          src_key_padding_mask=None)
        output = self.decoder(output)
        return output


class TextCNN(nn.Module):
    def __init__(self, n_token, emb_size, kernel_size, out_channels, n_class,
                 dropout, max_len):
        """
        docstring
        """
        super(TextCNN, self).__init__()
        self.dropout_rate = dropout
        self.num_class = n_class

        self.embedding = nn.Embedding(num_embeddings=n_token,
                                      embedding_dim=emb_size)  # 创建词向量对象
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=emb_size,
                          out_channels=out_channels,
                          kernel_size=h),  # 卷积层
                nn.ReLU(),  # 激活函数层
                nn.MaxPool1d(kernel_size=max_len - h + 1)  # 池化层
            ) for h in kernel_size
        ])  # 创建多个卷积层，包含了 图中的convolution、activation function和 maxPooling
        self.fc = nn.Linear(in_features=out_channels * len(kernel_size),
                            out_features=n_class)
        self.dropout = nn.functional.dropout

    def forward(self, inputs):
        """
        docstring
        """
        embed_x = self.embedding(inputs)  # 将句子转为词向量矩阵，大小为1*7*5，这里的1是表示只有1条句子
        embed_x = embed_x.permute(0, 2, 1)  # 将矩阵转置
        out = [conv(embed_x) for conv in self.convs
               ]  #计算每层卷积的结果，这里输出的结果已经经过池化层处理了，对应着图中的6 个univariate vectors
        out = torch.cat(out, dim=1)  # 对6 个univariate vectors进行拼接
        out = out.view(
            -1, out.size(1)
        )  #按照行优先的顺序排成一个n行1列的数据，这部分在图中没有表现出来，其实在程序运行的过程中6 个univariate vectors的大小可能是1*1*1，进行拼接后是1*1*6，我们在这里将其转换为1*6，可见每个值都没有丢失，只是矩阵的形状变了而已
        out = self.dropout(
            input=out, p=self.dropout_rate
        )  # 这里也没有在图中的表现出来，这里是随机让一部分的神经元失活，避免过拟合。它只会在train的状态下才会生效。进入train状态可查看nn.Module。train()方法
        out = self.fc(out)
        return out


# build model
class Bert(nn.Module):
    def __init__(self, sent_hidden_size, sent_num_layers, nclass, bert_path,
                 dropout):
        super(Bert, self).__init__()
        self.sent_rep_size = 256
        self.doc_rep_size = sent_hidden_size * 2
        self.all_parameters = {}
        parameters = []
        self.word_encoder = WordBertEncoder(bert_path, dropout)
        bert_parameters = self.word_encoder.get_bert_parameters()

        self.sent_encoder = SentEncoder(self.sent_rep_size, sent_hidden_size,
                                        sent_num_layers, dropout)
        self.sent_attention = Attention(self.doc_rep_size)
        parameters.extend(
            list(
                filter(lambda p: p.requires_grad,
                       self.sent_encoder.parameters())))
        parameters.extend(
            list(
                filter(lambda p: p.requires_grad,
                       self.sent_attention.parameters())))

        #self.out = nn.Linear(self.doc_rep_size, vocab.label_size, bias=True)
        self.out = nn.Linear(self.doc_rep_size, nclass, bias=True)
        parameters.extend(
            list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters
        self.all_parameters["bert_parameters"] = bert_parameters

        logging.info('Build model with bert word encoder, lstm sent encoder.')

        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))

    def forward(self, batch_inputs):
        # batch_inputs(batch_inputs1, batch_inputs2): b x doc_len x sent_len
        # batch_masks : b x doc_len x sent_len
        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[
            0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len,
                                           max_sent_len)  # sen_num x sent_len
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len,
                                           max_sent_len)  # sen_num x sent_len
        batch_masks = batch_masks.view(batch_size * max_doc_len,
                                       max_sent_len)  # sen_num x sent_len

        sent_reps = self.word_encoder(batch_inputs1,
                                      batch_inputs2)  # sen_num x sent_rep_size

        sent_reps = sent_reps.view(
            batch_size, max_doc_len,
            self.sent_rep_size)  # b x doc_len x sent_rep_size
        batch_masks = batch_masks.view(
            batch_size, max_doc_len,
            max_sent_len)  # b x doc_len x max_sent_len
        sent_masks = batch_masks.bool().any(2).float()  # b x doc_len

        sent_hiddens = self.sent_encoder(
            sent_reps, sent_masks)  # b x doc_len x doc_rep_size
        doc_reps, atten_scores = self.sent_attention(
            sent_hiddens, sent_masks)  # b x doc_rep_size

        batch_outputs = self.out(doc_reps)  # b x num_labels

        return batch_outputs


if __name__ == "__main__":
    import json
    vocab_list=load_json_data("data/vocab_list.json")
    f=open("data/vocab.txt",mode="w+")
    for vocab in vocab_list:
        f.write(vocab)
        f.write("\n")
    f.close()
    #bert_path = 'models/bert/bert-mini/'
    #model=Bert(200,2,14,bert_path,0.5)
    #print(model)
    # 获取当前设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ntokens = 7550 # 词表大小
    # emsize = 200 # 嵌入层维度
    # nhid = 200 # nn.TransformerEncoder 中前馈神经网络的维度
    # nlayers = 2 # 编码器中 nn.TransformerEncoderLayer 层数
    # nhead = 2 # 多头注意力机制中“头”的数目
    # dropout = 0.2 # dropout
    # model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    # print(model)