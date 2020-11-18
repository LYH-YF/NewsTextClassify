# build module
import torch
import math
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertModel
#from transformers import BertModel

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

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight.data.normal_(mean=0.0, std=0.05)

        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        self.bias.data.copy_(torch.from_numpy(b))

        self.query = nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0, std=0.05)

    def forward(self, batch_hidden, batch_masks):
        # batch_hidden: b x len x hidden_size (2 * hidden_size of lstm)
        # batch_masks:  b x len

        # linear
        key = torch.matmul(batch_hidden, self.weight) + self.bias  # b x len x hidden

        # compute attention
        outputs = torch.matmul(key, self.query)  # b x len

        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32))

        attn_scores = F.softmax(masked_outputs, dim=1)  # b x len

        # 对于全零向量，-1e32的结果为 1/len, -inf为nan, 额外补0
        masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)

        # sum weighted sources
        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)  # b x hidden

        return batch_outputs, attn_scores


# build word encoder
class WordBertEncoder(nn.Module):
    def __init__(self,bert_path,dropout):
        '''
        bert_path = '../emb/bert-mini/'
        dropout = 0.15
        '''
        super(WordBertEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.tokenizer = WhitespaceTokenizer()
        self.bert = BertModel.from_pretrained(bert_path)

        self.pooled = False
        logging.info('Build Bert encoder with pooled {}.'.format(self.pooled))

    def encode(self, tokens):
        tokens = self.tokenizer.tokenize(tokens)
        return tokens

    def get_bert_parameters(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return optimizer_parameters

    def forward(self, input_ids, token_type_ids):
        # input_ids: sen_num x bert_len
        # token_type_ids: sen_num  x bert_len

        # sen_num x bert_len x 256, sen_num x 256
        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids)

        if self.pooled:
            reps = pooled_output
        else:
            reps = sequence_output[:, 0, :]  # sen_num x 256

        if self.training:
            reps = self.dropout(reps)

        return reps


class WhitespaceTokenizer():
    """WhitespaceTokenizer with vocab."""

    def __init__(self,bert_path):
        vocab_file = bert_path + 'vocab.txt'
        self._token2id = self.load_vocab(vocab_file)
        self._id2token = {v: k for k, v in self._token2id.items()}
        self.max_len = 256
        self.unk = 1

        logging.info("Build Bert vocab with size %d." % (self.vocab_size))

    def load_vocab(self, vocab_file):
        f = open(vocab_file, 'r')
        lines = f.readlines()
        lines = list(map(lambda x: x.strip(), lines))
        vocab = dict(zip(lines, range(len(lines))))
        return vocab

    def tokenize(self, tokens):
        assert len(tokens) <= self.max_len - 2
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        output_tokens = self.token2id(tokens)
        return output_tokens

    def token2id(self, xs):
        if isinstance(xs, list):
            return [self._token2id.get(x, self.unk) for x in xs]
        return self._token2id.get(xs, self.unk)

    @property
    def vocab_size(self):
        return len(self._id2token)


# build sent encoder
class SentEncoder(nn.Module):
    def __init__(self, sent_rep_size,sent_hidden_size,sent_num_layers,dropout):
        '''
        sent_hidden_size = 256
        sent_num_layers = 2
        '''
        super(SentEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.sent_lstm = nn.LSTM(
            input_size=sent_rep_size,
            hidden_size=sent_hidden_size,
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sent_reps, sent_masks):
        # sent_reps:  b x doc_len x sent_rep_size
        # sent_masks: b x doc_len

        sent_hiddens, _ = self.sent_lstm(sent_reps)  # b x doc_len x hidden*2
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)

        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)

        return sent_hiddens