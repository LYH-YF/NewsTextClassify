from config import get_arg
import pandas as pd
import torch
import os
from model import TransformerModel,TextCNN
from dataloader import DataLoader
from sklearn.metrics import f1_score
def get_pred_trans(args,type):
    batch_size = args.batchsize
    ntokens = args.ntokens # 词表大小
    emsize = args.embsize  # 嵌入层维度
    nhid = args.hiddensize # nn.TransformerEncoder 中前馈神经网络的维度
    nlayers = args.nlayers # 编码器中 nn.TransformerEncoderLayer 层数
    nhead = args.nhead     # 多头注意力机制中“头”的数目
    max_len=args.maxlen    # 最大句子长度
    nclass=args.nclass     #文本类别数量
    dropout = args.dropout # dropout
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 先实例化一个模型
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers,nclass, max_len,dropout).to(device)
    # 模型加载训练好的参数
    checkpath=None
    listdirs=os.listdir("models/")
    for file in listdirs:
        if "trans" in file:
            checkpath="models/"+file
    if checkpath==None:
        print("no trained transformer model in models/")
        return None
    checkpoint = torch.load(checkpath)
    model.load_state_dict(checkpoint['state_dict'])
    dl=DataLoader()
    results=[]
    label=[]
    step=0
    print("start test:{} | device:{}".format(args.model,device))
    model.eval()
    for batch_data in dl.load_batch_data(batch_size,type):
        step+=1
        if batch_data["input"].size(1)>max_len:
            inputs=batch_data["input"][:,:max_len].T.to(device)
        else:
            inputs=batch_data["input"].T.to(device)
        output=model(inputs)
        idxs=output[-1].argsort(dim=1,descending=True)[:,:1]
        pred=idxs.T.tolist()
        results+=pred[0]
        label+=batch_data["target"].tolist()
        if step%100==0 and step>0:
            if type=="train":
                id_list=dl.train_id_list
            else:
                id_list=dl.valid_id_list
            print("predict {:5d}/{:5d}" .format(step,len(id_list)//batch_size+1))
    return results,label
def get_pred_textcnn(args,type):
    batch_size = args.batchsize
    ntokens = args.ntokens  # 词表大小
    emb_size = args.embsize # 嵌入层维度
    max_len = args.maxlen   #最大句子长度
    kernel_size = [4,3,2]
    out_channel = args.outchannel
    nclass = args.nclass
    dropout = args.dropout  # dropout
    # 获取当前设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    textcnn_model=TextCNN(ntokens,emb_size,kernel_size,out_channel,nclass,dropout,max_len).to(device)
    # 模型加载训练好的参数
    checkpath=None
    listdirs=os.listdir("models/")
    for file in listdirs:
        if "textcnn" in file:
            checkpath="models/"+file
    if checkpath==None:
        print("no trained transformer model in models/")
        return None
    checkpoint = torch.load(checkpath)
    textcnn_model.load_state_dict(checkpoint['state_dict'])
    dl=DataLoader()
    results=[]
    label=[]
    step=0
    print("start test:{} | device:{}".format(args.model,device))
    for batch_data in dl.load_batch_data(batch_size,type):
        step+=1
        if batch_data["input"].size(1)>max_len:
            inputs=batch_data["input"][:,:max_len].to(device)
        else:
            inputs=batch_data["input"].to(device)
        output = textcnn_model(inputs)
        idxs=output.argsort(dim=1,descending=True)[:,:1]
        pred=idxs.T.tolist()
        results+=pred[0]
        label+=batch_data["target"].tolist()
        if step%100==0 and step>0:
            if type=="train":
                id_list=dl.train_id_list
            else:
                id_list=dl.valid_id_list
            print("predict {:5d}/{:5d}" .format(step,len(id_list)//batch_size+1))
    return results,label
if __name__ == "__main__":
    args=get_arg()
    if args.model=="transformer":
        print("train set:")
        t_predict,t_label=get_pred_trans(args,"train")
        print("valid set:")
        v_predict,v_label=get_pred_trans(args,"valid")
    elif args.model=="textcnn":
        print("train set:")
        t_predict,t_label=get_pred_textcnn(args,"train")
        print("valid set:")
        v_predict,v_label=get_pred_textcnn(args,"valid")
    else:
        t_predict=None
        t_label=None
        v_predict=None
        v_label=None
        pass
    if t_predict !=None:
        print("train set scores:{}".format(f1_score(t_label,t_predict,average='macro')))
    if v_predict !=None:
        print("valid set scores:{}".format(f1_score(v_label,v_predict,average='macro')))