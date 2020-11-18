from model import *
import os
def build_model(args,ntokens,device):
    emb_size = args.embsize    # 嵌入层维度
    nhid = args.hiddensize     # nn.TransformerEncoder 中前馈神经网络的维度
    nhead = args.nhead         # 多头注意力机制中“头”的数目
    max_len = args.maxlen      # 最大句子长度
    nclass = args.nclass       # 文本类别数量
    dropout = args.dropout     # dropout
    if args.model=="transformer":
        nlayers = args.nlayers #编码器中 nn.TransformerEncoderLayer 层数
        model = TransformerModel(ntokens, emb_size, nhead,
                             nhid, nlayers, nclass, max_len, dropout).to(device)
    elif args.model=="textcnn":
        kernel_size = [4, 3, 2]
        out_channel = args.outchannel
        model = TextCNN(ntokens, emb_size, kernel_size,
                            out_channel, nclass, dropout, max_len).to(device)
    else:
        raise Exception("no model named {}".format(args.model))
    return model
def load_model_parameter(model,args):
    checkpath=None
    listdirs=os.listdir("models/")
    for file in listdirs:
        if args.model=="transformer":
            if "trans" in file:
                checkpath="models/"+file
        elif args.model=="textcnn":
            if "textcnn" in file:
                checkpath="models/"+file
    if checkpath==None:
        print("no trained {} model in models/".format(args.model))
        start_epoch=0
        best_val_loss=0.
    else:
        checkpoint = torch.load(checkpath)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch=checkpoint['epoch']
        best_val_loss=checkpoint['loss']
    return model,start_epoch,best_val_loss