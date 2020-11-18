from config import get_arg
import pandas as pd
import torch
import os
from dataloader import DataLoader
from buildmodel import *

def test_trans(args):
    dl=DataLoader()
    batch_size = args.batchsize
    ntokens = len(dl.vocab_list) # 词表大小
    max_len=args.maxlen    # 最大句子长度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 先实例化一个模型
    #model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers,nclass, max_len,dropout).to(device)
    model=build_model(args,ntokens,device)
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
    results=[]
    step=0
    print("start test:{} | device:{}".format(args.model,device))
    model.eval()
    for batch_data in dl.load_batch_data(batch_size,"test"):
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
            print("predict {:5d}/{:5d}" .format(step,len(dl.test_id_list)//batch_size+1))
    return results
def test_textcnn(args):
    dl=DataLoader()
    batch_size = args.batchsize
    ntokens = len(dl.vocab_list)  # 词表大小
    max_len = args.maxlen   #最大句子长度
    # 获取当前设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #textcnn_model=TextCNN(ntokens,emb_size,kernel_size,out_channel,nclass,dropout,max_len).to(device)
    textcnn_model=build_model(args,ntokens,device)
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
    #test_id_list=load_json_data("data/test_id.json")
    results=[]
    step=0
    print("start test:{} | device:{}".format(args.model,device))
    for batch_data in dl.load_batch_data(batch_size,"test"):
        step+=1
        if batch_data["input"].size(1)>max_len:
            inputs=batch_data["input"][:,:max_len].to(device)
        else:
            inputs=batch_data["input"].to(device)
        output = textcnn_model(inputs)
        idxs=output.argsort(dim=1,descending=True)[:,:1]
        pred=idxs.T.tolist()
        results+=pred[0]
        if step%100==0 and step>0:
            print("predict {:5d}/{:5d}" .format(step,len(dl.test_id_list)//batch_size+1))
    return results

if __name__ == "__main__":
    args=get_arg()
    result=None
    if args.model=="transformer":
        result=test_trans(args)
    elif args.model=="textcnn":
        result=test_textcnn(args)
    else:
        print("no model named {}".format(args.model))
        pass
    if result !=None:
        result=pd.DataFrame(result,columns=["label"])
        result.to_csv("submit_trans.csv",index=None)
    # result=test_for_p()
    # with open("result/r.json",mode="w+") as f:
    #     result=json.dumps(result,indent=4)
    #     f.write(result)
    # f.close()
    
