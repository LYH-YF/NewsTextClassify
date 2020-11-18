import torch
from torch import nn
from dataloader import DataLoader
def evaluate_transformer(eval_model, args):
    #valid_id_list=load_json_data("data/valid_id.json")
    dl=DataLoader()
    eval_model.eval() # 评估模式，不更新模型参数，仅评估模型当前的表现
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.
    max_len=args.maxlen
    batch_size=args.batchsize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for batch_data in dl.load_batch_data(batch_size,"valid"):
            if batch_data["input"].size(1)>max_len:
                inputs=batch_data["input"][:,:max_len].T.to(device)
            else:
                inputs=batch_data["input"].T.to(device)
            output = eval_model(inputs)
            # 计算损失
            label=batch_data["target"]
            loss = criterion(output[-1], label.to(device).long())
            total_loss += batch_size * loss.item()
    return total_loss / len(dl.valid_id_list) 
def evaluate_textcnn(eval_model,args):
    #valid_id_list=load_json_data("data/valid_id.json")
    dl=DataLoader()
    eval_model.eval() # 评估模式，不更新模型参数，仅评估模型当前的表现
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.
    max_len=args.maxlen
    batch_size=args.batchsize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for batch_data in dl.load_batch_data(batch_size,"valid"):
            if batch_data["input"].size(1)>max_len:
                inputs=batch_data["input"][:,:max_len].to(device)
            else:
                inputs=batch_data["input"].to(device)
            output = eval_model(inputs)
            # 计算损失
            label=batch_data["target"]
            loss = criterion(output, label.to(device).long())
            total_loss += batch_size * loss.item()
    return total_loss / len(dl.valid_id_list) 