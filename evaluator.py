import torch
from torch import nn
from dataloader import load_json_data,load_batch_data
def evaluate_transformer(eval_model, batch_size):
    valid_id_list=load_json_data("data/valid_id.json")
    eval_model.eval() # 评估模式，不更新模型参数，仅评估模型当前的表现
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.
    max_len=1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for batch_data in load_batch_data(batch_size,valid_id_list,"train",7550):
            if batch_data["input"].size(1)>max_len:
                inputs=batch_data["input"][:,:max_len].T.to(device)
            else:
                inputs=batch_data["input"].T.to(device)
            output = eval_model(inputs)
            # 计算损失
            label=batch_data["target"]
            loss = criterion(output[-1], label.to(device).long())
            total_loss += batch_size * loss.item()
    return total_loss / (len(valid_id_list) - 1)
def evaluate_textcnn(eval_model,batch_size):
    valid_id_list=load_json_data("data/valid_id.json")
    eval_model.eval() # 评估模式，不更新模型参数，仅评估模型当前的表现
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.
    max_len=1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for batch_data in load_batch_data(batch_size,valid_id_list,"train",7550):
            if batch_data["input"].size(1)>max_len:
                inputs=batch_data["input"][:,:max_len].to(device)
            else:
                inputs=batch_data["input"].to(device)
            output = eval_model(inputs)
            # 计算损失
            label=batch_data["target"]
            loss = criterion(output, label.to(device).long())
            total_loss += batch_size * loss.item()
    return total_loss / (len(valid_id_list) - 1)