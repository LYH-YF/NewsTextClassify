import time
import math
import torch
import os
from torch import nn
from model import TransformerModel,TextCNN
from evaluator import *
from dataloader import load_batch_data,load_json_data

def train(epoch,batch_size):
    ntokens = 7551 # 词表大小
    emsize = 200 # 嵌入层维度
    nhid = 200 # nn.TransformerEncoder 中前馈神经网络的维度
    nlayers = 2 # 编码器中 nn.TransformerEncoderLayer 层数
    nhead = 2 # 多头注意力机制中“头”的数目
    max_len=20
    nclass=14
    dropout = 0.2 # dropout
    # 获取当前设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #实例化模型
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, nclass,max_len,dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    # 学习率
    lr = 2.0
    # 随机梯度下降
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # 动态调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    train_id_list=load_json_data("data/train_id.json")
    total_loss = 0.
    #--------------------------------
    best_val_loss = float("inf")
    best_model = None
    for epo in range(epoch):
        step=-1
        total_loss=0.
        model.train() # 训练模式，更新模型参数
        epoch_start_time=time.time()
        for batch_data in load_batch_data(batch_size,train_id_list,"train",pad_token=7550):
            step+=1
            start_time = time.time() # 用于记录模型的训练时长
            # 获取批次数据
            optimizer.zero_grad()
            if batch_data["input"].size(1)>max_len:
                inputs=batch_data["input"][:,:max_len].T.to(device)
            else:
                inputs=batch_data["input"].T.to(device)
            output = model(inputs)
            # 计算损失
            label=batch_data["target"]
            # label=torch.zeros(inputs.size(1),nclass)
            # row_idx=torch.arange(inputs.size(1))
            # label[row_idx,batch_data["target"]]=1
            #loss = criterion(output.view(-1, nclass), label.to(device))
            #loss=criterion(output,label.to(device).long())
            #loss = criterion(output.transpose(0,1), label.to(device).long())
            loss=criterion(output[-1],label.to(device))
            # 计算梯度
            loss.backward()
            # 梯度裁剪，防止梯度消失/爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # 优化参数
            optimizer.step()
            
            # 打印训练记录
            total_loss += loss.item()
            log_interval = 100
            if step % log_interval == 0 and step > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch [{:3d}/{:3d}] | step [{:5d}/{:5d}] | '
                    'lr {:02.2f} | {:5.2f} ms/batch | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epo+1,epoch, step, len(train_id_list) // batch_size, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
        # 验证过程
        val_loss = evaluate_transformer(model, batch_size)
        # 打印验证结果
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epo, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)
        # 调整学习率
        scheduler.step()
        # 记录最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            # 保存模型
            torch.save({'state_dict': best_model.state_dict()}, 'models/trans_best_model'+str(epo)+'.pth.tar')

def train_textcnn(epoch,batch_size):
    ntokens = 7551 # 词表大小
    emb_size = 5 # 嵌入层维度
    max_len=20
    kernel_size=[4,3,2]
    out_chanel=2
    nclass=14
    dropout = 0.2 # dropout
    # 获取当前设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    textcnn_model=TextCNN(ntokens,emb_size,kernel_size,out_chanel,nclass,dropout,max_len).to(device)
    criterion = nn.CrossEntropyLoss()
    # 学习率
    lr = 2.0
    # 随机梯度下降
    optimizer = torch.optim.SGD(textcnn_model.parameters(), lr=lr)
    # 动态调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    train_id_list=load_json_data("data/train_id.json")
    total_loss = 0.
    #--------------------------------
    best_val_loss = float("inf")
    best_model = None
    for epo in range(epoch):
        step=-1
        total_loss=0.
        textcnn_model.train() # 训练模式，更新模型参数
        epoch_start_time=time.time()
        for batch_data in load_batch_data(batch_size,train_id_list,"train",pad_token=7550):
            step+=1
            start_time = time.time() # 用于记录模型的训练时长
            # 获取批次数据
            optimizer.zero_grad()
            if batch_data["input"].size(1)>max_len:
                inputs=batch_data["input"][:,:max_len].to(device)
            else:
                inputs=batch_data["input"].to(device)
            output = textcnn_model(inputs)
            # 计算损失
            label=batch_data["target"]
            #output=output.unsqueeze(1)
            loss = criterion(output, label.to(device).long())
            # 计算梯度
            loss.backward()
            # 梯度裁剪，防止梯度消失/爆炸
            torch.nn.utils.clip_grad_norm_(textcnn_model.parameters(), 0.5)
            # 优化参数
            optimizer.step()
            
            # 打印训练记录
            total_loss += loss.item()
            log_interval = 100
            if step % log_interval == 0 and step > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch [{:3d}/{:3d}] | step [{:5d}/{:5d}] | '
                    'lr {:02.2f} | {:5.2f} ms/batch | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epo+1,epoch, step, len(train_id_list) // batch_size, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
        # 验证过程
        val_loss = evaluate_textcnn(textcnn_model, batch_size)
        # 打印验证结果
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epo, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)
        # 调整学习率
        scheduler.step()
        # 记录最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = textcnn_model
            # 保存模型
            torch.save({'state_dict': best_model.state_dict()}, 'models/textcnn_best_model'+str(epo)+'.pth.tar')


if __name__ == "__main__":
    train(50,64)
    #train_textcnn(10,64)