import time
import math
import torch
from config import get_arg
from torch import nn
from model import TransformerModel, TextCNN
from evaluator import *
from dataloader import DataLoader


def train_trans(args):
    epoch = args.epoch
    batch_size = args.batchsize
    ntokens = args.ntokens  # 词表大小
    emsize = args.embsize  # 嵌入层维度
    nhid = args.hiddensize  # nn.TransformerEncoder 中前馈神经网络的维度
    nlayers = args.nlayers  # 编码器中 nn.TransformerEncoderLayer 层数
    nhead = args.nhead     # 多头注意力机制中“头”的数目
    max_len = args.maxlen    # 最大句子长度
    nclass = args.nclass  # 文本类别数量
    dropout = args.dropout  # dropout
    # 获取当前设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模型
    model = TransformerModel(ntokens, emsize, nhead,
                             nhid, nlayers, nclass, max_len, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    # 学习率
    lr = 2.0
    # 随机梯度下降
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # 动态调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    #train_id_list = load_json_data("data/train_id.json")
    dl=DataLoader()
    total_loss = 0.
    # --------------------------------
    best_val_loss = float("inf")
    best_model = None
    print("start train:{} | device:{}".format(args.model, device))
    for epo in range(epoch):
        step = -1
        total_loss = 0.
        model.train()  # 训练模式，更新模型参数
        epoch_start_time = time.time()
        for batch_data in dl.load_batch_data(batch_size, "train"):
            step += 1
            start_time = time.time()  # 用于记录模型的训练时长
            # 获取批次数据
            optimizer.zero_grad()
            if batch_data["input"].size(1) > max_len:
                inputs = batch_data["input"][:, :max_len].T.to(device)
            else:
                inputs = batch_data["input"].T.to(device)
            output = model(inputs)
            # 计算损失
            label = batch_data["target"]
            loss = criterion(output[-1], label.to(device))
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
                          epo +
                          1, epoch, step, len(
                              dl.train_id_list) // batch_size, scheduler.get_lr()[0],
                          elapsed * 1000 / log_interval,
                          cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
        # 验证过程
        val_loss = evaluate_transformer(model, batch_size)
        # 打印验证结果
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epo+1, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # 调整学习率
        scheduler.step()
        # 记录最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            # 保存模型
            torch.save({'state_dict': best_model.state_dict(),"epoch":epo+1,"loss":best_val_loss},
                       'models/trans_best_model.pth.tar')
            print("saved model at epoch {}".format(epo+1))


def train_textcnn(args):
    epoch = args.epoch
    batch_size = args.batchsize
    ntokens = args.ntokens  # 词表大小
    emb_size = args.embsize  # 嵌入层维度
    max_len = args.maxlen  # 最大句子长度
    kernel_size = [4, 3, 2]
    out_channel = args.outchannel
    nclass = args.nclass
    dropout = args.dropout  # dropout
    # 获取当前设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    textcnn_model = TextCNN(ntokens, emb_size, kernel_size,
                            out_channel, nclass, dropout, max_len).to(device)
    criterion = nn.CrossEntropyLoss()
    # 学习率
    lr = args.lr
    # 随机梯度下降
    #optimizer = torch.optim.SGD(textcnn_model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(textcnn_model.parameters(), lr=lr)
    #train_id_list = load_json_data("data/train_id.json")
    dl=DataLoader()
    total_loss = 0.
    # --------------------------------
    best_val_loss = float("inf")
    best_model = None
    print("start train:{} | device:{}".format(args.model, device))
    for epo in range(epoch):
        step = -1
        total_loss = 0.
        textcnn_model.train()  # 训练模式，更新模型参数
        epoch_start_time = time.time()
        for batch_data in dl.load_batch_data(batch_size, "train"):
            step += 1
            start_time = time.time()  # 用于记录模型的训练时长
            # 获取批次数据
            optimizer.zero_grad()
            if batch_data["input"].size(1) > max_len:
                inputs = batch_data["input"][:, :max_len].to(device)
            else:
                inputs = batch_data["input"].to(device)
            output = textcnn_model(inputs)
            # 计算损失
            label = batch_data["target"]
            # output=output.unsqueeze(1)
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
                          epo+1, epoch, step, len(dl.train_id_list) // batch_size,
                          optimizer.state_dict()['param_groups'][0]['lr'],
                          elapsed * 1000 / log_interval,
                          cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
        # 验证过程
        val_loss = evaluate_textcnn(textcnn_model, batch_size)
        # 打印验证结果
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epo+1, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        # 记录最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = textcnn_model
            # 保存模型
            torch.save({'state_dict': best_model.state_dict(),"epoch":epo+1,"loss":best_val_loss},
                       'models/textcnn_best_model.pth.tar')
            print("saved model at epoch {}".format(epo+1))


if __name__ == "__main__":
    args = get_arg()
    if args.model == "transformer":
        train_trans(args)
    elif args.model == "textcnn":
        train_textcnn(args)
    else:
        print("no model named {}".format(args.model))
        pass
