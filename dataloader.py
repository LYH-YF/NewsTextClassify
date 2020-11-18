import json
import torch


def load_json_data(path):
    f = open(path, mode="r")
    data = json.load(f)
    f.close()
    return data


def build_batch(id_list, type, pad_token):
    path = "data/processed/"+type+"set_"
    text_input = []
    text_inputlen = []
    target = []
    # 根据id加载数据
    for text_id in id_list:
        text_path = path+str(text_id)+".json"
        data = load_json_data(text_path)
        text_input.append(data["text"])
        text_inputlen.append(len(data["text"]))
        target.append(int(data["label"]))
    max_len = max(text_inputlen)
    for i in range(len(id_list)):
        text_input[i] = text_input[i]+[pad_token]*(max_len-text_inputlen[i])
    return text_input, target


def load_batch_data(batchsize, id_list, type, pad_token):
    '''
        id_list: train_id_list / test_id_list / valid_id_list
        type   : 'train' or 'test'
    '''
    # 获取数据总数和batch数量
    text_num = len(id_list)
    batch_num = int(text_num/batchsize)+1
    # 加载每一个batch数据
    for batch_i in range(batch_num):
        # 第i个batch最小索引
        idx_min = batch_i*batchsize
        # 第i个batch最大索引
        if (batch_i+1)*batchsize >= text_num:
            idx_max = text_num
        else:
            idx_max = (batch_i+1)*batchsize
        if idx_min == idx_max or idx_min >= text_num:
            break
        # 获取第i个batch的id_list
        batch_id_list = id_list[idx_min:idx_max]
        # 加载第i个batch
        text_input, target = build_batch(batch_id_list, type, pad_token)

        text_input = torch.tensor(text_input)
        target = torch.tensor(target)
        yield {"input": text_input, "target": target}

class DataLoader(object):
    def __init__(self) :
        self.vocab_list=load_json_data("data/vocab_list.json")
        self.vocab_dict=load_json_data("data/vocab_dict.json")
        self.train_id_list=load_json_data("data/train_id.json")
        self.valid_id_list=load_json_data("data/valid_id.json")
        self.test_id_list=load_json_data("data/test_id.json")
    def load_batch_data(self,batch_size,type):
        if type=="train":
            id_list=self.train_id_list
            file_type="train"
        elif type=="valid":
            id_list=self.valid_id_list
            file_type="train"
        elif type=="test":
            id_list=self.test_id_list
            file_type="test"
        else:
            raise Exception("no data type named {}: excepted train,valid or test".format(type))
        # 获取数据总数和batch数量
        text_num = len(id_list)
        batch_num = int(text_num/batch_size)+1
        # 加载每一个batch数据
        for batch_i in range(batch_num):
            # 第i个batch最小索引
            idx_min = batch_i*batch_size
            # 第i个batch最大索引
            if (batch_i+1)*batch_size >= text_num:
                idx_max = text_num
            else:
                idx_max = (batch_i+1)*batch_size
            if idx_min == idx_max or idx_min >= text_num:
                break
            # 获取第i个batch的id_list
            batch_id_list = id_list[idx_min:idx_max]
            # 加载第i个batch
            text_input, target = self.build_batch(batch_id_list, file_type)
            text_input = torch.tensor(text_input)
            target = torch.tensor(target)
            yield {"input": text_input, "target": target}
    def build_batch(self,batch_id_list,file_type):
        path = "data/processed/"+file_type+"set/"
        batch_text_input = []
        batch_text_inputlen = []
        batch_target = []
        # 根据id加载数据
        for text_id in batch_id_list:
            text_path = path+str(text_id)+".json"
            data = load_json_data(text_path)
            text_input=[]
            #word to index
            for word in data["text"]:
                try:
                    text_input.append(self.vocab_dict[word])
                except:
                    text_input.append(self.vocab_dict["UNK_token"])
            text_input.append(self.vocab_dict["END_token"])
            batch_text_input.append(text_input)
            batch_text_inputlen.append(len(text_input))
            batch_target.append(int(data["label"]))
        #pad sentence
        max_len = max(batch_text_inputlen)
        for i in range(len(batch_id_list)):
            batch_text_input[i] = batch_text_input[i]+[self.vocab_dict["PAD_token"]]*(max_len-batch_text_inputlen[i])
        return batch_text_input, batch_target
if __name__ == "__main__":
    dl=DataLoader()
    dl.load_batch_data(64,"aa")
    # test_id_list=load_json_data("data/test_id.json")
    # valid_id_list=load_json_data("data/valid_id.json")
    # train_id_list=load_json_data("data/test_id.json")
    # max_index=[]
    # min_index=[]
    # x=0
    # for data in load_batch_data(64,train_id_list,"train",1):
    #     max_index.append(torch.max(data["input"]))
    #     min_index.append(torch.min(data["input"]))
    #     x+=1
    #     print("\r{}".format(x),end="")
    # for data in load_batch_data(64,valid_id_list,"train",1):
    #     max_index.append(torch.max(data["input"]))
    #     min_index.append(torch.min(data["input"]))
    #     x+=1
    #     print("\r{}".format(x),end="")
    # print()
    # print("min:",min(min_index))
    # print("max:",max(max_index))
    # for data in load_batch_data(64,test_id_list,"test",1):
    #     max_index.append(torch.max(data["input"]))
    #     min_index.append(torch.min(data["input"]))
    #     x+=1
    #     print("\r{}".format(x),end="")
    # print()
    # print("min:",min(min_index))
    # print("max:",max(max_index))
