import json
import torch
def load_json_data(path):
    f=open(path,mode="r")
    data=json.load(f)
    f.close()
    return data    
def build_batch(id_list,type,pad_token):
    path="data/processed_"+type+"set_"
    text_input=[]
    text_inputlen=[]
    target=[]
    for text_id in id_list:
        text_path=path+str(text_id)+".json"
        data=load_json_data(text_path)
        text_input.append(data["text"])
        text_inputlen.append(len(data["text"]))
        target.append(data["label"])
    max_len=max(text_inputlen)
    for i in range(len(id_list)):
        text_input[i]=text_input[i]+[pad_token]*(max_len-text_inputlen[i])
    return text_input,target
def load_batch_data(batchsize,id_list,type,pad_token):
    text_num=len(id_list)
    batch_num=int(text_num/batchsize)+1
    for batch_i in range(batch_num):
        idx_min=batch_i*batchsize
        if (batch_i+1)*batchsize>=text_num:
            idx_max=text_num-1
        else:
            idx_max=(batch_i+1)*batchsize
        if idx_min==idx_max or idx_min>=text_num:
            break
        batch_id_list=id_list[idx_min:idx_max]
        text_input,target=build_batch(batch_id_list,type,pad_token)
        text_input=torch.tensor(text_input)
        target=torch.tensor(target)
        yield {"input":text_input,"target":target}

if __name__ == "__main__":
    pass