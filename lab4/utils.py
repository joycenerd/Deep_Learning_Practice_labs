import numpy as np
import torch


class OneHotEncoder():
    def __init__(self):
        self.sos=0
        self.eos=1

    def tokenize(self,word):
        chars=['SOS']+list(word)+['EOS']
        token=[]
        for ch in chars:
            if ch=='SOS':
                token.append(self.sos)
            elif ch=='EOS':
                token.append(self.eos)
            else:
                token.append(ord(ch)-ord('a')+2)
        token=torch.from_numpy(np.asarray(token))
        return token


def tf_sched(cur_epoch,epochs,final_tf_ratio):
    thres=int(0.2*epochs)
    if cur_epoch<thres:
        tf_ratio=1
    else:
        tf_ratio=final_tf_ratio+(cur_epoch-thres)*(1-final_tf_ratio)/(epochs-thres)
    if tf_ratio>1:
        tf_ratio=1
    elif tf_ratio<final_tf_ratio:
        tf_ratio=final_tf_ratio
    return tf_ratio


def klw_sched(anneal_method,cur_epoch,epochs,final_klw,anneal_cyc):
    if anneal_method=="monotonic":
        thres=0.2*epochs
        if cur_epoch<=thres:
            kl_w=0
        else:
            kl_w=final_klw*(cur_epoch-thres)/(epochs-thres)
    elif anneal_method=="cyclic":
        T=int(epochs/anneal_cyc)
        thres=int(T*0.2)
        cur_epoch%=T

        if cur_epoch<=thres:
            kl_w=0
        else:
            kl_w=final_klw*(cur_epoch-thres)/(T-thres)
    
    return kl_w



