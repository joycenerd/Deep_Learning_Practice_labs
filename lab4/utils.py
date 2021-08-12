from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

import numpy as np
import torch


class OneHotEncoder():
    def __init__(self):
        """
        char and int transformation
        """
        self.sos=0
        self.eos=1

    def tokenize(self,word):
        """
        encode char to int

        Args: 
            word: (str) the word we want to encode

        Returns:
            char2int token list
        """
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

    def inv_tokenize(self,token):
        """
        decode int back to char

        Args:
            token: (list) integer list that represent the word

        Returns:
            word: (str) output word
        """
        word=''
        for val in token:
            if val==1:
                break
            word+=chr(val-2+ord('a'))
        return word


def tf_sched(cur_epoch,epochs,final_tf_ratio):
    """
    modified teacher forcing ratio according to epoch counts

    Args:
        cur_epoch: (int) current epoch
        epochs: (int) total epochs
        final_tf_ration: (float) smallest teacher forcing ratio
    
    Returns:
        teacher forcing ratio for current epoch
    """
    thres=int(0.2*epochs)
    if cur_epoch<thres:
        tf_ratio=1
    else:
        tf_ratio=final_tf_ratio+(epochs-cur_epoch)*(1-final_tf_ratio)/(epochs-thres)
    if tf_ratio>1:
        tf_ratio=1
    elif tf_ratio<final_tf_ratio:
        tf_ratio=final_tf_ratio
    return tf_ratio


def klw_sched(anneal_method,cur_epoch,epochs,final_klw,anneal_cyc):
    """
    modified kl weight (regularization term) according to epoch counts
    Args:
        anneal_method: (str) monotonic or cyclic
        cur_epoch: (int) current epoch
        epochs: (int) total epochs
        final_klw: (float) highest kl weight
        anneal_cyc: (int) kl annealing cycle counts
    """
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


#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)


def gen_gauss_noise(z_size):
    hid_z=torch.normal(torch.zeros(1,1,z_size),torch.ones(1,1,z_size))
    cell_z=torch.normal(torch.zeros(1,1,z_size),torch.ones(1,1,z_size))
    return hid_z,cell_z


def Gaussian_score(words,train_path):
    words_list = []
    score = 0
    yourpath = train_path
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)


def print_tense_conversion(tense_conversion_res,bleu_score,logger=None,is_print=False):
    if is_print:
        for i in range(len(tense_conversion_res)):
            res=tense_conversion_res[i]
            print(f"input: {res[0]}")
            print(f"target: {res[1]}")
            print(f"prediction: {res[2]}")
            print("")
    print(f"Average BLEU-4 score: {bleu_score:.4f}")

    if logger!=None:
        for i in range(len(tense_conversion_res)):
            res=tense_conversion_res[i]
            logger.info(f"input: {res[0]}")
            logger.info(f"target: {res[1]}")
            logger.info(f"prediction: {res[2]}")
            logger.info("")
        logger.info(f"Average BLEU-4 score: {bleu_score:.4f}")


def print_gauss_gen(words_list,gauss_score,logger=None,is_print=False):
    if is_print:
        for i in range(len(words_list)):
            print(words_list[i])
            
    print(f"Gaussian score: {gauss_score:.4f}")

    if logger!=None:
        for i in range(len(words_list)):
            logger.info(words_list[i])
        logger.info(f"Gaussian score: {gauss_score:.4f}")



