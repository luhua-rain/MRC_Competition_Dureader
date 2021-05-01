from collections import Counter
import numpy as np
from tqdm import tqdm
import sentencepiece as spm
import pandas as pd
import random
import nltk
import torch
import json
import jieba
import args
import math
import six

random.seed(args.seed)
torch.manual_seed(args.seed)
sp_model = spm.SentencePieceProcessor()
sp_model.Load('data/spiece.model')

SPIECE_UNDERLINE = '▁'
def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    # return_unicode is used only for py2

    # note(zhiliny): in some systems, sentencepiece only accepts str for py2
    if six.PY2 and isinstance(text, unicode):
        text = text.encode('utf-8')

    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                piece[:-1].replace(SPIECE_UNDERLINE, ''))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    # note(zhiliny): convert back to unicode for py2
    if six.PY2 and return_unicode:
        ret_pieces = []
        for piece in new_pieces:
            if isinstance(piece, str):
                piece = piece.decode('utf-8')
            ret_pieces.append(piece)
        new_pieces = ret_pieces
    new_pieces = [piece.replace(SPIECE_UNDERLINE, '') for piece in new_pieces if piece != SPIECE_UNDERLINE]
    return new_pieces

def get_doc_strides(sp_model, content, max_c_len, ds=256):

    c_tokens = encode_pieces(sp_model, content)
    all_strides = []
    here_start = 0
    while here_start < len(c_tokens):
        here_c = ''.join(c_tokens[here_start:here_start + max_c_len])
        all_strides.append(here_c)
        here_start += ds
    if len(c_tokens) <= max_c_len:
        return all_strides[:1]
    if all_strides[-1] in all_strides[-2]:
        all_strides = all_strides[:-1]

    return all_strides

def normalized(norm_s):
    norm_s = norm_s.replace(u"，", u",")
    # norm_s = norm_s.replace(u"\xa0", u" ").replace(u"\n", u" ").replace(u"\u3000", u" ").replace(u"\u2003", u" ").replace(u"\u2002", u" ")
    # norm_s = norm_s.replace(u"。", u".")
    # norm_s = norm_s.replace(u"！", u"!")
    # norm_s = norm_s.replace(u"？", u"?")
    # norm_s = norm_s.replace(u"；", u";")
    # norm_s = norm_s.replace(u"（", u"(").replace(u"）", u")")
    # norm_s = norm_s.replace(u"【", u"[").replace(u"】", u"]")
    # norm_s = norm_s.replace(u"“", u"\"").replace(u"”", u"\"")

    return norm_s

class RougeL(object):
    def __init__(self, beta=1):
        self.beta = beta

    def lcs(self, string, sub):           # (pre, ref)
        if len(string) < len(sub):
            sub, string = string, sub
        lengths = np.zeros((len(string) + 1, len(sub) + 1))
        for j in range(1, len(sub) + 1):
            for i in range(1, len(string) + 1):
                if string[i - 1] == sub[j - 1]:
                    lengths[i][j] = lengths[i - 1][j - 1] + 1
                else:
                    lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
        return lengths[len(string)][len(sub)]

    def get_rouge_L(self, string, sub):      # (question, ref)
        if string == '' or sub =='':
            return 0.0
        lcs = self.lcs(string, sub)
        R, P = lcs/len(sub), lcs/len(string)      # R: 召回率，P: 准确率
        rouge_L = ( (1+self.beta**2)*R*P ) / (R+self.beta**2*P + 1e-12)
        return rouge_L

class F1(object):
    def get_F1(self, string, sub):  # (question, content)
        # if len(string)==0 or len(sub)==0:
        #     return 0.0
        common = Counter(string) & Counter(sub)
        overlap = sum(common.values())
        recall, precision = overlap/len(sub), overlap/len(string)
        # return precision
        return (2*recall*precision) / (recall+precision+1e-12)

def para_recall(question, paras, max_para_num=args.max_para_num):

    score = []
    f1 = F1()
    paras = [n for n in paras if len(n)>0]

    for idx, s in enumerate(paras):  # 计算每个段落与问题的得分
        score.append((idx, f1.get_F1(question, s)))  # [ (句子位置，得分),  ... ]
        # score.append((idx, roug_L.get_rouge_L(question, s)))  # [ (句子位置，得分),  ... ]

    score = sorted(score, key=lambda x: x[1], reverse=True)  # 先以得分排序, 从大到小
    # print(score[:20])
    new_score = []
    choose_paras = []  # 所选择的句子的位置集合，最后按顺序拼接
    for p in range(min(max_para_num, len(paras))):
        choose_paras.append(score[p][0])
        new_score.append(score[p][1])

    choose_paras = sorted(choose_paras, reverse=False)  # 再以句子顺序排序

    recall_paras = []
    for idx in choose_paras:
        recall_paras.append(paras[idx])

    return recall_paras, new_score

def split_train_and_dev():

    data = pd.read_csv(args.train_data, sep = '\t')
    data = [{'id':id, 'docid':docid, 'question':question, 'answer':answer}
            for id, docid, question, answer in zip(data['id'], data['docid'], data['question'], data['answer'])]

    train_data, dev_data = [], []
    content_data = read_corpus()

    ans_len = []

    for i in tqdm(data):
        text = content_data.get(i['docid'], '')
        ans_len.append(len(i['answer']))
        i['text'] = text
        drop_sc = random.random()
        if drop_sc < 0.1:

            dev_data.append(i)
        else:
            train_data.append(i)

    print('ans_len: ', sum(ans_len)/len(ans_len))
    print(len(train_data), len(dev_data)) # 4500, 500

    with open(args.split_train_data, 'w', encoding="utf-8") as fout:
        for feature in train_data:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')

    with open(args.split_dev_data, 'w', encoding="utf-8") as fout:
        for feature in dev_data:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')

def read_corpus():
    # {'docid1':text, 'docidN':text}

    data = {'docid':[], 'text':[]}
    with open(args.context_data, 'r', encoding='utf-8') as f:
        f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            item_index, item_context = line.split('\t', maxsplit=1)
            data['docid'].append(item_index)
            data['text'].append(item_context)

    content_data = {}
    for docid, text in zip(data['docid'], data['text']):
        content_data[docid] = text

    return content_data

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.5, emb_name='word_embedding.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embedding.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class Freeze():
    def __init__(self, model, batch_all, layer_num=12):
        self.model = model
        self.batch_num = 0
        self.batch_all = batch_all
        self.layer_num = layer_num
        self.old = 'none'
    def step(self):

        tmp = int(self.batch_all / 4 / self.layer_num) # 1/4个epoch后全部解冻
        layer_index = str(max(0, 12 - self.batch_num // tmp))
        if self.old != layer_index:
            self.old = layer_index
            print('unfreeze layer:{}'.format(layer_index))
        if layer_index == '0':
            for name, param in self.model.named_parameters():
                param.requires_grad = True
        else:
            flag = 0
            for name, param in self.model.named_parameters():
                if 'transformer' not in name: # 自己定义的层不冻结
                    param.requires_grad = True
                elif layer_index not in name and flag == 0: # 未解冻的层
                    param.requires_grad = False
                else:
                    flag = 1 # 表示之后的层解冻
                    param.requires_grad = True

        self.batch_num += 1

if __name__ == "__main__":

    # avg_conten: 1600, avg_ans: 58, avg_ques: 25
    # ans.split: 1:4070, 2:520, 3:226, 4:79, 5:41
    split_train_and_dev()
    # read_content()
    # refine_from_topk()