from xlnet.tokenization_xlnet import XLNetTokenizer
from bert.tokenization import BertTokenizer
from tqdm import tqdm
import pandas as pd
import torchtext
import pickle
import random
import torch
import json
import args
import csv
import re


"""
方案一： 每个文本当作一个训练例子，先不考虑每条数据里面各个文本的关系。
        对每个文本做完形填空，将#***#表示为[MASK]，在bert之后取出[MASK]对应的向量做3218分类。
        但是这没有考虑题目给出的10个候选的先验约束条件。（相当于BERT的MLM预训练）
        若要考虑约束，可以只将在候选集合的成语对应输出向量的位置取出来比较。也可以不做处理，后期看看两者的效果。
        
方案二： 将成语由四个字向量相加表示，将句子表示为一个向量，由这个句子向量与对应成语向量做点积，越大越好，
        与其他成语点积越小越好。
"""

def get_answer():
    answerDict = {}
    with open(args.answer_path, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            answerDict[row[0]] = row[1]
    print("len(answerDict): ",len(answerDict))
    return answerDict

def get_idiomDict():

    # with open(args.idiomDict_path, 'r', encoding='utf-8') as f:
    #     idioms = json.loads(f.readline())
    #     # print(idioms.keys())
    #     idiom2id = {idiom: ix  for ix, idiom in enumerate(idioms.keys())}
    #     id2idiom = {ix: idiom  for ix, idiom in enumerate(idioms.keys())}
    #     assert idiom2id[id2idiom[10]] == 10
    with open('id2idiom.pkl', 'rb') as f:
        id2idiom = pickle.load(f)
    with open('idiomDict.pkl', 'rb') as f:
        idiom2id = pickle.load(f)

    return idiom2id, id2idiom

def get_data():

    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    # with open('tokenizer.pkl', 'wb') as f:
    #     pickle.dump(tokenizer, f)
    idiom2id, id2idiom = get_idiomDict()

    answerDict = get_answer()

    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    tokenizer = XLNetTokenizer()

    print(tokenizer.vocab['<mask>'])
    print("len(tokenizer.vocab): ", len(tokenizer.vocab))

    train_data, dev_data = [], []
    avg_len = 0
    avg_len_1 = 0
    max_512 = 0
    with open(args.train_path, 'r', encoding='utf-8') as f:
        for step, line in enumerate(tqdm(f.readlines())):
            source = json.loads(line)

            candidates = [idiom2id[idiom] for idiom in source['candidates']]

            assert len(candidates) == 10

            for content in source['content']:
                index = [m.start() for m in re.finditer("#idiom", content)]

                pre_idiom = {content[n:n+13]: candidates[int(answerDict[content[n:n+13]])] for n in index}
                num = [candidates[int(answerDict[content[n:n+13]])] for n in index]
                label_relative = [int(answerDict[content[n:n+13]]) for n in index]

                for i in pre_idiom.keys():
                    content = content.replace(i, "¿")
                content = content.replace("“", '\"').replace("”", '\"').replace("’", '\"').replace("‘", '\"').replace('，',',')
                avg_len += len(content)
                avg_len_1 += len(tokenizer._tokenize(content))
                content = tokenizer._tokenize(content)
                while len(content) > 509:
                    max_512 += 1
                    drop = random.choice(range(len(content)))
                    if content[drop] != "¿":
                        del content[drop]

                input_ids = tokenizer.convert_tokens_to_ids(["<sep>"] + content + ["<sep>"]+["<cls>"])

                if step < 30:
                    print(''.join(tokenizer.convert_ids_to_tokens(input_ids)))
                    print(''.join(content))
                    print()
                index = []
                for n, number in enumerate(input_ids):
                    if number == 6:        # [MASK] : 103 , xlnet: <mask>:6
                        index.append(n)

                assert len(index) == len(num)

                label = [-1] * len(input_ids)
                label_mask = [0] * len(input_ids)
                for idx, i in enumerate(index):
                    if tokenizer.convert_ids_to_tokens([input_ids[i]])[0] != '<mask>':
                        print('error')
                        exit()
                    label[i] = num[idx]
                    label_mask[i] = 1

                if len(input_ids) > 512:
                    print("##### error")
                if (1+step) % 100 == 0:
                    dev_data.append(
                                    {"input_ids": input_ids,
                                     "candidate": candidates,
                                     "label_relative": label_relative,
                                     "label_mask": label_mask,
                                     "label": label})
                else:
                    train_data.append(
                                    {"input_ids":input_ids,
                                     "candidate":candidates,
                                     "label_relative": label_relative,
                                     "label_mask": label_mask,
                                     "label":label    })
    print('max_512:', max_512)
    print('avg_len:', avg_len/(len(train_data)+len(dev_data)))
    print('avg_len_1:', avg_len_1 / (len(train_data) + len(dev_data)))

    print("len(train_data): ", len(train_data)) # len(train_data): 456641, <512:len(train_data): 455354
    print("len(dev_data): ", len(dev_data))  # len(dev_data): 4601, <512: len(dev_data):  4591

    with open("./data/dev.json", 'w', encoding="utf-8") as fout:
        for feature in dev_data:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')
    with open("./data/train.json", 'w', encoding="utf-8") as fout:
        for feature in train_data:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')

def x_tokenize(ids):
    return [int(i) for i in ids]
def y_tokenize(y):
    return int(y)

class Dureader():
    def __init__(self, path='./data/'):
        self.WORD = torchtext.data.Field(batch_first=True, sequential=True, tokenize=x_tokenize,
                                             use_vocab=False,  # fix_length=args.max_p_len, #include_lengths=True ,
                                             pad_token=0)
        self.LABEL = torchtext.data.Field(batch_first=True, sequential=True, tokenize=x_tokenize,
                                             use_vocab=False,  # fix_length=args.max_p_len, #include_lengths=True ,
                                             pad_token=-1)
        #torchtext.data.Field(sequential=False, tokenize=y_tokenize, use_vocab=False, pad_token=-1)

        dict_fields = { 'input_ids':  ('input_ids', self.WORD),
                        'candidate':  ('candidate', self.WORD),
                        'label_relative': ('label_relative', self.LABEL),
                        'label_mask': ('label_mask', self.WORD),
                        'label':      ('label', self.LABEL),    }

        self.train, self.dev = torchtext.data.TabularDataset.splits(
                path=path,
                train="train.json",
                validation="dev.json",
                format='json',
                fields=dict_fields)
        self.train_iter, self.dev_iter = torchtext.data.BucketIterator.splits(
                [self.train, self.dev], batch_size=args.batch_size,
                sort_key=lambda x: len(x.input_ids), sort_within_batch=True, shuffle=True)


if __name__ == "__main__":
    # a = {'kjds':1}
    # print(list(a.items())[0][0])
    idiom2id, id2idiom = get_idiomDict()
    print('绿肥红瘦:', idiom2id['绿肥红瘦'])
    fre = {}
    # print(args.fre_idiom.values())
    for i, n  in args.fre_idiom.items():
        fre[idiom2id[i]] = n
    print(fre)
    fre = sorted(fre.items(), key=lambda x:x[0])
    print(fre)
    f = []
    for i in fre:
        if i[1] < 10:
            f.append(5.0)
        elif i[1] < 30:
            f.append(3.0)
        elif i[1] < 60:
            f.append(1.3)
        else:
            f.append(1.0)

    print(f)

    # print(id2idiom[572])
    # print(id2idiom, len(id2idiom))
    # get_data()
    # data = Dureader()
    # dev = data.dev_iter
    # for batch in dev:
    #     input_ids = batch.label_relative
    #     print(input_ids.shape)
    #     print(input_ids)
    #     print()
