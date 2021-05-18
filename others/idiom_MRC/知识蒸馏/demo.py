import logging
import torch as t
import argparse
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertModel, BertConfig, BertTokenizer, AdamW, BertPreTrainedModel
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re
from tqdm import tqdm, trange
import ipdb
import torch.multiprocessing
from torch.optim.lr_scheduler import *
import numpy as np
import random
import json
import os
import jieba
from random import shuffle
import time
from nltk.corpus import wordnet as wn
import pandas as pd
import math


class Config():
    def __init__(self):
        self.use_gpu = t.cuda.is_available()
        self.vocab_root = "../kernel/vocab.txt"
        self.bert_config_root = "../kernel/bert_config.json"
        self.pretrained_bert_root = "../kernel/chr_idiombert.bin"
        self.raw_test_data_root = "../data/test.txt"
        self.test_ans_root = "../kernel/dev_ans.csv"
        self.idiom_vocab_root = "../kernel/idiomList.txt"
        self.prob_file = "../kernel/prob.csv"
        self.data_root = "../kernel/"
        self.split_test_data_root = "../kernel/split_test_data.json"
        self.tokenizer = BertTokenizer(vocab_file=self.vocab_root)
        self.num_workers = 4
        self.test_batch_size = 512
        self.max_seq_length = 128
        with open(self.data_root + "idiom2index", mode="rb") as f1:
            self.idiom2index = pickle.load(f1)
        with open(self.data_root + "index2idiom", mode="rb") as f2:
            self.index2idiom = pickle.load(f2)
        self.hidden_dropout_prob = 0.5
        self.use_gpu = t.cuda.is_available()
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")


config = Config()


class BertCloze(BertPreTrainedModel):
    def __init__(self, bert_config, num_choices):
        super(BertCloze, self).__init__(bert_config)
        self.num_choices = num_choices
        self.bert = BertModel(bert_config)
        self.idiom_embedding = nn.Embedding(len(config.idiom2index), bert_config.hidden_size)
        self.my_fc = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(bert_config.hidden_size, 1)
        )
        # self.apply(self.init_weights)

    def forward(self, input_ids, option_ids, token_type_ids, attention_mask, positions, tags, labels=None):
        # print(input_ids.shape,option_ids.shape,token_type_ids.shape)
        encoded_layer, _ = self.bert(input_ids, token_type_ids, attention_mask)
        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        # encoded_layer [bs, maxseq, 768] blank_states [bs,768]
        encoded_options = self.idiom_embedding(option_ids)  # [bs, 10, 768]
        multiply_result = t.einsum('abc,ac->abc', encoded_options, blank_states)  # [bs, 10, 768]
        # ipdb.set_trace()
        logits = self.my_fc(multiply_result)
        reshaped_logits = logits.view(-1, self.num_choices)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss, reshaped_logits
        else:
            return reshaped_logits

    def freeze_all(self):
        for para in self.parameters():
            para.requires_grad = False


def load_model(model, model_path):
    model_CKPT = t.load(model_path)
    model.load_state_dict(model_CKPT)
    model.freeze_all()
    model.eval()
    model.to(config.device)
    print("load {} successfully".format(model_path))


def gen_vocab():
    vocab = eval(open(config.idiom_vocab_root, mode="r").readline())
    idiom2index = {idiom: idx for idx, idiom in enumerate(vocab)}
    index2idiom = {v: k for k, v in idiom2index.items()}
    with open(config.data_root + "idiom2index", mode="wb") as f1:
        pickle.dump(idiom2index, f1)
    with open(config.data_root + "index2idiom", mode="wb") as f2:
        pickle.dump(index2idiom, f2)


def get_ansdict(file_path):
    ans_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.split(',')
            ans_dict[line[0]] = int(line[1])
    return ans_dict


def gen_split_data():
    print("处理测试数据中...")
    input_data = open(config.raw_test_data_root, mode="r", encoding='utf-8')

    examples = []

    for data in tqdm(input_data):
        # print(data)
        data = eval(data)
        options = data['candidates']
        for context in data['content']:
            tags = re.findall("#idiom\d+#", context)
            for tag in tags:
                tmp_context = context
                for other_tag in tags:
                    if other_tag != tag:
                        tmp_context = tmp_context.replace(other_tag, "[UNK]")
                examples.append({
                    "tag": tag,
                    "context": tmp_context,
                    "options": options,
                })
    json.dump(examples, open(config.split_test_data_root, "w"))
    print("数据预处理完成")
    return examples


def convert_sentence_to_features(context, options, tag, label=None):
    parts = re.split(tag, context)
    assert len(parts) == 2
    before_part = config.tokenizer.tokenize(parts[0]) if len(parts[0]) > 0 else []
    after_part = config.tokenizer.tokenize(parts[1]) if len(parts[1]) > 0 else []

    half_length = int(config.max_seq_length / 2)
    if len(before_part) < half_length:  # cut at tail
        st = 0
        ed = min(len(before_part) + 1 + len(after_part), config.max_seq_length - 2)
    elif len(after_part) < half_length:  # cut at head
        ed = len(before_part) + 1 + len(after_part)
        st = max(0, ed - (config.max_seq_length - 2))
    else:  # cut at both sides
        st = len(before_part) + 3 - half_length
        ed = len(before_part) + 1 + half_length

    option_ids = [config.idiom2index[each] for each in options]
    tokens = before_part + ["[MASK]"] + after_part
    tokens = ["[CLS]"] + tokens[st:ed] + ["[SEP]"]
    position = tokens.index("[MASK]")
    input_ids = config.tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    padding = [0] * (config.max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    res = [input_ids, input_mask, segment_ids, option_ids, position, int(tag[6: -1])]
    if label is not None:
        res.append(label)
    new_res = []
    for item in res:
        new_res.append(t.tensor(item).long())
    return new_res


class IdiomData(Dataset):
    def __init__(self):
        self.split_data = gen_split_data()
        random.shuffle(self.split_data)
        print("读取数据结束")

    def __getitem__(self, idx):
        context = self.split_data[idx]["context"]
        options = self.split_data[idx]["options"]
        tag = self.split_data[idx]["tag"]
        return convert_sentence_to_features(
            context=context, options=options, tag=tag)

    def __len__(self):
        return len(self.split_data)


def getdataLoader():
    idiomdata = IdiomData()
    dataloader = DataLoader(idiomdata, batch_size=config.test_batch_size, shuffle=False,
                            num_workers=config.num_workers)
    return dataloader


def to_device(*args):
    ans = []
    for val in args:
        val = val.to(config.device)
        ans.append(val)
    return ans


def generate_prob(model):
    test_dataloader = getdataLoader()
    all_results = {}
    for batch in tqdm(test_dataloader):
        input_ids, input_mask, segment_ids, option_ids, positions, tags = batch
        input_ids, input_mask, segment_ids, option_ids, positions, tags = to_device(
            input_ids, input_mask, segment_ids, option_ids, positions, tags
        )
        with torch.no_grad():
            batch_logits = model(input_ids, option_ids, segment_ids, input_mask, positions, tags)
        for i, tag in enumerate(tags):
            logits = batch_logits[i].detach().cpu().numpy()
            prob = F.softmax(t.tensor(logits), dim=0)
            if "#idiom%06d#" % tag in all_results:
                all_results["#idiom%06d#" % tag] += prob
            else:
                all_results["#idiom%06d#" % tag] = prob
    with open(config.prob_file, "w") as f:
        for each in all_results:
            f.write(each)
            for i in range(10):
                f.write(',' + str(all_results[each][i].item()))
            f.write("\n")
    print("选项概率分布生成完成")


header_list = ['id', 'p0', 'p1', 'p2', 'p3', 'p4',
               'p5', 'p6', 'p7', 'p8', 'p9']
res = []
tmp = []
tmp_pos = []
max_prob = - 10000000


def get_group():
    testfile = open(config.raw_test_data_root, "r", encoding='utf-8')
    text = testfile.readlines()
    group = []
    for line in text:
        tags = re.findall("#idiom\d+#", line)
        group.append(tags)     # 一组就是一行
    print("数据分组完毕")
    return group


def beam_search(ids, probs, log_probs, ranks, order, i_range):
    # print(i_range)
    global max_prob, tmp, res, tmp_pos
    if order == len(ids):
        now_prob = 0
        # print("start")
        for k in range(order):
            # print(probs[k][tmp_pos[k]])
            # now_prob += probs[k][tmp_pos[k]]
            now_prob += log_probs[k][tmp_pos[k]]
        # print(now_prob)
        # print("end")
        if now_prob > max_prob:
            max_prob = now_prob
            # print(now_prob, tmp)
            res = tmp.copy()
        return
    for i in range(i_range):
        now_rank = ranks[order][i]
        flag = True
        for j in range(order):
            if tmp[j] == now_rank:
                flag = False
        if not flag:
            continue
        tmp[order] = now_rank
        tmp_pos[order] = i
        beam_search(ids, probs, log_probs, ranks, order + 1, i_range=i_range)


def generate_result(i_range=5):
    global max_prob, tmp, res, tmp_pos
    groups = get_group()
    ave_prob = pd.read_csv(config.prob_file, header=None, names=header_list, sep=',',
                           index_col='id')
    reses = []
    for group in tqdm(groups):  # 一行
        probs = []
        ids = []
        ranks = []
        log_probs = []
        now_prob = 0
        for id in group:
            ids.append(id)
            prob = list(ave_prob.loc[id, :])
            rank = list(np.argsort(-np.array(prob)))
            # ipdb.set_trace()
            prob.sort(reverse=True)
            log_prob = list(map(lambda x: math.log(x, 10), prob))
            # ipdb.set_trace()
            probs.append(prob)
            ranks.append(rank)
            log_probs.append(log_prob)
        for i in range(len(ids)):
            now_prob += probs[i][0]
        assert len(ids) == len(probs)
        assert len(probs[0]) == 10
        res = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        tmp = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        tmp_pos = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        max_prob = - 10000000
        # print(probs, ranks, ids)
        beam_search(ids, probs, log_probs, ranks, 0, i_range=i_range)
        really_res = res[0:len(ids)]
        # assert len(really_res) == len(set(really_res))
        # print(probs,really_res)
        reses.append((ids, really_res))
        # ipdb.set_trace()
    assert len(reses) == len(groups)
    search_idx = open("submission.csv", "w")
    for id, res in reses:
        for i in range(len(id)):
            search_idx.write(id[i] + ',' + str(res[i]) + '\n')
    print("结果搜索完成")


def check_result():
    def read_ans(filename):
        tmp_dict = {}
        with open(filename, "r") as f:
            for line in f:
                line = line.split(',')
                tmp_dict[line[0]] = int(line[1])
        return tmp_dict

    true_ans = read_ans(config.test_ans_root)
    my_ans = read_ans("submission.csv")
    true_count, all_count = 0, 0
    for k, v in tqdm(true_ans.items()):
        if v == my_ans[k]:
            true_count += 1
        all_count += 1
    print("结果准确率为 %f", float(true_count / all_count))


def start(check_accr=False):
    bert_config = BertConfig.from_json_file(config.bert_config_root)
    model = BertCloze(bert_config, num_choices=10)
    load_model(model, config.pretrained_bert_root)
    generate_prob(model)
    generate_result(i_range=5)
    if check_accr:
        check_result()
    print("程序运行完成")


start(check_accr=True)