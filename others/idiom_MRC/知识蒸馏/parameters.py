import logging
import torch as t
import argparse
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re
from tqdm import tqdm, trange
import ipdb
from torch.utils.data import random_split
import torch.multiprocessing
from torchnet import meter
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
import tensorboardX
from tensorboardX import SummaryWriter
from pytorch_transformers import XLNetModel, XLNetConfig, XLNetTokenizer, AdamW, XLNetPreTrainedModel


# 这里是各种配置的声明
class Config():
    def __init__(self):
        self.vocab_root = "../data/vocab.txt"
        self.xlnet_config_root = "../data/config.json"
        self.pretrained_xlnet_root = "../data/pytorch_model.bin"
        self.tokenizer_root = "../data/spiece.model"
        self.raw_train_data_root = "../data/train.txt"
        self.split_train_data_root = "../data/split_train_data.json"
        self.raw_train_label_root = "../data/train_answer.csv"
        self.raw_test_data_root = "../data/dev.txt"
        self.model_root = "../model/"
        self.data_root = "../data/"
        self.idiom_vocab_root = "../data/idiomList.txt"
        self.prob_file = "../data/prob.csv"
        self.result_file = "../data/result.csv"
        self.raw_result_file = "../data/raw_result.csv"
        self.xlnet_learning_rate = 2e-5
        self.other_learning_rate = 1e-3
        self.max_seq_length = 128
        self.num_train_epochs = 100
        self.warmup_proportion = 0.01
        self.hidden_dropout_prob = 0.5
        self.num_workers = 8
        self.eval_ratio = 0.02
        with open(self.data_root + "idiom2index", mode="rb") as f1:
            self.idiom2index = pickle.load(f1)
        with open(self.data_root + "index2idiom", mode="rb") as f2:
            self.index2idiom = pickle.load(f2)
        self.use_gpu = t.cuda.is_available()
        self.device = t.device("cuda" if self.use_gpu else "cpu")
        self.n_gpu = t.cuda.device_count()
        self.train_batch_size = 10 * self.n_gpu * int(256 / self.max_seq_length)
        self.test_batch_size = 32 * self.n_gpu * int(256 / self.max_seq_length)
        self.logger = logging.getLogger("xlnetCloze_train")
        self.logger.setLevel(logging.INFO)
        self.writer = SummaryWriter('tensorlog')
        self.decay = 0.3
        self.min_lr = 5e-7
        self.patience = 1
        self.seed = 42
        self.show_loss_step = 200
        self.version = 30
        self.tokenizer = XLNetTokenizer.from_pretrained(self.tokenizer_root)

    def show_members(self):
        show_str = "the config is defined as below\n"
        for name, value in vars(self).items():
            if name in ['writer', 'idiom2index', 'index2idiom', 'tokenizer', 'logger']:
                continue
            show_str += '\t%s=%s\n' % (name, value)
        return show_str

# 这是进行日志的初始化
def handle_log():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_predict', action="store_true", default=False)
    args = parser.parse_args()
    config.do_predict = args.do_predict

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 使用FileHandler输出日志到文件，仅在训练时进行
    if not config.do_predict:
        fh = logging.FileHandler('%slog%s.log' % (config.data_root, config.version), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        config.logger.addHandler(fh)

    # 使用StreamHandler输出日志到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    config.logger.addHandler(ch)

    config.logger.info(config.show_members())
    config.logger.info("use env: {} , num of gpu: {}".format(config.device, config.n_gpu))

    jieba.setLogLevel(logging.INFO)


torch.multiprocessing.set_sharing_strategy('file_system')
config = Config()
handle_log()
