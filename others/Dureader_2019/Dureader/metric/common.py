# coding:utf8
from functools import reduce
import math
import json
from collections import defaultdict
import sys

def get_match_size(cand_ngram, refs_ngram):
    ref_set = defaultdict(int)
    for ref_ngram in refs_ngram:
        tmp_ref_set = defaultdict(int)
        for ngram in ref_ngram:
            tmp_ref_set[ngram] += 1
        for ngram, count in tmp_ref_set.items():
            ref_set[ngram] = max(ref_set[ngram], count)
    cand_set = defaultdict(int)
    for ngram in cand_ngram:
        cand_set[ngram] += 1
    match_size = 0
    for ngram, count in cand_set.items():
        match_size += min(count, ref_set.get(ngram, 0))
    cand_size = len(cand_ngram)
    return match_size, cand_size

def get_ngram(sent, n_size, label=None):
    def _ngram(sent, n_size):
        ngram_list = []
        for left in range(len(sent) - n_size):
            ngram_list.append(sent[left: left + n_size + 1])
        return ngram_list

    ngram_list = _ngram(sent, n_size)
    if label is not None:
        ngram_list = [ngram + '_' + label for ngram in ngram_list]
    return ngram_list

def word2char(str_in):
    str_out = str_in.replace(' ', '')
    return ''.join(str_out.split())
