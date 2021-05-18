# coding:utf8

from functools import reduce
import math
import json
import numpy as np
from collections import defaultdict
import sys
import importlib
importlib.reload(sys)


class RougeL(object):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.inst_scores = []

    def lcs(self, string, sub):
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

    def add_inst(self,
            cand,
            ref_list,
            yn_label=None, yn_ref=None, entity_ref=None):
        precs, recalls = [], []
        for i, ref in enumerate(ref_list):
            basic_lcs = self.lcs(cand, ref)
            yn_bonus, entity_bonus = 0.0, 0.0
            if yn_ref is not None and yn_label is not None:
                yn_bonus = self.add_yn_bonus(cand, ref, yn_label, yn_ref[i])
            elif entity_ref is not None:
                entity_bonus = self.add_entity_bonus(cand, entity_ref)
            p_denom = len(cand) + self.alpha * yn_bonus + self.beta * entity_bonus
            r_denom = len(ref) + self.alpha * yn_bonus + self.beta * entity_bonus
            prec = (basic_lcs + self.alpha * yn_bonus + self.beta * entity_bonus) \
                    / p_denom if p_denom > 0. else 0.
            rec = (basic_lcs + self.alpha * yn_bonus + self.beta * entity_bonus) \
                    / r_denom if r_denom > 0. else 0.
            precs.append(prec)
            recalls.append(rec)

        prec_max = max(precs)
        rec_max = max(recalls)
        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.gamma**2) * prec_max * rec_max) / \
                    float(rec_max + self.gamma**2 * prec_max)
        else:
            score = 0.0
        self.inst_scores.append(score)

    def add_yn_bonus(self, cand, ref, yn_label, yn_ref):
        if yn_label != yn_ref:
            return 0.0
        lcs_ = self.lcs(cand, ref)
        return lcs_

    def add_entity_bonus(self, cand, entity_ref):
        lcs_ = 0.0
        for ent in entity_ref:
            if ent in cand:
                lcs_ += len(ent)
        return lcs_

    def score(self):
        return 1. * sum(self.inst_scores) / len(self.inst_scores)
