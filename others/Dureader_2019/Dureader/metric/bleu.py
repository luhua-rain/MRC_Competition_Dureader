# coding:utf8

from functools import reduce
import math
import json
from collections import defaultdict
import sys 
import importlib
import common
importlib.reload(sys)


class BLEU(object):
    def __init__(self, n_size):
        self.match_ngram = {}
        self.candi_ngram = {}
        self.bp_r = 0
        self.bp_c = 0
        self.n_size = n_size

    def add_inst(self, cand, ref_list):
        for n_size in range(self.n_size):
            self.count_ngram(cand, ref_list, n_size)
        self.count_bp(cand, ref_list)

    def count_ngram(self, cand, ref_list, n_size):
        cand_ngram = common.get_ngram(cand, n_size)
        refs_ngram = []
        for ref in ref_list:
            refs_ngram.append(common.get_ngram(ref, n_size))
        if n_size not in self.match_ngram:
            self.match_ngram[n_size] = 0
            self.candi_ngram[n_size] = 0
        match_size, cand_size = common.get_match_size(cand_ngram, refs_ngram)
        self.match_ngram[n_size] += match_size
        self.candi_ngram[n_size] += cand_size

    def count_bp(self, cand, ref_list):
        self.bp_c += len(cand)
        self.bp_r += min([
            (abs(len(cand) - len(ref)), len(ref))
            for ref in ref_list]
            )[1]

    def score(self):
        prob_list = []
        for n_size in range(self.n_size):
            try:
                if self.candi_ngram[n_size] == 0:
                    _score = 0.0
                else:
                    _score = self.match_ngram[n_size] / float(self.candi_ngram[n_size])
            except:
                _score = 0
            prob_list.append(_score)
        bleu_list = [prob_list[0]]
        for n in range(1, self.n_size):
            bleu_list.append(bleu_list[-1] * prob_list[n])
        for n in range(self.n_size):
            bleu_list[n] = bleu_list[n] ** (1./float(n+1))
        if float(self.bp_c) == 0.0:
            bp = 0.0
        else:
            bp = math.exp(min(1 - self.bp_r / float(self.bp_c), 0))
        for n in range(self.n_size):
            bleu_list[n] = bleu_list[n] * bp
        return bleu_list

class BLEUWithBonus(BLEU):
    def __init__(self, n_size, alpha=1.0, beta=1.0):
        super(BLEUWithBonus, self).__init__(n_size)
        self.alpha = alpha
        self.beta = beta

    def add_inst(self,
            cand,
            ref_list,
            yn_label=None, yn_ref=None, entity_ref=None):
        #super(BLEUWithBonus, self).add_inst(cand, ref_list)
        BLEU.add_inst(self, cand, ref_list)
        if yn_label is not None and yn_ref is not None:
            self.add_yn_bonus(cand, ref_list, yn_label, yn_ref)
        elif entity_ref is not None:
            self.add_entity_bonus(cand, entity_ref)

    def add_yn_bonus(self, cand, ref_list, yn_label, yn_ref):
        for n_size in range(self.n_size):
            cand_ngram = common.get_ngram(cand, n_size, label=yn_label)
            ref_ngram = []
            for ref_id, r in enumerate(yn_ref):
                ref_ngram.append(common.get_ngram(ref_list[ref_id], n_size, label=r))
            match_size, cand_size = common.get_match_size(cand_ngram, ref_ngram)
            self.match_ngram[n_size] += self.alpha * match_size
            self.candi_ngram[n_size] += self.alpha * match_size

    def add_entity_bonus(self, cand, entity_ref):
        for n_size in range(self.n_size):
            cand_ngram = common.get_ngram(cand, n_size, label='ENTITY')
            ref_ngram = []
            for reff_id, r in enumerate(entity_ref):
                ref_ngram.append(common.get_ngram(r, n_size, label='ENTITY'))
            match_size, cand_size = common.get_match_size(cand_ngram, ref_ngram)
            self.match_ngram[n_size] += self.beta * match_size
            self.candi_ngram[n_size] += self.beta * match_size
