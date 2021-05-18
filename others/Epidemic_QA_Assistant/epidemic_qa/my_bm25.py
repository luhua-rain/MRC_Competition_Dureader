#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains function of computing rank scores for documents in
corpus and helper class `BM25` used in calculations. Original algorithm
descibed in [1]_, also you may check Wikipedia page [2]_.


.. [1] Robertson, Stephen; Zaragoza, Hugo (2009).  The Probabilistic Relevance Framework: BM25 and Beyond,
       http://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf
.. [2] Okapi BM25 on Wikipedia, https://en.wikipedia.org/wiki/Okapi_BM25



Examples
--------

.. sourcecode:: pycon

    >>> from gensim.summarization.bm25 import get_bm25_weights
    >>> corpus = [
    ...     ["black", "cat", "white", "cat"],
    ...     ["cat", "outer", "space"],
    ...     ["wag", "dog"]
    ... ]
    >>> result = get_bm25_weights(corpus, n_jobs=-1)


Data:
-----
.. data:: PARAM_K1 - Free smoothing parameter for BM25.
.. data:: PARAM_B - Free smoothing parameter for BM25.
.. data:: EPSILON - Constant used for negative idf of document in corpus.

"""


import math
from six import iteritems
from six.moves import range
from functools import partial
from multiprocessing import Pool
from collections import Counter

# BM25
PARAM_K1 = 1.5
PARAM_K2 = 1.5
PARAM_B = 0.75
EPSILON = 0.25

# F1EXP
s = 0.15
k = 0.25

class BM25(object):
    """Implementation of Best Matching 25 ranking function.

    Attributes
    ----------
    corpus_size : int
        Size of corpus (number of documents).
    avgdl : float
        Average length of document in `corpus`.
    doc_freqs : list of dicts of int
        Dictionary with terms frequencies for each document in `corpus`. Words used as keys and frequencies as values.
    idf : dict
        Dictionary with inversed documents frequencies for whole `corpus`. Words used as keys and frequencies as values.
    doc_len : list of int
        List of document lengths.
    """

    def __init__(self, corpus):
        """
        Parameters
        ----------
        corpus : list of list of str
            Given corpus.

        """
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self._initialize(corpus)
        self.keyword = ['中关村管委会', '中医局', '中央后勤保障部', '中央政治局常委会', '交通委', '交通运输部', '人力社保局', '人力资源', '人力资源和社会保障局', '人力资源社会保障部', '人力资源部', '人民政府', '人民政府办公室', '人民检察院', '人社厅', '人社部', '人防办', '住建委', '住房公积金管理中心', '住房城乡建设委', '体育局', '侨联', '信息化部', '信访办', '公园管理中心', '公安局', '公安部', '农业农村局', '农业农村部', '农机监理所', '农村农业部', '办公厅', '医保局', '医疗保险协会', '卫健委', '卫生健康委', '卫生健康部', '友协', '发展改革委', '发改委', '台办', '司法局', '商务局', '商务部', '团市委', '园林绿化局', '国务院', '国家体育总局', '国家发展改革委', '国家发改委', '国家林草局', '国家税务总局', '国家能源局', '国家药品监管局', '国家铁路局', '国税局', '国资委', '地方志编委会', '地方金融监管局', '城市管理委', '城管执法局', '外汇管理局', '天安门地区管委会', '妇联', '审计局', '审计署', '密码管理局', '工业和信息化部', '工信部', '市场监督管理局', '市场监管局', '市场监管总局', '市委组织部', '市教委', '广播电视局', '应急管理局', '总工会', '扶贫办', '投资促进服务中心', '政务服务中心', '政务服务大厅', '政务服务管理局', '政府侨办', '政府外办', '教委', '教育局', '教育督导委员会', '教育部', '文化和旅游局', '文化执法总队', '文物局', '服贸司', '林草局', '档案局', '残联', '民政局', '民政部', '民族宗教委', '气象局', '气象部', '水利部', '水务局', '海关', '海关总署', '烟草局', '煤矿安监局', '生态环境局', '生态环境部', '电子税务局', '省委办公厅', '省政府办公厅', '知识产权局', '社会保障部', '社科联', '科协', '科委', '税务局', '税务总局', '粮食和物资储备局', '粮食物资局', '红十字会', '经济信息化局', '经济技术开发区管委会', '统计局', '编办', '能源局', '自然资源部', '药监局', '规划自然资源委', '财政局', '财政部', '资源部', '退役军人事务局', '邮政局', '邮政管理局', '重大项目办', '金融监督管理局', '银保监局', '首都文明办', '高级人民法院']

    def _initialize(self, corpus):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        self.nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.corpus_size += 1
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.nd:
                    self.nd[word] = 0
                self.nd[word] += 1

        self.avgdl = float(num_doc) / self.corpus_size
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in iteritems(self.nd):
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = float(idf_sum) / len(self.idf)

        eps = EPSILON * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_score(self, document, index):
        """Computes BM25 score of given `document` in relation to item of corpus selected by `index`.

        Parameters
        ----------
        document : list of str
            Document to be scored.
        index : int
            Index of document in corpus selected to score with `document`.

        Returns
        -------
        float
            BM25 score.

        """
        keyfreq_dict = Counter(document)
        score = 0
        doc_freqs = self.doc_freqs[index]
        for word in document:
            if word not in doc_freqs:
                continue

            # keyfreq = keyfreq_dict[word]*(PARAM_K2+1)/(keyfreq_dict[word]+PARAM_K2) #math.log(keyfreq_dict[word]+1)

            score += (self.idf[word] * doc_freqs[word] * (PARAM_K1 + 1)
                      / (doc_freqs[word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.doc_len[index] / self.avgdl)))

            # score += (1 / (1 + s + s * self.doc_len[index] / self.avgdl) *
            #           ((self.corpus_size + 1) / self.nd[word])**k
            #           )
            # if word in self.keyword:
            #     score *= 1.3
        return score

    def get_scores(self, document):
        """Computes and returns BM25 scores of given `document` in relation to
        every item in corpus.

        Parameters
        ----------
        document : list of str
            Document to be scored.

        Returns
        -------
        list of float
            BM25 scores.

        """
        scores = [self.get_score(document, index) for index in range(self.corpus_size)]
        return scores

    def get_scores_bow(self, document):
        """Computes and returns BM25 scores of given `document` in relation to
        every item in corpus.

        Parameters
        ----------
        document : list of str
            Document to be scored.

        Returns
        -------
        list of float
            BM25 scores.

        """
        scores = []
        for index in range(self.corpus_size):
            score = self.get_score(document, index)
            if score > 0:
                scores.append((index, score))
        return scores


def _get_scores_bow(bm25, document):
    """Helper function for retrieving bm25 scores of given `document` in parallel
    in relation to every item in corpus.

    Parameters
    ----------
    bm25 : BM25 object
        BM25 object fitted on the corpus where documents are retrieved.
    document : list of str
        Document to be scored.

    Returns
    -------
    list of (index, float)
        BM25 scores in a bag of weights format.

    """
    return bm25.get_scores_bow(document)


def _get_scores(bm25, document):
    """Helper function for retrieving bm25 scores of given `document` in parallel
    in relation to every item in corpus.

    Parameters
    ----------
    bm25 : BM25 object
        BM25 object fitted on the corpus where documents are retrieved.
    document : list of str
        Document to be scored.

    Returns
    -------
    list of float
        BM25 scores.

    """
    return bm25.get_scores(document)


def iter_bm25_bow(corpus, n_jobs=1):
    """Yield BM25 scores (weights) of documents in corpus.
    Each document has to be weighted with every document in given corpus.

    Parameters
    ----------
    corpus : list of list of str
        Corpus of documents.
    n_jobs : int
        The number of processes to use for computing bm25.

    Yields
    -------
    list of (index, float)
        BM25 scores in bag of weights format.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.summarization.bm25 import iter_bm25_weights
        >>> corpus = [
        ...     ["black", "cat", "white", "cat"],
        ...     ["cat", "outer", "space"],
        ...     ["wag", "dog"]
        ... ]
        >>> result = iter_bm25_weights(corpus, n_jobs=-1)

    """
    bm25 = BM25(corpus)

    n_processes = effective_n_jobs(n_jobs)
    if n_processes == 1:
        for doc in corpus:
            yield bm25.get_scores_bow(doc)
        return

    get_score = partial(_get_scores_bow, bm25)
    pool = Pool(n_processes)

    for bow in pool.imap(get_score, corpus):
        yield bow
    pool.close()
    pool.join()


def get_bm25_weights(corpus, n_jobs=1):
    """Returns BM25 scores (weights) of documents in corpus.
    Each document has to be weighted with every document in given corpus.

    Parameters
    ----------
    corpus : list of list of str
        Corpus of documents.
    n_jobs : int
        The number of processes to use for computing bm25.

    Returns
    -------
    list of list of float
        BM25 scores.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.summarization.bm25 import get_bm25_weights
        >>> corpus = [
        ...     ["black", "cat", "white", "cat"],
        ...     ["cat", "outer", "space"],
        ...     ["wag", "dog"]
        ... ]
        >>> result = get_bm25_weights(corpus, n_jobs=-1)

    """
    bm25 = BM25(corpus)

    n_processes = effective_n_jobs(n_jobs)
    if n_processes == 1:
        weights = [bm25.get_scores(doc) for doc in corpus]
        return weights

    get_score = partial(_get_scores, bm25)
    pool = Pool(n_processes)
    weights = pool.map(get_score, corpus)
    pool.close()
    pool.join()
    return weights
