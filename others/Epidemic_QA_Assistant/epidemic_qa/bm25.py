import torch
import torch.nn as nn
import args
import random
import numpy as np
import json
import sentencepiece as spm
from tqdm import tqdm
import six
import pickle

sp_model = spm.SentencePieceProcessor()
sp_model.Load('./data/spiece.model')

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

class mat_BM25():
    def __init__(self, corpus=None, param_k1=1.5, param_b=0.75, epsilon=0.25):

        with open('../data/hit_stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set(f.read().split('\n')) | set(';；”“"：: ,，.。【[]】的地得不是？?/@#$%^&*()（）～·`') | set(['\n', '\t', ' '])
        with open('../data/context.json', 'r') as f:
            corpus = []
            g = f.read().split('\n')
            for data_piece in tqdm(g):
                context = json.loads(data_piece)['context']
                # 切词
                item_str = encode_pieces(sp_model, context)
                # 去停用词
                doc = []
                for word in item_str:
                    if word not in stopwords:
                        doc.append(word)

                corpus.append(doc)

        self.doc_len = []
        self.corpus_size = len(corpus)  # 文档数量
        self.param_k1, self.param_b, self.epsilon = param_k1, param_b, epsilon
        self.index2word, self.word2index, self.word_size, self.avg_dl = None, None, 0, 0
        self.f, self.df, self.idf, self.avg_idf = [], None, None, 0
        self.loss_func = nn.CrossEntropyLoss()
        self.initialize(corpus)

    def initialize(self, corpus):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        print('initialize begin')
        word_set = set()
        for document in corpus:
            word_set |= set(document)
        self.word_size = len(word_set)
        self.word2index = dict(zip(list(word_set), range(self.word_size)))
        self.index2word = dict(zip(range(self.word_size), list(word_set)))

        for document in corpus:
            vector = torch.zeros([1, self.word_size])
            self.doc_len.append(len(document))
            for word in document:
                vector[0][self.word2index[word]] += 1
            self.f.append(vector)

        self.avg_dl = (sum(self.doc_len) / self.corpus_size)      # 平均文档长度
        print('generate document frequency')
        self.f = torch.cat(self.f)     # (corpus_size, word_size)

        apperance = (self.f > 0) * 1
        apperance = apperance.float()
        # print(self.f.shape, '\n', apperance.shape, apperance)

        self.df = torch.ones([1, self.corpus_size])    # 每个词出现的文档个数
        self.df = torch.matmul(self.df, apperance)     # (1, word_size)
        self.idf = torch.log(self.corpus_size - self.df + 0.5) - torch.log(self.df + 0.5)   # \log\frac{N-n_i+0.5}{n_i+0.5} # (1, word_size)
        self.avg_idf = float(torch.sum(self.idf) / self.word_size)

        print('generate inversed document frequency')
        self.idf[self.idf < 0] = self.epsilon * self.avg_idf

        self.idf = self.idf[0].detach().to(args.device)
        self.f = self.f.detach().to(args.device)
        self.doc_len = torch.transpose(torch.tensor([self.doc_len]), 0, 1).detach().float()

        self.doc_len = self.doc_len.to(args.device)
        print('initialize over')

    def get_scores(self, document, vector=None):
        if vector is None:
            vector = torch.ones([len(document),2 ])     # 全部相等
        scores = torch.zeros([self.corpus_size, 1]).to(args.device) + vector[0] * 0
        #print(vector.device)
        #print(scores.device)

        for i, word in enumerate(document):
            if word not in self.word2index:
                continue
            index = self.word2index[word]
            f = self.f[:, index].unsqueeze(1).float().to(args.device)    # 所有文档的word词频： (corpus_size, 1)
            #print(self.doc_len.device)

            scores += (vector[i][0] * self.idf[index] + vector[i][1]) * f * (self.param_k1 + 1) \
                / (f + self.param_k1 * (1 - self.param_b + self.param_b * self.doc_len / self.avg_dl))
        return torch.transpose(scores, 0, 1)[0]

    def input_batch(self, batch_data):
        # 输入的batch为[(得分向量, 切词后的问题, 问题对应的政策序号), ()...]

        batch_size = len(batch_data)
        correct, total = 0, batch_size

        data, labels = [], []
        ret_tmp = np.zeros(300)
        for vectors, document, true_index in batch_data:

            res = self.get_scores(document, vectors)#(8932)

            args_index = list(reversed(torch.argsort(res).tolist()))
            neg_nums = 8932        # 负样本个数
            index_piece = args_index[:neg_nums+1]       # 一条数据的序号(1-8932)
            if true_index not in index_piece:
                index_piece[-1] = true_index
            random.shuffle(index_piece)
            true_label = index_piece.index(true_index)      # 在一条数据中的位置
            tmp_data = []
            for index in index_piece:
                tmp_data.append(res[index].unsqueeze(0))                 # 每条样本的分数
            data.append(torch.cat(tmp_data).unsqueeze(0))
            labels.append(torch.tensor(true_label).to(args.device))         # 正例的位置

            list_index = index_piece.index(true_index)
            if list_index < 300:
                ret_tmp += np.array([0 for i in range(list_index)] + [1 for i in range(list_index, 300)])

        data = torch.cat(data, dim=0)
        labels = torch.tensor(labels).to(data.device)

        loss = self.loss_func(data, labels)

        _, predict = torch.max(data, dim=-1)
        correct = (predict == labels).sum()

        return loss, correct, total, ret_tmp


if __name__ == '__main__':
    mat_BM25_model = mat_BM25()
    with open('../data/attention_bm25.model', 'wb') as f:
        pickle.dump(mat_BM25_model, f)
