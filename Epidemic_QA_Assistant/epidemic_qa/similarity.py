# -*- coding:utf-8 -*-
# author:zjl

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pkuseg
import json
import time
from tqdm import tqdm, trange
import args
from function import normalized
import os
import numpy as np
import pandas as pd
from gensim import corpora
# from gensim.summarization import bm25
from my_bm25 import BM25
from function import read_corpus, encode_pieces
import my_bm25
import sentencepiece as spm

sp_model = spm.SentencePieceProcessor()
sp_model.Load('data/spiece.model')

SPIECE_UNDERLINE = '▁'

SAVE_PATH = './data/similarity'


def out_put_time():
    time_array = time.localtime(int(time.time()))
    print('time: ',time.strftime("%Y-%m-%d %H:%M:%S", time_array))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def context2json(context_path, save_path):
    res = []
    with open(context_path, 'r', encoding='utf-8') as f:
        f.readline()
        id2index = dict()
        index2id = dict()
        while True:
            line = f.readline()
            if not line:
                break
            item_index, item_context = line.split('\t', maxsplit=1)
            id2index[item_index] = len(id2index.keys())
            index2id[len(index2id.keys())] = item_index
            res.append(json.dumps({'id': item_index, 'context': item_context}))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(os.path.join(save_path, 'context.json'), 'w', encoding='utf-8') as g:
            g.write('\n'.join(res))
        with open(os.path.join(save_path, 'id2index.pkl'), 'wb') as g:
            pickle.dump(id2index, g)
        with open(os.path.join(save_path, 'index2id.pkl'), 'wb') as g:
            pickle.dump(index2id, g)

def get_index_id_trans(id2index_path, index2id_path):
    with open(id2index_path, 'rb') as f1:
        id2index = pickle.load(f1)
    with open(index2id_path, 'rb') as f2:
        index2id = pickle.load(f2)
    return id2index, index2id


def get_seg(seg: str):
    keyword = ['中关村管委会', '中医局', '中央后勤保障部', '中央政治局常委会', '交通委', '交通运输部', '人力社保局', '人力资源', '人力资源和社会保障局', '人力资源社会保障部', '人力资源部', '人民政府', '人民政府办公室', '人民检察院', '人社厅', '人社部', '人防办', '住建委', '住房公积金管理中心', '住房城乡建设委', '体育局', '侨联', '信息化部', '信访办', '公园管理中心', '公安局', '公安部', '农业农村局', '农业农村部', '农机监理所', '农村农业部', '办公厅', '医保局', '医疗保险协会', '卫健委', '卫生健康委', '卫生健康部', '友协', '发展改革委', '发改委', '台办', '司法局', '商务局', '商务部', '团市委', '园林绿化局', '国务院', '国家体育总局', '国家发展改革委', '国家发改委', '国家林草局', '国家税务总局', '国家能源局', '国家药品监管局', '国家铁路局', '国税局', '国资委', '地方志编委会', '地方金融监管局', '城市管理委', '城管执法局', '外汇管理局', '天安门地区管委会', '妇联', '审计局', '审计署', '密码管理局', '工业和信息化部', '工信部', '市场监督管理局', '市场监管局', '市场监管总局', '市委组织部', '市教委', '广播电视局', '应急管理局', '总工会', '扶贫办', '投资促进服务中心', '政务服务中心', '政务服务大厅', '政务服务管理局', '政府侨办', '政府外办', '教委', '教育局', '教育督导委员会', '教育部', '文化和旅游局', '文化执法总队', '文物局', '服贸司', '林草局', '档案局', '残联', '民政局', '民政部', '民族宗教委', '气象局', '气象部', '水利部', '水务局', '海关', '海关总署', '烟草局', '煤矿安监局', '生态环境局', '生态环境部', '电子税务局', '省委办公厅', '省政府办公厅', '知识产权局', '社会保障部', '社科联', '科协', '科委', '税务局', '税务总局', '粮食和物资储备局', '粮食物资局', '红十字会', '经济信息化局', '经济技术开发区管委会', '统计局', '编办', '能源局', '自然资源部', '药监局', '规划自然资源委', '财政局', '财政部', '资源部', '退役军人事务局', '邮政局', '邮政管理局', '重大项目办', '金融监督管理局', '银保监局', '首都文明办', '高级人民法院']
    if seg == 'pkuseg':
        return pkuseg.pkuseg(user_dict=keyword).cut
    else:
        return str.split

def tokenization(text, seg, stopwords):
    result = []
    words = seg(text)
    for word, flag in words:
        if word not in stopwords:
            result.append(word)
    return result

class TFIDF():
    @staticmethod
    def train(json_path, seg, save_path):
        # Similarity.get_index_id_trans()
        print('===== loading data =====')
        with open(json_path, 'r') as f:
            real_documents = []
            g = f.read().split('\n')
            for _, data_piece in enumerate(tqdm(g)):
                context = json.loads(data_piece)['context']
                item_str = encode_pieces(sp_model, context)#seg(context)
                real_documents.append(item_str)
        tfidf_vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            max_df=0.7,
            min_df=1/3000
        )
        document = [' '.join(sent) for sent in real_documents]
        tfidf_model = tfidf_vectorizer.fit(document)
        # 下面这个pickle存不了。。
        # sparse matrix, [n_samples, n_features] Tf-idf-weighted document-term matrix.
        sparse_result = tfidf_model.transform(document).todense()
        with open(os.path.join(save_path, 'tfidf_model.pkl'), 'wb') as f:
            pickle.dump(tfidf_model, f)
        with open(os.path.join(save_path, 'document.pkl'), 'wb') as f:
            pickle.dump(document, f)

    @staticmethod
    def predict(query, seg, tfidf_model, sparse_result):
        test_document = [' '.join(encode_pieces(sp_model, query))]
        result = tfidf_model.transform(test_document).todense()
        scores = np.array(result * sparse_result.T)[0]
        return list(reversed(list(np.argsort(scores))))[:300]

    @staticmethod
    def context_eval(seg, tfidf_model, sparse_result):
        id2index, index2id = get_index_id_trans(os.path.join(SAVE_PATH, 'id2index.pkl'), os.path.join(SAVE_PATH, 'index2id.pkl'))
        print('===== loading data =====')
        acc = 0
        total = 0
        with open(os.path.join(SAVE_PATH, 'context.json'), 'r') as f:
            g = f.read().split('\n')
            for _, data_piece in enumerate(tqdm(g)):
                context = normalized(json.loads(data_piece)['context'])
                res = TFIDF.predict(context, seg, tfidf_model, sparse_result)
                total += 1
                if index2id[json.loads(data_piece)['id']] in res:
                    acc += 1
        print("total: {}, acc: {}, rate: {}".format(total, acc, acc/total))
        return acc/total

    @staticmethod
    def train_eval(train_data, seg, id2index, tfidf_model, sparse_result):
        print('===== loading data =====')
        tmp = {}
        ret_tmp = np.zeros(300)
        with open(train_data, 'r', encoding='utf-8') as f:
            g = f.read().split('\n')
            for _, data_piece in enumerate(tqdm(g[-500:])):
                context = json.loads(data_piece)['question']
                res = TFIDF.predict(context, seg, tfidf_model, sparse_result)
                tmp[json.loads(data_piece)['id']] = res
                doc_index = id2index[json.loads(data_piece)['docid']]
                if doc_index in res:
                    list_index = res.index(doc_index)
                    ret_tmp += np.array([0 for i in range(list_index)] + [1 for i in range(list_index, 300)])
        with open('./data/similarity/tfidf_train_top300.pkl', 'wb') as f:
            pickle.dump(tmp, f)
        return ret_tmp

    @staticmethod
    def test_eval(test_data, seg, save_path, tfidf_model, sparse_result, topk=1):
        data = pd.read_csv(test_data, sep='\t')
        res = {}
        for i in trange(len(data)):
            id = data['id'][i]
            tmp = TFIDF.predict(data['question'][i], seg, tfidf_model, sparse_result)[:topk]
            res[id] = tmp
        with open(os.path.join(save_path, 'tfidf_test_top{}.pkl'.format(topk)), 'wb') as f:
            pickle.dump(res, f)

def get_all_question_token():
    # 1-gram:5190、2-gram:28560、all:33688
    seg = get_seg('pkuseg')
    data = pd.read_csv(args.train_data, sep='\t')
    data = [{'docid': docid, 'question': question, 'answer': answer}
            for id, docid, question, answer in zip(data['id'], data['docid'], data['question'], data['answer'])]
    train_question = [n['question'] for n in data]
    tokens = []
    for q in tqdm(train_question):
        if q[-1] in '？?': # 去掉问号
            q = q[:-1]
        q = seg(q)
        tokens.extend(q)
        for x in range(len(q[:]) - 1):
            tokens.append(''.join([q[x], q[x + 1]]))

    tokens = list(set(tokens))
    # print(tokens[:100])
    print('question 1、2gram tokens:', len(tokens))
    return tokens

class nCoV_BM25():
    @staticmethod
    def train(json_path, seg, save_path, stopwords):
        all_question_gram = get_all_question_token()

        print('===== loading data =====')
        with open(json_path, 'r') as f:
            corpus = []
            g = f.read().split('\n')
            for _, data_piece in enumerate(tqdm(g)):

                context = normalized(json.loads(data_piece)['context'])
                # 切词
                item_str = seg(context)
                # 去停用词
                doc = []
                for word in item_str:
                    if word not in stopwords:
                        doc.append(word)
                # 2-gram
                ngram = [''.join([doc[x], doc[x + 1]]) for x in range(len(doc[:]) - 1)]
                ngram = set(ngram)
                ngram = list(ngram & set(all_question_gram)) # 过滤掉不在问题中的2-gram
                # ngram = [x for x in ngram if x in all_question_gram]
                doc.extend(ngram)

                corpus.append(doc)
        dictionary = corpora.Dictionary(corpus)
        print('document dictionary length: {}'.format(len(dictionary)))
        # corpus为文本全集
        bm25Model = BM25(corpus)
        with open(os.path.join(save_path, 'bm25_2gram.Model'), 'wb') as f:
            pickle.dump(bm25Model, f)

    @staticmethod
    def get_avg_idf(bm25Model):
        return sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())

    @staticmethod
    def predict(from_train, query, seg, bm25Model, stopwords):
        train_question, question2id = from_train
        context = normalized(query)
        item_str = seg(context)
        # n-gram
        ngram = [''.join([item_str[x], item_str[x + 1]]) for x in range(len(item_str[:]) - 1)]
        item_str.extend(ngram)

        doc = []
        for word in item_str:
            if word not in stopwords:
                doc.append(word)
        avg_idf = nCoV_BM25.get_avg_idf(bm25Model)
        scores = bm25Model.get_scores(doc)
        scores = np.array(scores)
        # fak = find_from_train(query, train_question, question2id)
        # scores[fak] = (scores[fak]+1)*10
        return list(reversed(list(np.argsort(scores))))[:300]

    @staticmethod
    def train_eval(from_train, train_data, seg, bm25Model, stopwords, id2index):
        print('===== loading data =====')
        tmp = {}
        ret_tmp = np.zeros(300)
        with open(train_data, 'r', encoding='utf-8') as f:
            g = f.read().split('\n')
            for _, data_piece in enumerate(tqdm(g[-500:])):
                question = json.loads(data_piece)['question']
                res = nCoV_BM25.predict(from_train, question, seg, bm25Model, stopwords)
                tmp[json.loads(data_piece)['id']] = res
                doc_index = id2index[json.loads(data_piece)['docid']]
                if doc_index in res:
                    list_index = res.index(doc_index)
                    ret_tmp += np.array([0 for i in range(list_index)] + [1 for i in range(list_index, 300)])
        with open('./data/similarity/bm25_train_2gram_top300.pkl', 'wb') as f:
            pickle.dump(tmp, f)
        return ret_tmp

    @staticmethod
    def dev_eval(from_train, dev_data, seg, save_path, bm25Model, stopwords, topk=1):
        res = {}
        with open(dev_data, 'r', encoding='utf-8') as f:
            for step, line in enumerate(tqdm(f.readlines())):
                sample = json.loads(line)
                id, question = sample['id'], sample['question']
                tmp = nCoV_BM25.predict(from_train, question, seg, bm25Model, stopwords)[:topk]
                res[id] = tmp

        with open(os.path.join(save_path, 'bm25_dev_top{}.pkl'.format(topk)), 'wb') as f:
            pickle.dump(res, f)

    @staticmethod
    def test_eval(from_train, test_data, seg, save_path, bm25Model, stopwords, topk=1):
        data = pd.read_csv(test_data, sep='\t')
        res = {}
        for i in trange(len(data)):
            id = data['id'][i]
            tmp = nCoV_BM25.predict(from_train, data['question'][i], seg, bm25Model, stopwords)[:topk]
            res[id] = tmp
        with open(os.path.join(save_path, 'bm25_test_top{}.pkl'.format(topk)), 'wb') as f:
            pickle.dump(res, f)

def use_bm25(mode, data_path, seg, save_path, bm25Model, stopwords, topk=5):

    corpus = read_corpus()
    data = pd.read_csv(args.train_data, sep='\t')
    data = [{'docid': docid, 'question': question, 'answer': answer, 'text': corpus[docid]}
            for id, docid, question, answer in zip(data['id'], data['docid'], data['question'], data['answer'])]
    train_question = []
    question2id = {}
    for i in data:
        answer, question, text, docid = i['answer'], i['question'], i['text'], i['docid']
        question2id[question] = docid
        train_question.append(question)

    if mode == 'dev_eval':
        nCoV_BM25.dev_eval(
            [train_question, question2id], dev_data=data_path,seg=seg,
            save_path=save_path,bm25Model=bm25Model,stopwords=stopwords,topk=topk)
    elif mode == 'test_eval':
        nCoV_BM25.test_eval(
            [train_question, question2id], test_data=data_path, seg=seg,
            save_path=save_path, bm25Model=bm25Model, stopwords=stopwords, topk=topk)
    elif mode == 'train_eval':
        id2index, index2id = get_index_id_trans(os.path.join(save_path, 'id2index.pkl'),
                                                os.path.join(save_path, 'index2id.pkl'))
        ret_tmp = nCoV_BM25.train_eval(
            from_train=[train_question, question2id],
            train_data=data_path,seg=seg,bm25Model=bm25Model,stopwords=stopwords,id2index=id2index)
        tmp = []
        print('k=', my_bm25.PARAM_K1)
        for i in range(0, 20, 1):
            tmp.append(ret_tmp[i] / 500)
            print(i+1, ret_tmp[i] / 500)
        print(tmp)

    elif mode == 'train_model':
        nCoV_BM25.train('./data/similarity/context.json', seg=seg, save_path=save_path, stopwords=stopwords)

def use_tfidf(mode, data_path, seg, save_path, tfidf_model, topk=5):

    with open(os.path.join(save_path, 'document.pkl'), 'rb') as f:
        document = pickle.load(f)
    sparse_result = tfidf_model.transform(document).todense()

    if mode == 'test_eval':
        TFIDF.test_eval(
            test_data=data_path, seg=seg, sparse_result=sparse_result,
            save_path=save_path, tfidf_model=tfidf_model, topk=topk)
    elif mode == 'train_eval':
        id2index, index2id = get_index_id_trans(os.path.join(save_path, 'id2index.pkl'),
                                                os.path.join(save_path, 'index2id.pkl'))
        ret_tmp = TFIDF.train_eval(train_data=data_path,seg=seg,tfidf_model=tfidf_model,
                                   id2index=id2index, sparse_result=sparse_result)
        tmp = []
        for i in range(0, 100, 1):
            tmp.append(ret_tmp[i] / 500)
            print(i+1, ret_tmp[i] / 500)
        print(tmp)

    elif mode == 'train_model':
        TFIDF.train('./data/similarity/context.json', seg=seg, save_path=save_path)


if __name__ == '__main__':

    # seg = get_seg('pkuseg')
    # print(seg('人力资源和社会保障局'))
    # exit()
    # context2json(args.context_data, SAVE_PATH)

    # all_train_data_5000 = pd.read_csv('./data/NCPPolicies_train_20200301.csv', sep='\t')
    # all_train_data_5000 = [
    #     {'id':id, 'docid': docid, 'question': question, 'answer': answer}
    #     for id, docid, question, answer in zip(all_train_data_5000['id'], all_train_data_5000['docid'], all_train_data_5000['question'], all_train_data_5000['answer'])
    # ]
    # for i in range(len(all_train_data_5000)):
    #     all_train_data_5000[i] = json.dumps(all_train_data_5000[i])
    # with open('./data/all_train_data_5000.json', 'w') as f:
    #     f.write('\n'.join(all_train_data_5000))

    id2index, index2id = get_index_id_trans(os.path.join(SAVE_PATH, 'id2index.pkl'), os.path.join(SAVE_PATH, 'index2id.pkl'))
    with open('./data/stopwords/hit_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = set(f.read().split('\n')) | set(';；”“"：: ,，.。【[]】的地凰得不是？?/@#$%^&*()（）～·`') | set(['\n', '\t', ' '])

    test_data = './data/NCPPolicies_test.csv'
    dev_data = './data/dev_data.json'
    train_data = './data/all_train_data_5000.json'

    # idf = bm25Model.idf
    # idf = sorted(idf.items(), key=lambda x:x[1])
    # print(len(idf))
    # exit()
    # use_bm25(mode='train_model', data_path=None, seg=get_seg('pkuseg'), save_path=SAVE_PATH,
    #          bm25Model=None, stopwords=stopwords, topk=60)
    # use_tfidf(mode='train_model', data_path=None, seg=None, save_path=SAVE_PATH, tfidf_model=None, topk=60)

    # bm25Model = pickle.load(open(os.path.join(SAVE_PATH, 'bm25_2gram.Model'), 'rb'))
    tfidf_model = pickle.load(open(os.path.join(SAVE_PATH, 'tfidf_model.pkl'), 'rb'))

    # for k in [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.7, 3]:
    #     my_bm25.PARAM_K1 = k
    # use_bm25(mode='train_eval', data_path=train_data, seg=get_seg('pkuseg'), save_path=SAVE_PATH,
    #      bm25Model=bm25Model, stopwords=stopwords, topk=60)


    use_tfidf('train_eval', train_data, None, SAVE_PATH, tfidf_model=tfidf_model, topk=60)