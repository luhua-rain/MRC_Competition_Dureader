# -*- coding: utf-8 -*-
import sentencepiece as spm
import six
from function import normalized, RougeL, F1, encode_pieces
from tqdm import tqdm
import json
import args
from function import read_corpus
import pickle
import random
from collections import Counter

class Process(object):

    def __init__(self):
        self.f1 = F1()
        self.rouge_L = RougeL()
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load('data/spiece.model')

    def find_answer(self, most_related_para_tokens, segmented_answer):
        # find matching ans_spans
        start, end = [], []
        for i in range(0, len(most_related_para_tokens) - len(segmented_answer) + 1):
            if segmented_answer == most_related_para_tokens[i: i + len(segmented_answer)]:
                start.append(i)
                end.append(i + len(segmented_answer) - 1)
        if len(start) == 0:
            start.append(0)
            end.append(0)

        return start, end

    def make_test_data(self, topk_path, pkl_name):
        not_recall = 0
        print('load test data in {}'.format(topk_path))
        with open('./data/similarity/id2index.pkl', 'rb') as f:
            id2index = pickle.load(f)
        with open(topk_path, 'rb') as f:
            topk = pickle.load(f)
        print('4500:', len(topk))
        print('top:', len(topk[list(topk.keys())[0]]))
        data = []
        with open(args.split_train_data, 'r', encoding='utf-8') as f:
            for step, line in enumerate(tqdm(f.readlines())):
                sample = json.loads(line)
                sample['top'] = topk[sample['id']][:15]
                if id2index[sample['docid']] not in sample['top']:
                    sample['top'][0] = id2index[sample['docid']] # 百分百召回
                    not_recall += 1
                data.append(sample)
        print('not_recall: ',not_recall)
        dev_data = []
        content_data = read_corpus()
        index_content_data = [_ for _ in range(len(content_data))]
        for docid, text in content_data.items():
            index_content_data[id2index[docid]] = text
        print('len(index_content_data):', len(index_content_data))

        for i in data:
            i['recall_paras'] = [index_content_data[index] for index in i['top']]
            dev_data.append(i)
            del i['top']

        with open(pkl_name, 'w', encoding="utf-8") as fout:
            for feature in dev_data:
                fout.write(json.dumps(feature, ensure_ascii=False) + '\n')

    def get_doc_strides(self, content, max_c_len=args.max_c_len, ds=256):
        c_tokens = encode_pieces(self.sp_model, content)
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

        return all_strides[:15]

    def process_data(self, file_name, pkl_name): # 入口函数

        process_data = []
        ans = []
        a = 0
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):

                sample = json.loads(line)

                docid, question, answer, text = sample['docid'], sample['question'], sample['answer'], sample['text']
                recall_paras = sample['recall_paras']
                pre_sample = {'question': question, 'docs': [], 'answers':answer}

                recall_paras = [normalized(n) for n in recall_paras]
                text = normalized(text)
                answer = normalized(answer)

                ans.append(answer)
                #recall_paras = [text]
                doc_stride = []
                for _, text in enumerate(recall_paras[:5]):

                    max_c_len = 512 - len(encode_pieces(self.sp_model, question)) - 5

                    #text = '凰'.join([x for x in text.split(' ') if len(x) != 0])
                    #answer = '凰'.join([x for x in answer.split(' ') if len(x) != 0])
                    doc_stride.extend(self.get_doc_strides(text, max_c_len=max_c_len, ds=256))

                doc_stride = list(set(doc_stride))
                for ds_id, doc_span in enumerate(doc_stride):

                        doc_span_token = ['<cls>'] + encode_pieces(self.sp_model, question) + ['<sep>'] + \
                                        encode_pieces(self.sp_model, doc_span)

                        ref_ans_token = encode_pieces(self.sp_model, answer)

                        start, end = self.find_answer(doc_span_token, ref_ans_token)
                        if start[0] == 0:
                            ans_dict = {'is_impossible': True, 'answers': [[0, 0, 0, 0]]}
                            doc = {'content': doc_span_token, 'ans_dict': ans_dict}
                            pre_sample['docs'].append(doc)
                            a += 1
                        else:
                            ans_dict = {'is_impossible': False, 'answers': [[0, 0, answer, 0]], 'muti_ans':[start, end]}
                            doc = {'content': doc_span_token, 'ans_dict': ans_dict}
                            pre_sample['docs'].append(doc)

                process_data.append(pre_sample)

        print('no-ans:',a)
        print('len(ans):',len(set(ans)))
        print('len(process_data): ', len(process_data))
        with open(pkl_name, 'w', encoding="utf-8") as fout:
            for feature in process_data:
                fout.write(json.dumps(feature, ensure_ascii=False) + '\n')

if __name__ == '__main__':

    process = Process()

    process.make_test_data(topk_path='./data/similarity/bm25_train_top300.pkl', pkl_name=args.split_train_data)
    process.process_data(args.split_train_data, pkl_name=args.process_split_train_data)

