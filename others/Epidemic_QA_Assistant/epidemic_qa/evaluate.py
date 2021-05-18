# -*- coding:utf-8 -*-
from xlnet.tokenization_xlnet import XLNetTokenizer
from xlnet.modeling_xlnet import XLNetConfig, XLNetverify_cls, XLNet_pure, XLNet_rerank
from process import RougeL
from tqdm import tqdm
import torch
import json
import args
import numpy as np
from function import normalized
import pickle
from process import  encode_pieces
import sentencepiece as spm
from function import read_corpus, para_recall
from collections import Counter

def get_seg(seg: str):
    if seg == 'pkuseg':
        return pkuseg.pkuseg().cut
    else:
        return str.split

def get_all_subdoc():
    all_corpus = []
    with open(args.context_data, 'r', encoding='utf-8') as f:
        f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            item_index, item_context = line.split('\t', maxsplit=1)
            all_corpus.append({'id': item_index, 'context': item_context})
    # 1024/512 : 26717; 512/256 : 53501
    subdoc_len = 1024
    ds = 1024
    all_subdoc = []
    for step, i in enumerate(all_corpus):
        c_tokens = i['context']
        if len(c_tokens) < subdoc_len:
            all_subdoc.append(c_tokens)
        else:
            here_start = 0
            while here_start < len(c_tokens):
                here_c = c_tokens[here_start:here_start + subdoc_len]
                all_subdoc.append(here_c)
                here_start += ds
            if all_subdoc[-1] in all_subdoc[-2]:
                all_subdoc = all_subdoc[:-1]
    return all_subdoc

def make_dev_data(topk_path, pkl_name):
    print('load test data in {}'.format(topk_path))
    with open('./data/similarity/id2index.pkl', 'rb') as f:
        id2index = pickle.load(f)
    with open(topk_path, 'rb') as f:
        topk = pickle.load(f)
    print('500:', len(topk))
    print('top:', len(topk[list(topk.keys())[0]]))
    data = []
    with open(args.split_dev_data, 'r', encoding='utf-8') as f:
        for step, line in enumerate(tqdm(f.readlines())):
            sample = json.loads(line)
            sample['top'] = topk[sample['id']]
            data.append(sample)

    dev_data = []
    content_data = read_corpus()
    index_content_data = [_ for _ in range(len(content_data))]

    for docid, text in content_data.items():
        index_content_data[id2index[docid]] = text
    #index_content_data = get_all_subdoc()
    print('len(index_content_data):', len(index_content_data))

    for i in data:
        i['recall_paras'] = [index_content_data[index] for index in i['top']]
        i['top'] = [str(n) for n in i['top']]
        dev_data.append(i)

    with open(pkl_name, 'w', encoding="utf-8") as fout:
        for feature in dev_data:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')

def get_doc_strides(sp_model, content, max_c_len=args.max_c_len, ds=256):

    c_tokens = encode_pieces(sp_model, content)
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

def evaluate(model, one_or_more='one', path=args.split_dev_data):

    with open('./data/similarity/id2index.pkl', 'rb') as f:
        id2index = pickle.load(f)

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load('data/spiece.model')

    rouge_L = RougeL(beta=1)

    device = torch.device("cuda",0)
    tokenizer = XLNetTokenizer()

    model = model.to(device)

    predict_data = []
    hit_rate = {'docid_correct':0, 'ans_in_doc_correct':0, 'total':0, 'docid_score':0, 'ans_in_doc_score':0, 'wrong_score':0}
    look_data = []
    with torch.no_grad():
        model.eval()
        with open(path, 'r', encoding='utf-8') as f:
            for step, line in enumerate(tqdm(f.readlines()[0::1])):

                sample = json.loads(line)
                docid, question, answer, text, recall_paras = \
                    sample['docid'], sample['question'], sample['answer'], sample['text'], sample['recall_paras']
                top = [int(t) for t in sample['top']]

                recall_paras = [normalized(n) for n in recall_paras]
                ori_text = normalized(text)
                answer = normalized(answer)

                all_best_score = -999
                pre_ans = ''
                doc_index = top[0]

                doc_head_len = len(encode_pieces(sp_model, question))

                if one_or_more == 'more':
                    texts = recall_paras[:15]
                else:
                    texts = [text]

                for doc_num, text in enumerate(texts):
                    text = '凰'.join([x for x in text.split(' ') if len(x) != 0])

                    max_c_len = 512 - len(encode_pieces(sp_model, question)) - 5
                    doc_strides = get_doc_strides(sp_model, content=text, max_c_len=max_c_len, ds=256)

                    for ds_id, doc_span in enumerate(doc_strides):

                        tokens = ['<cls>'] + encode_pieces(sp_model, question) + ['<sep>'] + \
                                         encode_pieces(sp_model, doc_span)

                        ques_len = len(encode_pieces(sp_model, question))
                        input_ids = tokenizer.convert_tokens_to_ids(tokens)

                        tokentype_ids = [1] * len(input_ids)
                        tokentype_ids[:ques_len + 1] = [0] * (ques_len + 1)
                        assert len(tokentype_ids) == len(input_ids)

                        tokentype_ids = torch.tensor(tokentype_ids).unsqueeze(0).to(device)
                        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

                        start, end, verify_gate, cls_logit = model(input_ids, token_type_ids=tokentype_ids)

                        # cls_logit = torch.nn.Softmax(dim=-1)(cls_logit)
                        # cls_logit = cls_logit.cpu().squeeze().tolist()

                        start, end = start.cpu().squeeze().tolist(), end.cpu().squeeze().tolist()
                        #verify_gate = torch.nn.Softmax(dim=-1)(verify_gate)
                        verify_gate = verify_gate.cpu().squeeze().tolist()
                        verify_gate = [i[1] for i in verify_gate]

                        is_ans = (start[0] + end[0])

                        start_g = sorted(start)[-5:][0]
                        end_g = sorted(end)[-5:][0]

                        for s, s_prob in enumerate(start[1:-2]):
                            # 不遍历doc_head
                            if s < doc_head_len-2:
                                continue
                            if s_prob < start_g:
                                continue
                            for e, e_prob in enumerate(end[s + 1:s + 1 + 280]):
                                if e_prob < end_g:
                                    continue

                                v_score = np.min(verify_gate[s + 1:e + s + 2])
                                h_score = (s_prob + e_prob)*1 - is_ans + v_score
                                #h_score = h_score * cls_logit[1]
                                if doc_num > 14 and h_score > 0:
                                    h_score *= 0.7
                                if h_score > all_best_score:
                                    here_ans = ''.join(tokens[s + 1:e + s + 2])
                                    all_best_score = h_score
                                    pre_ans = here_ans
                                    doc_index = top[doc_num]

                pre_ans = pre_ans.replace('凰',' ')
                sc = rouge_L.get_rouge_L(answer, pre_ans)
                if doc_index == id2index[docid]:
                    hit_rate['docid_correct'] += 1
                    hit_rate['ans_in_doc_correct'] += 1
                    hit_rate['docid_score'] += sc
                    hit_rate['ans_in_doc_score'] += sc
                elif answer in recall_paras[top.index(doc_index)]:
                    hit_rate['ans_in_doc_correct'] += 1
                    hit_rate['ans_in_doc_score'] += sc
                else:
                    is_docid_in_top15 = id2index[docid] in top[:15]
                    look_data.append({'predict_ans':pre_ans,
                                      'real_ans':answer,
                                      'top15_docs':recall_paras[:15],
                                      'model_choose_doc':recall_paras[top.index(doc_index)],
                                      'real_doc':ori_text,
                                      'is_docid_in_top15':is_docid_in_top15,
                                      'question':question,
                                      'rouge-L':sc})

                    hit_rate['wrong_score'] += sc

                hit_rate['total'] += 1

                predict_data.append({'pre':pre_ans,
                                     'rel':answer})
                if step % 20 == -1:
                    print('docid_hit_rate:{}\n'
                          'ans_in_doc_hit_rate:{}\n'
                          'docid_score:{}\n'
                          'ans_in_doc_score:{}\n'
                          'wrong_score:{}\n'.format(
                        float(hit_rate['docid_correct']) / (hit_rate['total']),
                        float(hit_rate['ans_in_doc_correct']) / (hit_rate['total']),
                        float(hit_rate['docid_score']) / (hit_rate['total']),
                        float(hit_rate['ans_in_doc_score']) / (hit_rate['total']),
                        float(hit_rate['wrong_score']) / (hit_rate['total'])
                    ))

    score = 0.0
    for n in predict_data:
        a = rouge_L.get_rouge_L(n['pre'], n['rel'])
        score += a
    print('docid_hit_rate:{}\n'
          'ans_in_doc_hit_rate:{}\n'
          'docid_score:{}\n'
          'ans_in_doc_score:{}\n'
          'wrong_score:{}\n'.format(
        float(hit_rate['docid_correct']) / (hit_rate['total']),
        float(hit_rate['ans_in_doc_correct']) / (hit_rate['total']),
        float(hit_rate['docid_score']) / (hit_rate['total']),
        float(hit_rate['ans_in_doc_score']) / (hit_rate['total']),
        float(hit_rate['wrong_score']) / (hit_rate['total'])
    ))
    print('rouge_L : ', score / len(predict_data))

    with open('look_data.json', 'w', encoding="utf-8") as fout:
        for feature in look_data:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')

    return score / len(predict_data)

def eval(one_or_more):

    config = XLNetConfig('/home/lh/xlnet_model_base/config.json')
    #model = XLNetverify_cls(config)
    # model = XLNetverify_cls_(config)
    model = XLNet_rerank(config)
    # model = XLNet_pure(config)
    import collections
    para = torch.load(args.best_model_path, 'cpu').state_dict()
    pretrained_dict = collections.OrderedDict()
    for i, j in para.items():
        #if 'embedding' in i:
         #   print(i)
        i = i.replace('module.', '')
        pretrained_dict[i] = j

    model.load_state_dict(pretrained_dict)
    print(len(pretrained_dict))

    rouge_L = evaluate(model.cpu(), one_or_more, path=args.split_dev_data)

if __name__ == "__main__":

    make_dev_data(topk_path='./data/similarity/bm25_train_top300.pkl', pkl_name=args.split_dev_data)
    eval(one_or_more='more')



