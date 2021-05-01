from xlnet.tokenization_xlnet import XLNetTokenizer
from xlnet.modeling_xlnet import XLNetConfig, XLNetverify_cls, XLNet_rerank
from tqdm import tqdm
import torch
import json
import args
import pickle
import numpy as np
import pandas as pd
from function import encode_pieces, normalized, read_corpus, para_recall
import sentencepiece as spm

def make_test_data(topk_path, pkl_name):

    print('load test data in {}'.format(topk_path))
    with open('./data/similarity/id2index.pkl', 'rb') as f:
        id2index = pickle.load(f)
    with open(topk_path, 'rb') as f:
        topk = pickle.load(f)
    print('1643:', len(topk))
    print('top:', len(topk[list(topk.keys())[0]]))

    data = pd.read_csv(args.test_data, sep='\t')
    data = [{'qid': qid, 'question': question}
            for qid, question in zip(data['id'], data['question'])]

    test_data = []
    content_data = read_corpus()
    index_content_data = [_ for _ in range(len(content_data))]
    print('len(index_content_data):', len(index_content_data))

    for docid, text in content_data.items():
        index_content_data[id2index[docid]] = text
    for i in data:
        i['recall_paras'] = [index_content_data[index] for index in topk[i['qid']]]
        test_data.append(i)

    with open(pkl_name, 'w', encoding="utf-8") as fout:
        for feature in test_data:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')

def get_doc_strides(sp_model, content, max_c_len=args.max_c_len, ds=256):

    c_tokens = encode_pieces(sp_model, content)
    all_strides = []
    here_start = 0
    while here_start < len(c_tokens):
        here_c = ''.join(c_tokens[here_start:here_start + max_c_len])
        all_strides.append(here_c)
        here_start += ds
    if len(c_tokens) <= 512:
        return all_strides[:1]
    if all_strides[-1] in all_strides[-2]:
        all_strides = all_strides[:-1]

    return all_strides[:15]

def predict(model, paths):

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load('data/spiece.model')

    device = torch.device("cuda", 1)
    tokenizer = XLNetTokenizer()

    model = model.to(device)

    predict_data = []

    with torch.no_grad():
        model.eval()
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                for step, line in enumerate(tqdm(f.readlines()[0::1])):

                    sample = json.loads(line)
                    qid, question, recall_paras = \
                        sample['qid'], sample['question'], sample['recall_paras']

                    recall_paras = [normalized(d) for d in recall_paras]
                    max_c_len = 512 - len(question) - 5
                    all_best_score = -999
                    pre_ans = ''

                    doc_head_len = len(encode_pieces(sp_model, question))

                    for text in recall_paras[:15]:
                        text = '凰'.join([x for x in text.split(' ') if len(x) != 0])
                        doc_strides = get_doc_strides(sp_model, content=text, max_c_len=max_c_len, ds=256)

                        for ds_id, doc_span in enumerate(doc_strides):

                            tokens = ['<cls>'] + encode_pieces(sp_model, question) + ['<sep>'] + \
                                         encode_pieces(sp_model, doc_span)

                            ques_len = len(encode_pieces(sp_model, question))
                            
                            input_ids = tokenizer.convert_tokens_to_ids(tokens)
                            tokentype_ids = [1] * len(input_ids)
                            tokentype_ids[:ques_len + 1] = [0] * (ques_len + 1)
                            assert len(tokentype_ids) == len(input_ids)

                            input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
                            tokentype_ids = torch.tensor(tokentype_ids).unsqueeze(0).to(device)

                            start, end, verify_gate, cls_logit = model(input_ids, token_type_ids=tokentype_ids)

                            is_ans = (start.cpu().squeeze().tolist()[0] + end.cpu().squeeze().tolist()[0])

                            #cls_logit = torch.nn.Softmax(dim=-1)(cls_logit)
                            #cls_logit = cls_logit.cpu().squeeze().tolist()

                            start, end = start.cpu().squeeze().tolist(), end.cpu().squeeze().tolist()
                            # verify_gate = torch.nn.Softmax(dim=-1)(verify_gate)
                            verify_gate = verify_gate.cpu().squeeze().tolist()
                            verify_gate = [i[1] for i in verify_gate]

                            start_g = sorted(start)[-5:][0]
                            end_g = sorted(end)[-5:][0]

                            for s, s_prob in enumerate(start[1:-2]):
                                if s < doc_head_len - 2:
                                    continue
                                if s_prob < start_g:
                                    continue
                                for e, e_prob in enumerate(end[s + 1:s + 1 + 280]):
                                    if e_prob < end_g:
                                        continue

                                    v_score = np.min(verify_gate[s + 1:e + s + 2])
                                    h_score = (s_prob + e_prob) * 1 - is_ans + v_score  # + 0.002*sim # - is_ans
                                    # h_score = h_score * cls_logit[1]

                                    if h_score > all_best_score:
                                        here_ans = ''.join(tokens[s + 1:e + s + 2])
                                        all_best_score = h_score
                                        pre_ans = here_ans

                    pre_ans = pre_ans.replace('凰',' ')
                    predict_data.append({'pre': pre_ans,
                                         'qid':qid,
                                         'question':question
                                         })

    with open('result.json', 'w', encoding="utf-8") as fout:
        for feature in predict_data:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')

def load_model(path):
    config = XLNetConfig('/home/lh/xlnet_model_base/config.json')
    #model = XLNetverify_cls(config)
    model = XLNet_rerank(config)
    import collections
    para = torch.load(path, 'cpu').state_dict()
    pretrained_dict = collections.OrderedDict()
    for i, j in para.items():
        i = i.replace('module.', '')
        pretrained_dict[i] = j
    model.load_state_dict(pretrained_dict)
    return model

def pre():

    #model = load_model('./checkpoints/single_model/yes_verify_soft_0.7836.pth')
    #model = load_model('./checkpoints/topdoc_model/top15_bert_recall_rerank_0.7224.pth')
    model = load_model('./checkpoints/final_model.pth')
    rouge_L = predict(model.cpu(), paths=['./data/all_test_data.json'])

if __name__ == '__main__':

    from train import train
    #train()
    make_test_data(topk_path='./data/similarity/bm25_test_2gram_top30.dict',
                   pkl_name='./data/all_test_data.json')
    pre()
