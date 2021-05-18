# -*- coding: utf-8 -*-

from xlnet.tokenization_xlnet import XLNetTokenizer
from process import encode_pieces
import sentencepiece as spm
from tqdm import tqdm
import random
import torch
import json
import args
import pandas as pd
from collections import Counter

class F1(object):
    def get_F1(self, string, sub):  # (question, content)
        if string == '' or sub =='':
            return 0.0
        common = Counter(string) & Counter(sub)
        overlap = sum(common.values())
        recall, precision = overlap/len(sub), overlap/len(string)
        return precision
        # return (2*recall*precision) / (recall+precision+1e-12)

def gen_context(tokens, fake_answer, positive_or_negative, other=None):
    # function for context Binary classification
    tokens = ''.join(tokens)
    paras = []
    for x in tokens.lower().split('。'):
        # paras.extend([n for n in x.split('凰') if len(n) != 0])
        paras.append(x)

    if len(paras) <= 2:
        return [0,0,0]
        # print('paras<=2:', tokens)

    if positive_or_negative == True:
        if fake_answer[-1] == '。':
            fake_answer = fake_answer[:-1]

        ans_index = -1
        for index, x in enumerate(paras):
            if fake_answer in x or x in fake_answer:
                ans_index = index
                break
        if ans_index == -1:
            fake_answer = fake_answer.split('。')[0]
            for index, x in enumerate(paras):
                if fake_answer in x or x in fake_answer:
                    ans_index = index
                    break

        if ans_index == 0:
            positive = [ [paras[ans_index], paras[ans_index + 1]], [paras[ans_index], paras[ans_index + 1]] ]
        elif ans_index == len(paras)-1:
            positive = [ [paras[ans_index - 1], paras[ans_index]], [paras[ans_index - 1], paras[ans_index]] ]
        else:
            positive = [ [paras[ans_index - 1], paras[ans_index]], [paras[ans_index], paras[ans_index + 1]] ]
        return positive
    else:
        index = random.randint(1,len(paras)-2)
        negative = [ [paras[index], paras[index-1]], [paras[index+1], paras[index]], [paras[index+1], paras[index-1]] ]
        return negative

def get_data(path, pickle_path):

    ans = []
    data = pd.read_csv(args.train_data, sep='\t')
    ques2ans = {}
    for question, answer in zip(data['question'], data['answer']):
        ques2ans[question] = answer

    f1 = F1()
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load('data/spiece.model')
    
    tokenizer = XLNetTokenizer()
    print("len(tokenizer.vocab): ", len(tokenizer.vocab))

    train_data, dev_data = [], []
    data_num = {'noans':0, 'hasans':0}
    index_error = 0

    with open(path, 'r', encoding='utf-8') as f:
        for step, line in enumerate(tqdm(f.readlines())):

            sample = json.loads(line)
            question = sample['question']
            # print(encode_pieces(sp_model, question))
            for doc in sample['docs']:

                content, ans_dict = doc['content'], doc['ans_dict']
                drop_sc = random.random()
                if drop_sc < 0.88 and ans_dict['is_impossible']:
                    continue
                tokens = content  # tokenizer._tokenize(content)
                ques_len = len(encode_pieces(sp_model, question))

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                ques_ids = input_ids[:ques_len+1]

                tokentype_ids = [1] * len(input_ids)
                tokentype_ids[:ques_len + 1] = [0] * (ques_len + 1)
                assert len(tokentype_ids) == len(input_ids)

                gt_score = ans_dict['answers'][0][3]
                if ans_dict['answers'][0][3] < 0.9 and ans_dict['is_impossible'] == False:
                    continue

                if ans_dict['is_impossible']:
                    data_num['noans'] += 1
                    start_position, end_position = 0, 0
                    verify_ids = [0] * len(input_ids)
                    verify_ids[0] = 1
                    cls_label = 0
                    gt_score = 0.0
                    ground_truth = ['']
                    context_cls = gen_context(tokens[ques_len+2:], fake_answer=None, positive_or_negative=False)
                    context_cls = context_cls[random.randint(0,2)]

                else:
                    cls_label = 1
                    data_num['hasans'] += 1
                    verify_ids = [0] * len(input_ids)

                    for idx in range(len(ans_dict['answers'])):
                        assert len(ans_dict['answers']) == 1
                        start_position, end_position = ans_dict['answers'][idx][1][0], ans_dict['answers'][idx][1][1]

                        if end_position >= len(input_ids):
                            print("##### end_position error")
                            continue
                        fake_answer = ''.join(tokenizer.convert_ids_to_tokens(input_ids[start_position:end_position + 1]))
                        for i in range(end_position + 1 - start_position):
                            verify_ids[i + start_position] = 1
                        ground_truth = ans_dict['answers'][0][2]
                        ground_truth = [ground_truth]
                        context_cls = gen_context(tokens[ques_len+2:], fake_answer=fake_answer, positive_or_negative=True, other=ans_dict['answers'][0][2])
                        if random.random() > 0.7:
                            context_cls = context_cls[1]
                        else:
                            context_cls = context_cls[0]
                        # if ans_dict['answers'][idx][3] < 0.95:
                        #     print(fake_answer)
                        #     print(ans_dict['answers'][0][2])
                        #     print()
                        ans.append(''.join(tokenizer.convert_ids_to_tokens(input_ids[start_position:end_position+1])))
                        # ans.append(ans_dict['answers'][0][2])
                        if step < 0:
                            print(ans_dict['answers'][idx][3])
                            print('ques:', ''.join(tokenizer.convert_ids_to_tokens(ques_ids)))
                            print('real:', ans_dict['answers'][0][2])
                            print('fake:', fake_answer)
                            print(''.join(tokenizer.convert_ids_to_tokens(input_ids)))
                            print(len(input_ids), len(''.join(tokenizer.convert_ids_to_tokens(input_ids))))
                            print()

                    if len(input_ids) > 512:
                        print(len(input_ids))
                        print("##### input_ids error")
                        continue
                # if context_cls == 0:
                #     continue
                # context_cls = ['<cls>']+ encode_pieces(sp_model, context_cls[0]) + ['<sep>'] + encode_pieces(sp_model, context_cls[1])
                # context_cls = tokenizer.convert_tokens_to_ids(context_cls)
                # print('context_cls：', ''.join(tokenizer.convert_ids_to_tokens(context_cls)))
                # print(cls_label)
                # print()
                train_data.append(
                    {"input_ids": input_ids,
                     # "context_cls": context_cls,
                     "tokentype_ids":tokentype_ids,
                     "verify_ids": verify_ids,
                     "cls_label": cls_label,
                     "start": start_position,
                     "end": end_position,
                     "gt_score":gt_score,
                     "ques_len":ques_len,
                     "ground_truth":ground_truth})

    print(data_num)
    print('ans_num: ', len(list(set(ans))))
    print("len(train_data): ", len(train_data))  # len(train_data): 456641, <512:len(train_data): 455354

    with open(pickle_path, 'w', encoding="utf-8") as fout:
        for feature in train_data:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')

class Dureader():
    def __init__(self):

        self.train_data = []
        with open(args.id_train, 'r', encoding="utf-8") as f:
            for step, line in enumerate(tqdm(f.readlines())):
                sample = json.loads(line)
                self.train_data.append([len(sample['input_ids']), sample, len(sample['context_cls'])])

        self.train_data.sort(key=lambda x:x[0], reverse=False)

        self.batch_data = []
        for batch in self.train_dataloader():
            input_ids, start, end, verify, gt_score, cls_label, ques_len = \
                batch['input_ids'], batch['start'], batch['end'], batch['verify_ids'], batch['gt_score'], batch['cls_label'], batch['ques_len']

            input_ids, start, end, verify, gt_score, cls_label, ques_len = \
                torch.tensor(input_ids), torch.tensor(start), torch.tensor(end), torch.tensor(verify), torch.tensor(gt_score), torch.tensor(cls_label), torch.tensor(ques_len)

            batch['input_ids'], batch['start'], batch['end'], batch['verify_ids'], batch['gt_score'], batch['cls_label'], batch['ques_len'] = \
                input_ids, start, end, verify, gt_score, cls_label, ques_len

            tokentype_ids = torch.tensor(batch['tokentype_ids'])
            batch['tokentype_ids'] = tokentype_ids
            # print([len(x) for x in batch['context_cls']])
            # context_cls = torch.tensor(batch['context_cls'])
            # batch['context_cls'] = context_cls

            self.batch_data.append(batch)

    def get_batch(self):
        random.shuffle(self.batch_data)
        return self.batch_data

    def train_dataloader(self):
        texts = []
        for sample in self.train_data:
            texts.append(sample)
            if len(texts) == args.batch_size:
                max_len = texts[-1][0]
                max_context_len = max([x[2] for x in texts])
                pad_texts = {"context_cls":[],"ground_truth":'',"tokentype_ids":[], "input_ids": [],"verify_ids": [],"start": [],"end": [],"gt_score":[], "cls_label":[], "ques_len":[]}
                for i in texts:
                    i[1]['input_ids'] += [args.pad_id] * (max_len-i[0])
                    i[1]['verify_ids'] += [args.pad_id] * (max_len - i[0])
                    i[1]['tokentype_ids'] += [args.pad_id] * (max_len - i[0])
                    # i[1]['context_cls'] += [args.pad_id] * (max_context_len - len(i[1]['context_cls']))
                    # i[1]['start'] += [args.pad_id] * (max_len - i[0])
                    # i[1]['end'] += [args.pad_id] * (max_len - i[0])

                    pad_texts['ground_truth'] = i[1]['ground_truth']
                    # pad_texts['context_cls'].append(i[1]['context_cls'])
                    pad_texts['tokentype_ids'].append(i[1]['tokentype_ids'])
                    pad_texts['input_ids'].append(i[1]['input_ids'])
                    pad_texts['verify_ids'].append(i[1]['verify_ids'])
                    pad_texts['start'].append(i[1]['start'])
                    pad_texts['end'].append(i[1]['end'])
                    pad_texts['gt_score'].append(i[1]['gt_score'])
                    pad_texts['cls_label'].append(i[1]['cls_label'])
                    pad_texts['ques_len'].append(i[1]['ques_len'])

                yield pad_texts
                texts = []

if __name__ == "__main__":

    get_data(path=args.process_split_train_data, pickle_path=args.id_train)

    loader = Dureader()
    # for _ in range(1):
    #     for batch in loader.get_batch():
    #         print(batch['ground_truth'])

        # exit()
        # print(len(batch['input_ids'][0]))


