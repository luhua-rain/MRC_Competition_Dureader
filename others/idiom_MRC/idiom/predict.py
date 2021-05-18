from bert.modeling import BertForMaskedLM

from dataloader import get_idiomDict, get_answer
from read_test import get_testdata, get_name, get_rule_testdata
from tqdm import tqdm
import collections
import pandas as pd
import torch
import pickle
import args
import csv

def rule_predict():

    name = get_name()
    test_data = get_rule_testdata()

    model = torch.load(args.best_model_path, 'cpu')
    model = model.to(args.device)

    i_name = 0
    answer_dict = collections.OrderedDict()

    error = 0
    with torch.no_grad():
        model.eval()

        print("##### action #####")
        for examples in tqdm(test_data):
            active_logits = []
            for example in examples:

                input_ids, label_mask, candidate = torch.tensor(example['input_ids']).long(), torch.tensor(example['label_mask']).long(), torch.tensor(example['candidate']).long()
                input_ids, label_mask, candidate = input_ids.unsqueeze(0).to(args.device), label_mask.unsqueeze(0).to(args.device), candidate.unsqueeze(0).to(args.device)

                if input_ids.shape[1] > 512:
                    error += 1
                    num = (label_mask.view(-1) == 1).sum()
                    for i in range(num):
                        answer_dict[name[i_name]] = 0
                        i_name += 1
                else:
                    active = label_mask.view(-1) == 1
                    out = model(input_ids)
                    candidate_repeat = candidate.unsqueeze(1).repeat(1, out.size(1), 1)
                    out = torch.gather(out, dim=-1, index=candidate_repeat)
                    out = out.view(-1, 10)[active]
                    # print(out.shape) # (1,10)、(2,10)、(3, 10)、(4, 10)
                    active_logits.append(out.view(-1))

            active_logits = torch.cat(active_logits, dim=-1)
            length = int(active_logits.shape[0] / 10)
            label = [-1 for _ in range(length)]

            logits = active_logits.cpu().tolist()
            # print(logits)
            for n in range( length ):
                pre = logits.index(max(logits))
                label[divmod(pre, 10)[0]] = divmod(pre, 10)[-1]

                for i in range(10): # 将该向量置0
                    logits[i + 10*divmod(pre, 10)[0]] = -100
                for i in range( length ): # 将每个向量对应位置置0
                    logits[i*10 + divmod(pre, 10)[-1]] = -100

            print(label)
            print()
            for i in label:
                answer_dict[name[i_name]] = i
                i_name += 1

    print("error: ", error)
    assert len(answer_dict.keys()) == len(name)
    with open('result.csv', 'w', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        for i, n in answer_dict.items():
            stu = [i, n]
            csv_write.writerow(stu)

def debug():
    import json
    import re
    answerDict = get_answer()
    # print(answerDict)
    model = torch.load(args.best_model_path, 'cpu')
    model = model.to(args.device)
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    idiom2id, id2idiom = get_idiomDict()

    with torch.no_grad():
        model.eval()
        with open('./data/debug.json', 'r', encoding='utf-8') as f:
            for example in tqdm(f.readlines()):

                source = json.loads(example)
                index = [m.start() for m in re.finditer("#idiom", source['content'])]
                if len(index) > 1:
                    continue
                # print(index)
                # print(source['content'])
                pre_idiom = {source['content'][n:n + 13]: 0 for n in index}
                name = source['content'][index[0]:index[0] + 13]
                answer = int(answerDict[name])

                content = source['content']
                for i in pre_idiom.keys():
                    content = content.replace(i, "¿")
                input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + list(content) + ["[SEP]"])
                candidate = [idiom2id[idiom] for idiom in source['candidates']]

                index = []
                label_mask = [0] * len(input_ids)
                for step, num in enumerate(input_ids):
                    if num == 103:  # [MASK] : 103
                        index.append(step)

                for idx, i in enumerate(index):
                    label_mask[i] = 1

                input_ids, label_mask, candidate = torch.tensor(input_ids).long(), torch.tensor(label_mask).long(), torch.tensor(candidate).long()
                input_ids, label_mask, candidate = input_ids.unsqueeze(0).to(args.device), label_mask.unsqueeze(0).to(args.device), candidate.unsqueeze(0).to(args.device)

                out = model(input_ids)

                candidate_repeat = candidate.unsqueeze(1).repeat(1, out.size(1), 1)
                out = torch.gather(out, dim=-1, index=candidate_repeat)
                active = label_mask.view(-1) == 1

                _, predicted = torch.max(out.view(-1, 10), dim=-1)

                active_logits = predicted.view(-1)[active]
                active_logits = active_logits.cpu().tolist()
                # print(source['content'])
                # print(source['candidates'])
                # print(active_logits, answer)
                # print()
                if active_logits[0] != answer:
                    with open('error.json', 'a', encoding='utf-8') as f:
                        feature = {'content':source['content'],
                                   'pre_ans':[source['candidates'][active_logits[0]], source['candidates'][answer]]}                  
                        f.write(json.dumps(feature, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    rule_predict()
    # debug()

