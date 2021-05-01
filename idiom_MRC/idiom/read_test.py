from xlnet.tokenization_xlnet import XLNetTokenizer
from dataloader import get_answer, get_idiomDict
from bert.tokenization import BertTokenizer
from tqdm import tqdm
import pickle
import random
import torch
import json
import args
import re

def get_name():
    name = []
    idiom2id, id2idiom = get_idiomDict()

    with open(args.test_path, 'r', encoding='utf-8') as f:
        for step, line in enumerate(tqdm(f.readlines())):
            source = json.loads(line)

            candidates = [idiom2id[idiom] for idiom in source['candidates']]
            assert len(candidates) == 10

            for content in source['content']:

                content = "[CLS]" + content + "[SEP]"
                index = [m.start() for m in re.finditer("#idiom", content)]
                # {'#idiom000000#': 1445}
                name += [content[n:n + 13] for n in index]

    print("len(name): ", len(name))
    return name

def get_rule_testdata():

    idiom2id, id2idiom = get_idiomDict()
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    tokenizer = XLNetTokenizer()

    test_data = []
    batch_data = []
    with open(args.test_path, 'r', encoding='utf-8') as f:
        for step, line in enumerate(tqdm(f.readlines())):
            source = json.loads(line)
            candidates = [idiom2id[idiom] for idiom in source['candidates']]
            assert len(candidates) == 10

            for content in source['content']:
                index = [m.start() for m in re.finditer("#idiom", content)]
                pre_idiom = {content[n:n + 13]: 0 for n in index}

                for i in pre_idiom.keys():
                    content = content.replace(i, "¿")

                content = tokenizer._tokenize(content)
                while len(content) > 509:
                    drop = random.choice(range(len(content)))
                    if content[drop] != "¿":
                        del content[drop]

                input_ids = tokenizer.convert_tokens_to_ids(["<sep>"] + content + ["<sep>"] + ["<cls>"])

                index = []
                for n, num in enumerate(input_ids):
                    if num == 6:            # [MASK] : 103, xlnet:<mask>: 6
                        index.append(n)

                label_mask = [0] * len(input_ids)
                for idx, i in enumerate(index):
                    if tokenizer.convert_ids_to_tokens([input_ids[i]])[0] != '<mask>':
                        print('error')
                        exit()
                    label_mask[i] = 1

                if step < 30:
                    print(''.join(tokenizer.convert_ids_to_tokens(input_ids)))
                    print(''.join(content))
                    print()

                batch_data.append(
                        {"input_ids": input_ids,
                         "candidate": candidates,
                         "label_mask": label_mask  })

            test_data.append(batch_data)
            batch_data = []

    print("len(test_data): ", len(test_data))  # len(dev_data): 4601, <512: len(dev_data):  4591
    return test_data

if __name__ == "__main__":
    # idiom2id, id2idiom = get_idiomDict()
    # print(id2idiom, len(id2idiom))
    # get_name()
    get_rule_testdata()
