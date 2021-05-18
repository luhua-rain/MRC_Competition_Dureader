import json
import args
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from tokenization import BertTokenizer

random.seed(args.seed)

def read_squad_examples(zhidao_input_file, search_input_file, is_training=True):
    total, error = 0, 0
    examples = []

    with open(search_input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):

            source = json.loads(line.strip())
            if (len(source['answer_spans']) == 0):
                continue
            if source['answers'] == []:
                continue
            if (source['match_scores'][0] < 0.8):
                continue
            if (source['answer_spans'][0][1] > args.max_seq_length):
                continue

            docs_index = source['answer_docs'][0]

            start_id = source['answer_spans'][0][0]
            end_id = source['answer_spans'][0][1] + 1          ## !!!!!
            question_type = source['question_type']

            passages = []
            try:
                answer_passage_idx = source['documents'][docs_index]['most_related_para']
            except:
                continue

            doc_tokens = source['documents'][docs_index]['segmented_paragraphs'][answer_passage_idx]
            ques_len = len(source['documents'][docs_index]['segmented_title']) + 1
            doc_tokens = doc_tokens[ques_len:]
            start_id , end_id = start_id -  ques_len, end_id - ques_len

            if start_id >= end_id or end_id > len(doc_tokens) or start_id >= len(doc_tokens):
                continue
 
            new_doc_tokens = ""
            for idx, token in enumerate(doc_tokens):
                if idx == start_id:
                    new_start_id = len(new_doc_tokens)
                    break
                new_doc_tokens = new_doc_tokens + token

            new_doc_tokens = "".join(doc_tokens)
            new_end_id = new_start_id + len(source['fake_answers'][0])            

            if source['fake_answers'][0] != "".join(new_doc_tokens[new_start_id:new_end_id]):
                continue

            if is_training:
                new_end_id = new_end_id - 1
                example = {
                        "qas_id":source['question_id'],
                        "question_text":source['question'].strip(),
                        "question_type":question_type,
                        "doc_tokens":new_doc_tokens.strip(),
                        "start_position":new_start_id,
                        "end_position":new_end_id }      

                examples.append(example)
    with open(zhidao_input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):

            source = json.loads(line.strip())
            if (len(source['answer_spans']) == 0):
                continue
            if source['answers'] == []:
                continue
            if (source['match_scores'][0] < 0.8):
                continue
            if (source['answer_spans'][0][1] > args.max_seq_length):
                continue
            docs_index = source['answer_docs'][0]

            start_id = source['answer_spans'][0][0]
            end_id = source['answer_spans'][0][1] + 1          ## !!!!!
            question_type = source['question_type']

            passages = []
            try:
                answer_passage_idx = source['documents'][docs_index]['most_related_para']
            except:
                continue

            doc_tokens = source['documents'][docs_index]['segmented_paragraphs'][answer_passage_idx]

            ques_len = len(source['documents'][docs_index]['segmented_title']) + 1
            doc_tokens = doc_tokens[ques_len:]
            start_id , end_id = start_id -  ques_len, end_id - ques_len

            if start_id >= end_id or end_id > len(doc_tokens) or start_id >= len(doc_tokens):
                continue

            new_doc_tokens = ""
            for idx, token in enumerate(doc_tokens):
                if idx == start_id:
                    new_start_id = len(new_doc_tokens)
                    break
                new_doc_tokens = new_doc_tokens + token

            new_doc_tokens = "".join(doc_tokens)
            new_end_id = new_start_id + len(source['fake_answers'][0])

            if source['fake_answers'][0] != "".join(new_doc_tokens[new_start_id:new_end_id]):
                continue

            if is_training:
                new_end_id = new_end_id - 1
                example = {
                        "qas_id":source['question_id'],
                        "question_text":source['question'].strip(),
                        "question_type":question_type,
                        "doc_tokens":new_doc_tokens.strip(),
                        "start_position":new_start_id,
                        "end_position":new_end_id }

                examples.append(example)

    print("len(examples):",len(examples))
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):

    features = []

    for example in tqdm(examples):
        query_tokens = list(example['question_text'])
        question_type = example['question_type']    

        doc_tokens = example['doc_tokens']
        doc_tokens = doc_tokens.replace(u"“", u"\"")
        doc_tokens = doc_tokens.replace(u"”", u"\"")
        start_position = example['start_position']
        end_position = example['end_position']

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        start_position = start_position + 1
        end_position = end_position + 1

        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
            start_position = start_position + 1
            end_position = end_position + 1

        tokens.append("[SEP]")
        segment_ids.append(0)
        start_position = start_position + 1
        end_position = end_position + 1

        for i in doc_tokens:
            tokens.append(i)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        if end_position >= max_seq_length:
            continue

        if len(tokens) > max_seq_length:
            tokens[max_seq_length-1] = "[SEP]"
            input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])      ## !!! SEP
            segment_ids = segment_ids[:max_seq_length]
        else:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        assert len(input_ids) == len(segment_ids)

        features.append(
                        {"input_ids":input_ids,
                         "input_mask":input_mask,
                         "segment_ids":segment_ids,
                         "start_position":start_position,
                         "end_position":end_position })

    with open("./train.data", 'w', encoding="utf-8") as fout:
        for feature in features:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')
    print("len(features):",len(features))
    return features

if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    # 生成训练数据， train.data
    examples = read_squad_examples(zhidao_input_file=args.zhidao_input_file,
                                   search_input_file=args.search_input_file)
    features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
                                            max_seq_length=args.max_seq_length, max_query_length=args.max_query_length)

    # 生成验证数据， dev.data。记得注释掉生成训练数据的代码，并在196行将train.data改为dev.data
   # examples = read_squad_examples(zhidao_input_file=args.dev_zhidao_input_file,
   #                                search_input_file=args.dev_search_input_file)
   # features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
   #                                         max_seq_length=args.max_seq_length, max_query_length=args.max_query_length)
