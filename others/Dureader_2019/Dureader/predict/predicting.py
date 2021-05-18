import json
import args
import torch
import pickle
import torch.nn as nn
from tqdm import tqdm

import predict_data
from tokenization import BertTokenizer
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForQuestionAnswering, BertConfig

def find_best_answer_for_passage(start_probs, end_probs, passage_len, question):
        best_start, best_end, max_prob = -1, -1, 0

        start_probs, end_probs  = start_probs.unsqueeze(0), end_probs.unsqueeze(0)
        prob_start, best_start = torch.max(start_probs, 1)
        prob_end, best_end = torch.max(end_probs, 1)
        num = 0
        while True:
            if num > 3:
                break
            if best_end >= best_start:
                break
            else:
                start_probs[0][best_start], end_probs[0][best_end] = 0.0, 0.0
                prob_start, best_start = torch.max(start_probs, 1)
                prob_end, best_end = torch.max(end_probs, 1)
            num += 1
        max_prob = prob_start * prob_end

        if best_start <= best_end:
            return (best_start, best_end), max_prob
        else:
            return (best_end, best_start), max_prob

def find_best_answer(sample, start_probs, end_probs, prior_scores=(0.44, 0.23, 0.15, 0.09, 0.07)):

        best_p_idx, best_span, best_score = None, None, 0

        for p_idx, passage in enumerate(sample['doc_tokens'][:args.max_para_num]):
    
            passage_len = min(args.max_seq_length, len(passage['doc_tokens']))
            answer_span, score = find_best_answer_for_passage(start_probs[p_idx], end_probs[p_idx], passage_len, sample['question_text'])

            score *= prior_scores[p_idx]

            answer = "p" + sample['question_text'] + "。" + sample['doc_tokens'][p_idx]['doc_tokens']
            answer = answer[answer_span[0]: answer_span[1]+1]

            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
   
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            para = "p" + sample['question_text'] + "。" + sample['doc_tokens'][best_p_idx]['doc_tokens']
            best_answer = ''.join(para[best_span[0]: best_span[1]+1])

        return best_answer, best_p_idx

def evaluate(model, result_file):

    with open(args.predict_example_files,'rb') as f:
        eval_examples = pickle.load(f)

    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        model.eval()
        pred_answers, ref_answers = [], []

        for step, example in enumerate(tqdm(eval_examples)):
            start_probs, end_probs = [], []
            question_text = example['question_text']
   
            for p_num, doc_tokens in enumerate(example['doc_tokens'][:args.max_para_num]):             
                (input_ids, input_mask, segment_ids) = \
                                         predict_data.predict_data(question_text, doc_tokens['doc_tokens'], tokenizer, args.max_seq_length, args.max_query_length)
               
                start_prob, end_prob = model(input_ids, segment_ids, attention_mask=input_mask)     # !!!!!!!!!!
                start_probs.append(start_prob.squeeze(0))
                end_probs.append(end_prob.squeeze(0))
                
            best_answer, docs_index = find_best_answer(example, start_probs, end_probs)

            pred_answers.append({'question_id': example['id'],
                                 'question':example['question_text'],
                                 'question_type': example['question_type'],
                                 'answers': [best_answer],
                                 'entity_answers': [[]],
                                 'yesno_answers': []})
            if 'answers' in example:
                ref_answers.append({'question_id': example['id'],
                                    'question_type': example['question_type'],
                                    'answers': example['answers'],
                                    'entity_answers': [[]],
                                    'yesno_answers': []})
        with open(result_file, 'w', encoding='utf-8') as fout:
            for pred_answer in pred_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
        with open("../metric/ref.json", 'w', encoding='utf-8') as fout:
            for pred_answer in ref_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

def eval_all():
   
    output_model_file = "../model_dir/best_model"
    output_config_file = "../model_dir/bert_config.json"  
    
    config = BertConfig(output_config_file)
    model = BertForQuestionAnswering(config)
    model.load_state_dict(torch.load(output_model_file)) #, map_location='cpu'))
    evaluate(model.cpu(),result_file="../metric/predicts.json")
   
eval_all()
