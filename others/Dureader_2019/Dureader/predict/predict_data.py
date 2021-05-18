import torch
import json

def predict_data(question_text, doc_tokens, tokenizer, max_seq_length, max_query_length):

    features = []
    query_tokens = list(question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tokens, segment_ids = [], []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)

    for i in doc_tokens:
            tokens.append(i)
            segment_ids.append(1)

    tokens.append("[SEP]")
    segment_ids.append(1)

    if len(tokens) > max_seq_length:
        tokens[max_seq_length-1] = "[SEP]"
        input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])      ## !!! SEP
        segment_ids = segment_ids[:max_seq_length]
    else:
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
   
    input_mask = [1] * len(input_ids)

    assert len(input_ids) == len(segment_ids)

    return (torch.LongTensor(input_ids).unsqueeze(0), torch.LongTensor(input_mask).unsqueeze(0), torch.LongTensor(segment_ids).unsqueeze(0))

