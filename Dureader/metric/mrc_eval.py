# coding:utf8
"""
This module computes evaluation metrics for DuReader dataset.
"""

import argparse
import itertools
import json
import sys
import importlib
import zipfile

from collections import Counter
from bleu import BLEUWithBonus
from rouge import RougeL

EMPTY = ''
YESNO_LABELS = set(['Yes', 'No', 'Depends'])

def normalize(s):
    """
    Normalize strings to space joined chars.
    Args:
        s: a list of strings.
    Returns:
        A list of normalized strings.
    """
    if not s:
        return s
    normalized = []
    for ss in s:
        tokens = [c for c in list(ss) if len(c.strip()) != 0]
        norm_s = ''.join(tokens)
        norm_s = norm_s.replace(u"，", u",")
        norm_s = norm_s.replace(u"。", u".")
        norm_s = norm_s.replace(u"！", u"!")
        norm_s = norm_s.replace(u"？", u"?")
        norm_s = norm_s.replace(u"；", u";")
        norm_s = norm_s.replace(u"（", u"(").replace(u"）", u")")
        norm_s = norm_s.replace(u"【", u"[").replace(u"】", u"]")
        norm_s = norm_s.replace(u"“", u"\"").replace(u"“", u"\"")
        normalized.append(norm_s)
    return normalized


def data_check(obj):
    """
    Check data.

    Raises:
        Raises AssertionError when data is not legal.
    """
    assert 'question_id' in obj, "Missing 'question_id' field."
    #assert 'question_type' in obj, \
    #        "Missing 'question_type' field. question_id: {}".format(obj['question_type'])

    #assert 'yesno_answers' in obj, \
    #        "Missing 'yesno_answers' field. question_id: {}".format(obj['question_id'])
    if "yesno_answers" in obj:
        assert isinstance(obj['yesno_answers'], list), \
            r"""'yesno_answers' field must be a list, if the 'question_type' is not
            'YES_NO', then this field should be an empty list.
            question_id: {}""".format(obj['question_id'])
    else:
        obj["yesno_answers"] = []

    if "entity_answers" not in obj:
        obj["entity_answers"] = []


def read_file(file_name, is_ref=False):
    """
    Read predict answers or reference answers from file.

    Args:
        file_name: the name of the file containing predict result or reference
                   result.

    Returns:
        A dictionary mapping question_id to the result information. The result
        information itself is also a dictionary with has four keys:
        - question_type: type of the query.
        - yesno_answers: A list of yesno answers corresponding to 'answers'.
        - answers: A list of predicted answers.
        - entity_answers: A list, each element is also a list containing the entities
                    tagged out from the corresponding answer string.
    """
    def _open(file_name, mode, zip_obj=None):
        if zip_obj is not None:
            return zip_obj.open(file_name, mode)
        return open(file_name, mode, encoding='utf-8')

    results = {}
    if is_ref:
#        keys = ['source', 'answers', 'yesno_answers', 'entity_answers', 'question_type']
        keys = ['answers', 'yesno_answers', 'entity_answers', 'question_type']
    else:
        keys = ['answers', 'yesno_answers', 'entity_answers', 'question_type'] 
    try:
        zf = zipfile.ZipFile(file_name, 'r') if file_name.endswith('.zip') else None
    except:
        zf = None
    file_list = [file_name] if zf is None else zf.namelist()

    for fn in file_list:
        line_num = 0
        for line in _open(fn, 'r', zip_obj=zf):
            try:
                line_num += 1
                obj = json.loads(line.strip())
            except ValueError:
                #raise ValueError("Every line of data should be legal json, in line %s" % str(line_num))
                print >> sys.stderr, ValueError("Every line of data should be legal json, in line %s" % str(line_num))
                continue
            data_check(obj)
            qid = obj['question_id']
            assert qid not in results, "Duplicate question_id: {}".format(qid)
            results[qid] = {}
            for k in keys:
                if k == 'answers':
                    results[qid][k] = normalize(obj[k])
                else:
                    results[qid][k] = obj[k]
            if is_ref:
                for i, e in enumerate(results[qid]['entity_answers']):
                    results[qid]['entity_answers'][i] = normalize(e)
    return results

def main(args):
    err = None
    metrics = {}
    bleu4, rouge_l = 0.0, 0.0
    alpha = args.ab
    beta = args.ab
    bleu_eval = BLEUWithBonus(4, alpha=alpha, beta=beta)
    rouge_eval = RougeL(alpha=alpha, beta=beta, gamma=1.2)
    try:
        pred_result = read_file(args.pred_file)
        ref_result = read_file(args.ref_file, is_ref=True)
        for qid, results in ref_result.items():
            cand_result = pred_result.get(qid, {})
            #pred_answers = cand_result.get('answers', [EMPTY])[0]
            pred_answers = cand_result.get('answers', [])
            if not pred_answers:
                pred_answers = EMPTY
            else:
                pred_answers = pred_answers[0]
            pred_yn_label = None
            ref_entities = None
            ref_answers = results.get('answers', [])
            if not ref_answers:
                continue
            if results['question_type'] == 'ENTITY':
                ref_entities = set(
                        itertools.chain(*results.get('entity_answers', [[]])))
                with open('e.data','a', encoding='utf-8') as f:
                    f.write(str(ref_entities) + '\n')
                if not ref_entities:
                    ref_entities = None
            if results['question_type'] == 'YES_NO':
                cand_yesno = cand_result.get('yesno_answers', [])
                pred_yn_label = None if len(cand_yesno) == 0 \
                        else cand_yesno[0]
            bleu_eval.add_inst(
                    pred_answers,
                    ref_answers,
                    yn_label=pred_yn_label,
                    yn_ref=results['yesno_answers'],
                    entity_ref=ref_entities)
            rouge_eval.add_inst(
                    pred_answers,
                    ref_answers,
                    yn_label=pred_yn_label,
                    yn_ref=results['yesno_answers'],
                    entity_ref=ref_entities)
        bleu4 = bleu_eval.score()[-1]
        rouge_l = rouge_eval.score()
    except ValueError as ve:
        err = ve
    except AssertionError as ae:
        err = ae
    # too keep compatible to leaderboard evaluation.
    metrics['errorMsg'] = 'success' if err is None else err
    metrics['errorCode'] = 0 if err is None else 1
    metrics['data'] = [
            {'type': 'BOTH', 'name': 'ROUGE-L', 'value': round(rouge_l* 100, 2)},
            {'type': 'BOTH', 'name': 'BLEU-4', 'value': round(bleu4 * 100, 2)},
            ]
    print(json.dumps(metrics, ensure_ascii=False).encode('utf8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file', help='predict file')
    parser.add_argument('ref_file', help='reference file')
    parser.add_argument('task',
            help='task name, only to keep compatible with leaderboard eval')
    parser.add_argument('--ab', type=float, default=1.0,
            help='common value of alpha and beta')
    args = parser.parse_args()
    main(args)
