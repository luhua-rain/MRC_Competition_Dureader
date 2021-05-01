# -*- coding: utf-8 -*-
import args
import pandas as pd
from tqdm import tqdm
from function import read_corpus, para_recall, encode_pieces
import pickle
import re
import json
from similarity import get_seg
from collections import Counter
import sentencepiece as spm

def normalized(norm_s):
    norm_s = norm_s.replace(u"，", u",")
    norm_s = norm_s.replace(u"。", u".")
    norm_s = norm_s.replace(u"！", u"!")
    norm_s = norm_s.replace(u"？", u"?")
    norm_s = norm_s.replace(u"；", u";")
    norm_s = norm_s.replace(u"（", u"(").replace(u"）", u")")
    norm_s = norm_s.replace(u"【", u"[").replace(u"】", u"]")
    norm_s = norm_s.replace(u"“", u"\"").replace(u"”", u"\"")
    return norm_s

def unk():

    a, b = 0, 0
    corpus = read_corpus()
    print(corpus['72a3eb8cada539ab9583e5ba0652b04b'])
    exit()
    data = pd.read_csv(args.train_data, sep = '\t')
    data = [{'docid':docid, 'question':question, 'answer':answer, 'text':corpus[docid]}
            for id, docid, question, answer in zip(data['id'], data['docid'], data['question'], data['answer'])]

    test_data = pd.read_csv(args.test_data, sep='\t')
    test_data = [{'qid': qid, 'question': question}
                for qid, question in zip(test_data['id'], test_data['question'])]

    test_question = [i['question'] for i in test_data]
    train_question = []
    question2id = {}
    for i in tqdm(data):
        answer, question, text, docid = i['answer'], i['question'], i['text'], i['docid']
        question2id[question] = docid
        train_question.append(question)
        # print(text)
        # print(question+'  :  '+answer)
        # print()
        if len(text.split(answer)) > 2:
            # print(text)
            # print(len(text.split(answer)), answer)
            a+= 1
            # print()
        if len(text.split(answer)) < 2:
            b += 1
            print(text)
            print(answer)
            print()

    print('no-ans:',b)
    print('muti-ans:', a)
# unk()

def analysis_test():

    corpus = read_corpus()

    data = pd.read_csv(args.train_data, sep='\t')
    data = [{'docid': docid, 'question': question, 'answer': answer, 'text': corpus[docid]}
            for id, docid, question, answer in zip(data['id'], data['docid'], data['question'], data['answer'])]

    train_question = []
    question2id = {}
    for i in data:
        answer, question, text, docid = i['answer'], i['question'], i['text'], i['docid']
        question2id[question] = docid
        train_question.append(question)

    data = pd.read_csv('./result/0.6881.csv', sep = '\t')
    predict_data = [{'id':id, 'answer':answer, 'docid':docid}
            for id, docid, answer in zip(data['id'], data['docid'], data['answer'])]

    data = pd.read_csv(args.test_data, sep='\t')
    id2question = {}
    for qid, question in zip(data['id'], data['question']):
        id2question[qid] = question
    corpus = read_corpus()
    import json
    with open('predict.json', 'w', encoding='utf-8') as f:

        for i in predict_data:
            ques = id2question[i['id']]
            # similar_q, score = para_recall(ques, train_question, max_para_num=5)
            # print(similar_q[:3])
            print('ques:', ques)
            # print('ans:', i['answer'])
            # print('doc:', corpus.get(i['docid'], 'none'))
            # print()
            f.write(json.dumps({'question':ques, 'predict':i['answer']}, ensure_ascii=False) + '\n')
    # print('\"')
    # print('“')
# analysis_test()

def top_para():
    # 计算单文档抽取关键句子后的答案保持率
    correct, total = 0, 0
    data = pd.read_csv(args.train_data, sep='\t')
    data = [{'id': id, 'docid': docid, 'question': question, 'answer': answer}
            for id, docid, question, answer in zip(data['id'], data['docid'], data['question'], data['answer'])]
    content_data = read_corpus()

    for step, i in enumerate(data):
        answer, question, docid = i['answer'], i['question'], i['docid']
        text = content_data[docid]
        if answer in text:
            if len(text) > 0:
                text = re.split('[\n\t]', text)
                # paras, _ = para_recall(question, text, max_para_num=30, sort=False) # 极限0.969

                text = '。'.join(text)[:512] # 3072:9919, 2048：9715，1536：0.9361， 1024：0.8578， 512：0.5854
            if answer in text:
                correct += 1
            total += 1
            if step % 200 == 0:
                print(float(correct) / total)

    print(float(correct)/total)
    print(correct, total)
# top_para()

def topk_para():
    # 计算召回文档抽取关键句子后的答案保持率
    with open('./data/similarity/id2index.pkl', 'rb') as f:
        id2index = pickle.load(f)
    with open('./data/similarity/bm25_train_top300.pkl', 'rb') as f:
        topk = pickle.load(f)

    correct, total = 0, 0
    data = pd.read_csv(args.train_data, sep='\t')
    del data['id'], data['docid']
    print(data)
    exit()
    data = [{'id': id, 'docid': docid, 'question': question, 'answer': answer}
            for id, docid, question, answer in zip(data['id'], data['docid'], data['question'], data['answer'])]

    content_data = read_corpus()
    index_content_data = [_ for _ in range(len(content_data))]
    for docid, text in content_data.items():
        index_content_data[id2index[docid]] = text

    correct, total = 0, 0
    for step, i in enumerate(data):
        answer, question, docid, top = i['answer'], i['question'], i['docid'], topk[step]
        text = content_data[docid]
        top = top[:5]

        candidate_text = [index_content_data[index] for index in top]
        merge_para = []
        main_para = []
        for num, text in enumerate(candidate_text):
            if num > 5:
                text = text.split('。')
                if len(text) > 2:
                    merge_para.extend(
                        [text[l - 1] + '。' + text[l] + '。' + text[l + 1] for l in range(1, len(text) - 1, 2)])
                else:
                    merge_para.extend(text)
            else:
                main_para.extend([text[l - 1] + '。' + text[l] + '。' + text[l + 1] for l in range(1, len(text) - 1, 2)])

        # paras, _ = para_recall(question, merge_para, max_para_num=1, sort=False)
        main_para, _ = para_recall(question, main_para, max_para_num=20000, sort=False)
        temp = '。'.join(main_para).split('。')
        # print(len('。'.join(main_para)))
        # print(len('。'.join(paras)))
        a = []
        for t in temp:
            if t not in a:
                a.append(t)
        # print(len(''.join(main_para+paras)))

        if answer in '。'.join(main_para):
            correct += 1
        total += 1
        if step % 200 == 0:
            print(float(correct) / total)

    print(correct, total)

topk_para()

def noise():
    from find_noise import DataFilter, cut_sent

    filter = DataFilter()
    correct, total = 0, 0
    data = pd.read_csv(args.train_data, sep='\t')
    data = [{'id': id, 'docid': docid, 'question': question, 'answer': answer}
            for id, docid, question, answer in zip(data['id'], data['docid'], data['question'], data['answer'])]
    content_data = read_corpus()
    lenth = 0

    for step, i in enumerate(data):
        answer, question, docid = i['answer'], i['question'], i['docid']
        text = content_data[docid]
        if answer in text:
            filter_paras = []
            for p in cut_sent(text):
                if True:
                    filter_paras.append(p)
                else:
                    if not p in filter.noise_paras:
                        filter_p = p
                        temp = []
                        for tem in filter_p.split(' '):
                            if not tem in filter.noise_words:
                                temp.append(tem)

                        filter_paras.append(' '.join(temp))
                # else:
                #     print(p)
            text = ''.join(filter_paras)
            text = text[:2048+1024]
            lenth += len(text)
            if answer in text:
                correct += 1
            total += 1
            if step % 200 == 0:
                print(float(correct) / total)
    print(lenth/step)

# noise()

def candidate_vote():
    from function import RougeL
    import json
    rouge_L = RougeL(beta=1)
    score = 0
    with open('./data/dev_candidate_ans.json', 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            sample = json.loads(line)
            candidate_ans, rel, question = sample['candidate_ans'], sample['rel'], sample['question']
            single_ans = candidate_ans[0]
            top14_ans = sorted(candidate_ans[1:], key=lambda x : x[1], reverse=True)
            print('ques', question)
            print('rel:', rel)
            print('single:', single_ans)

            # for i in range(5):
            #     print(top14_ans[i][0], rouge_L.get_rouge_L(top14_ans[i][0], rel))
            choose = {}
            for i in top14_ans:
                choose[str(i[0])] = choose.get(str(i[0]), 0) + i[1]
            choose = sorted(choose.items(), key=lambda x : x[1], reverse=True)

            score += rouge_L.get_rouge_L(choose[0][0], rel)
            print()
    print(score/500)
# candidate_vote()

def similary_question():
    test_data = pd.read_csv(args.test_data, sep='\t')
    test_data = [{'qid': qid, 'question': question}
                 for qid, question in zip(test_data['id'], test_data['question'])]
    data = pd.read_csv(args.train_data, sep='\t')
    data = [{'docid': docid, 'question': question, 'answer': answer}
            for id, docid, question, answer in zip(data['id'], data['docid'], data['question'], data['answer'])]
    train_question = [n['question'] for n in data]

    for i in test_data:
        ques = i['question']
        similary, _ = para_recall(ques, train_question, max_para_num=5)
        print(similary)
        print(ques)
        print()
# similary_question()

def calculate_corpus():
    # 统计训练集中出现的总文档数目
    data = pd.read_csv(args.train_data, sep='\t')
    data = [{'docid': docid, 'question': question, 'answer': answer}
            for id, docid, question, answer in zip(data['id'], data['docid'], data['question'], data['answer'])]
    corpus = read_corpus()

    all_corpus = [n for i, n in corpus.items()]
    a = 0
    num = []
    for i in tqdm(data):
        question, answer = i['question'], i['answer']
        for idx, n in enumerate(all_corpus):
            if answer in n:
                num.append(idx)
                a+=1
    print(a/len(data))
    print(len(set(num)))
    print(len(all_corpus))
# calculate_corpus()
def badcase():
    a = 0
    import json
    with open('./look_data.json', 'r', encoding='utf-8') as f:
        for sample in tqdm(f.readlines()):
            sample = json.loads(sample)
            model_choose_doc, real_doc, score, question, predict_ans, real_ans = \
                sample['model_choose_doc'], sample['real_doc'], sample['rouge-L'], sample['question'], sample['predict_ans'], sample['real_ans']
            is_docid_in_top15 = sample['is_docid_in_top15']
            print('is_docid_in_top15:',is_docid_in_top15)
            if is_docid_in_top15:
                print(sample['top15_docs'].index(real_doc))
            else:
                a += 1
            print('model_choose_index:',sample['top15_docs'].index(model_choose_doc))
            print('question:', question)
            print('predict_ans:', predict_ans)
            print('real_ans:', real_ans)
            print('model choose:', model_choose_doc)
            print('real doc:', real_doc)
            print(score)
            print()
    print(a)

# badcase()
def get_question():
    a = 0
    seg = get_seg('pkuseg')
    data = pd.read_csv(args.train_data, sep='\t')
    data = [{'docid': docid, 'question': question, 'answer': answer}
            for id, docid, question, answer in zip(data['id'], data['docid'], data['question'], data['answer'])]
    train_question = [n['question'] for n in data]
    # ans = [n['answer'] for n in data[:-500]]
    # print(len(set(ans)))
    # exit()
    for q in train_question:
        # q = q.replace('的','')
        print(q)
    #     q = seg(q)
    #     common = Counter(q)
    #     if 2 in list(common.values()):
    #         a += 1
    #         print(common)
    #         print(''.join(q))
    #         print()
    # print(a)

    with open('question_1.json', 'w', encoding="utf-8") as fout:
        for feature in train_question[:2500]:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')
    with open('question_2.json', 'w', encoding="utf-8") as fout:
        for feature in train_question[2500:]:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')

# get_question()
# keyword = ['中关村管委会', '中医局', '中央后勤保障部', '中央政治局常委会', '交通委', '交通运输部', '人力社保局', '人力资源', '人力资源和社会保障局', '人力资源社会保障部', '人力资源部', '人民政府', '人民政府办公室', '人民检察院', '人社厅', '人社部', '人防办', '住建委', '住房公积金管理中心', '住房城乡建设委', '体育局', '侨联', '信息化部', '信访办', '公园管理中心', '公安局', '公安部', '农业农村局', '农业农村部', '农机监理所', '农村农业部', '办公厅', '医保局', '医疗保险协会', '卫健委', '卫生健康委', '卫生健康部', '友协', '发展改革委', '发改委', '台办', '司法局', '商务局', '商务部', '团市委', '园林绿化局', '国务院', '国家体育总局', '国家发展改革委', '国家发改委', '国家林草局', '国家税务总局', '国家能源局', '国家药品监管局', '国家铁路局', '国税局', '国资委', '地方志编委会', '地方金融监管局', '城市管理委', '城管执法局', '外汇管理局', '天安门地区管委会', '妇联', '审计局', '审计署', '密码管理局', '工业和信息化部', '工信部', '市场监督管理局', '市场监管局', '市场监管总局', '市委组织部', '市教委', '广播电视局', '应急管理局', '总工会', '扶贫办', '投资促进服务中心', '政务服务中心', '政务服务大厅', '政务服务管理局', '政府侨办', '政府外办', '教委', '教育局', '教育督导委员会', '教育部', '文化和旅游局', '文化执法总队', '文物局', '服贸司', '林草局', '档案局', '残联', '民政局', '民政部', '民族宗教委', '气象局', '气象部', '水利部', '水务局', '海关', '海关总署', '烟草局', '煤矿安监局', '生态环境局', '生态环境部', '电子税务局', '省委办公厅', '省政府办公厅', '知识产权局', '社会保障部', '社科联', '科协', '科委', '税务局', '税务总局', '粮食和物资储备局', '粮食物资局', '红十字会', '经济信息化局', '经济技术开发区管委会', '统计局', '编办', '能源局', '自然资源部', '药监局', '规划自然资源委', '财政局', '财政部', '资源部', '退役军人事务局', '邮政局', '邮政管理局', '重大项目办', '金融监督管理局', '银保监局', '首都文明办', '高级人民法院']
#
# print(len(keyword))

def _test_topk():

    correct, total = 0, 0
    with open('./data/similarity/id2index.pkl', 'rb') as f:
        id2index = pickle.load(f)
    with open('./data/similarity/bm25_train_top300.pkl', 'rb') as f:
        topk = pickle.load(f)
    print('500:', len(topk))
    print('top:', len(topk[list(topk.keys())[0]]))

    data = pd.read_csv(args.train_data, sep='\t')
    data = [{'docid': docid, 'question': question, 'answer': answer, 'id':id}
            for id, docid, question, answer in zip(data['id'], data['docid'], data['question'], data['answer'])]
    for n in range(1,16):
        correct, total = 0, 0
        for i in data[-500:]:
            if id2index[i['docid']] in topk[i['id']][:n]:
                correct += 1
            total += 1

        print(n, float(correct)/total)
# _test_topk()


def trans():
    with open('./data/similarity/bm25_test_2gram_top30.dict', 'rb') as f:
        topk = pickle.load(f)
    with open('./data/similarity/index2id.pkl', 'rb') as f:
        index2id = pickle.load(f)

    for key, _ in topk.items():
        print(topk[key])
        print(key)
        topk[key] = [index2id[i] for i in topk[key] ]
        print(topk[key])
        print()

    with open('bm25_test_2gram_top30.dict', 'wb') as f:
        pickle.dump(topk, f)

# trans()

with open('bm25_test_2gram_top30.dict', 'rb') as f:
    topk = pickle.load(f)
# print(topk)
