import collections
import csv

def merge(paths):
    answer_dict = collections.OrderedDict()
    model_csv = []
    for path in paths:
        fp = open(path, 'r')
        model_csv.append(fp.readlines())

    revise = 0
    for i in range(23209):
        vote_num = collections.OrderedDict()
        for model in model_csv:
            name = model[i].split(',')[0]
            pre = model[i].split(',')[1].strip()

            if pre in vote_num:
                vote_num[pre] += 1
            else:
                vote_num[pre] = 1

        vote_num = sorted(vote_num.items(), key=lambda x:x[1], reverse=True)
        if len(vote_num) > 1:
            print(vote_num, model_csv[0][i])
            revise += 1
        # print(vote_num)
        answer_dict[name] = int(vote_num[0][0])

    print(len(answer_dict))
    # print(answer_dict)
    print('revise: ', revise)

    with open('last_1.csv', 'w', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        for i, n in answer_dict.items():
            stu = [i, n]
            csv_write.writerow(stu)


if __name__ == "__main__":
    merge([ './distill_result_89.91.csv', './ernie_result_88.48.csv', './bertlstm.csv',
           './xlnetlstm.csv', './xlnetlinear.csv', './bert_result_88.52.csv'
           ])

    # merge(['./bert_result_88.52.csv', './ernie_result_88.48.csv',
    #        './bertlstm.csv', './roberta.csv'
    #        ])
    # bertlstm = 87.23/87.64
    # bert_result_88.47 = 88.47
    # xlnetlinear = 89.06/89.63
    # xlnetlstm = 89.11/89.66
    # xlnet_chr_89.18 = 89.18

    # vote = 90.11/90.58(bertlstm、xlnetlstm、xlnetlinear)
    # vote = 90.49(xlnetlstm、xlnet_chr_89.18、xlnetlinear)
    # vote = 90.58(bert_result_88.47、xlnetlinear、xlnet_chr_89.18)


