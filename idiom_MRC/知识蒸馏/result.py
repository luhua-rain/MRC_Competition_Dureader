from train import XlnetCloze, load_model, to_device
from data import getdataLoader
from parameters import *


# 生成每个句子对应的十个概率
def generate_prob(epoch=-1):
    xlnet_config = XLNetConfig.from_json_file(config.xlnet_config_root)
    model = XlnetCloze(xlnet_config)
    test_dataloader = getdataLoader(mode="test")
    load_model(epoch, model)
    model.to(config.device)
    model.eval()
    all_results = {}
    if config.n_gpu > 1:
        model = nn.DataParallel(model)
    for batch in tqdm(test_dataloader):
        input_ids, attention_mask, position, option_ids, tags = batch
        input_ids, attention_mask, position, option_ids, tags = to_device(
            input_ids, attention_mask, position, option_ids, tags
        )
        with torch.no_grad():
            batch_logits = model.forward(input_ids, attention_mask, position, option_ids, tags)
        for i, tag in enumerate(tags):
            logits = batch_logits[i].detach().cpu().numpy()
            prob = F.softmax(t.Tensor(logits))
            all_results["#idiom%06d#" % tag] = prob
    with open(config.prob_file, "w") as f:
        for each in all_results:
            f.write(each)
            for i in range(10):
                f.write(',' + str(all_results[each][i].item()))
            f.write("\n")
    with open(config.raw_result_file, "w") as f:
        for each in all_results:
            f.write(each)
            f.write("," + str(all_results[each].max(dim=0)[1].item()))
            f.write("\n")
    print("generate test result finished")


# 使用“测试时增强”来提高performance，无效！
def generate_multi_prob(epoch=-1, times=10):
    xlnet_config = XLNetConfig.from_json_file(config.xlnet_config_root)
    model = XlnetCloze(xlnet_config)
    test_dataloader = getdataLoader(mode="test")
    load_model(epoch, model)
    model.to(config.device)
    model.eval()
    all_results = {}
    if config.n_gpu > 1:
        model = nn.DataParallel(model)
    for i in trange(times):
        for batch in tqdm(test_dataloader):
            input_ids, attention_mask, position, option_ids, tags = batch
            input_ids, attention_mask, position, option_ids, tags = to_device(
                input_ids, attention_mask, position, option_ids, tags
            )
            with torch.no_grad():
                batch_logits = model.forward(input_ids, attention_mask, position, option_ids, tags)
            for i, tag in enumerate(tags):
                logits = batch_logits[i].detach().cpu().numpy()
                prob = F.softmax(t.Tensor(logits))
                if "#idiom%06d#" % tag in all_results:
                    all_results["#idiom%06d#" % tag] += prob
                else:
                    all_results["#idiom%06d#" % tag] = prob
    for k, v in all_results.items():
        all_results[k] = v / times
    with open(config.prob_file, "w") as f:
        for each in all_results:
            f.write(each)
            for i in range(10):
                f.write(',' + str(all_results[each][i].item()))
            f.write("\n")
    print("generate test result finished")


header_list = ['id', 'p0', 'p1', 'p2', 'p3', 'p4',
               'p5', 'p6', 'p7', 'p8', 'p9']
res = []
tmp = []
tmp_pos = []
max_prob = - 10000000


def get_group():
    testfile = open(config.data_root + "dev.txt", "r", encoding='utf-8')
    text = testfile.readlines()
    group = []
    for line in text:
        tags = re.findall("#idiom\d+#", line)
        group.append(tags)
    print("数据分组完毕")
    return group


# 使用beam_search得到最终结果，即后期处理
def beam_search(ids, probs, log_probs, ranks, order):
    global max_prob, tmp, res, tmp_pos
    if order == len(ids):
        now_prob = 0
        # print("start")
        for k in range(order):
            # print(probs[k][tmp_pos[k]])
            # now_prob += probs[k][tmp_pos[k]]
            now_prob += log_probs[k][tmp_pos[k]]
        # print(now_prob)
        # print("end")
        if now_prob > max_prob:
            max_prob = now_prob
            # print(now_prob, tmp)
            res = tmp.copy()
        return
    for i in range(5):
        now_rank = ranks[order][i]
        flag = True
        for j in range(order):
            if tmp[j] == now_rank:
                flag = False
        if not flag:
            continue
        tmp[order] = now_rank
        tmp_pos[order] = i
        beam_search(ids, probs, log_probs, ranks, order + 1)


def generate_result():
    global max_prob, tmp, res, tmp_pos
    groups = get_group()
    ave_prob = pd.read_csv(config.prob_file, header=None, names=header_list, sep=',',
                           index_col='id')
    reses = []
    for group in tqdm(groups):
        probs = []
        ids = []
        ranks = []
        log_probs = []
        now_prob = 0
        for id in group:
            ids.append(id)
            prob = list(ave_prob.loc[id, :])
            rank = list(np.argsort(-np.array(prob)))   # 输出索引
            # ipdb.set_trace()
            prob.sort(reverse=True)
            log_prob = list(map(lambda x: math.log(x, 10), prob))
            # ipdb.set_trace()
            probs.append(prob)
            ranks.append(rank)
            log_probs.append(log_prob)
        for i in range(len(ids)):
            now_prob += probs[i][0]
        assert len(ids) == len(probs)
        assert len(probs[0]) == 10
        res = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        tmp = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        tmp_pos = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        max_prob = - 10000000
        # print(probs, ranks, ids)
        beam_search(ids, probs, log_probs, ranks, 0)
        really_res = res[0:len(ids)]
        assert len(really_res) == len(set(really_res))  # 防止预测为同一个成语
        # print(probs,really_res)
        reses.append((ids, really_res))
        # ipdb.set_trace()
    assert len(reses) == len(groups)
    search_idx = open(config.result_file, "w")
    for id, res in reses:
        for i in range(len(id)):
            search_idx.write(id[i] + ',' + str(res[i]) + '\n')


if __name__ == '__main__':
    generate_prob(epoch=21)
    generate_result()
