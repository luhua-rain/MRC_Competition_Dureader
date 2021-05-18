from parameters import *

# 各种辅助函数

def gen_vocab():
    vocab = eval(open(config.idiom_vocab_root, mode="r").readline())
    idiom2index = {idiom: idx for idx, idiom in enumerate(vocab)}
    index2idiom = {v: k for k, v in idiom2index.items()}
    with open(config.data_root + "idiom2index", mode="wb") as f1:
        pickle.dump(idiom2index, f1)
    with open(config.data_root + "index2idiom", mode="wb") as f2:
        pickle.dump(index2idiom, f2)


def get_ansdict():
    ans_dict = {}
    with open(config.data_root + "train_answer.csv", "r") as f:
        for line in f:
            line = line.split(',')
            ans_dict[line[0]] = int(line[1])
    return ans_dict


def gen_split_data(mode="train"):
    if mode == "train":
        config.logger.info("处理训练数据中...")
        input_data = open(config.raw_train_data_root, mode="r")
        ans_dict = get_ansdict()
    elif mode == "test":
        config.logger.info("处理测试数据中...")
        input_data = open(config.raw_test_data_root, mode="r")

    examples = []

    for data in tqdm(input_data):
        data = eval(data)
        options = data['candidates']
        for context in data['content']:
            tags = re.findall("#idiom\d+#", context)
            for tag in tags:
                tmp_context = context
                for other_tag in tags:
                    if other_tag != tag:   # 将其他idiom变为 <unk>
                        tmp_context = tmp_context.replace(other_tag, "<unk>")
                if mode == "train":
                    label = ans_dict[tag]
                    examples.append({
                        "tag": tag,
                        "context": tmp_context,
                        "options": options,
                        "label": label
                    })
                elif mode == "test":
                    examples.append({
                        "tag": tag,
                        "context": tmp_context,
                        "options": options,
                    })
    json.dump(examples, open(config.data_root + "split_%s_data.json" % (mode), "w"))
    config.logger.info("数据处理完成")
    return examples


def show_watch_index(step, ave_loss=None, ave_hard_loss=None, ave_soft_loss=None,
                     ave_train_accr=None, eval_accr=None, now_lrs=None, ave_teacher_accr=None):
    if ave_loss is not None:
        config.logger.info("global_step@{}: loss@{}".format(step, ave_loss.mean))
        config.writer.add_scalar("train/loss", ave_loss.mean, step)
        ave_loss.reset()
    if ave_hard_loss is not None:
        config.logger.info("global_step@{}: hard_loss@{}".format(step, ave_hard_loss.mean))
        config.writer.add_scalar("train/hard_loss", ave_hard_loss.mean, step)
        ave_hard_loss.reset()
    if ave_soft_loss is not None:
        config.logger.info("global_step@{}: soft_loss@{}".format(step, ave_soft_loss.mean))
        config.writer.add_scalar("train/soft_loss", ave_soft_loss.mean, step)
        ave_soft_loss.reset()
    if ave_train_accr is not None:
        config.logger.info("epoch@{}: train_accr@{}  ".format(step, ave_train_accr.mean))
        config.writer.add_scalar("train/accr", ave_train_accr.mean, step)
        ave_train_accr.reset()
    if eval_accr is not None:
        config.logger.info("epoch@{}: val_accr@{}  ".format(step, eval_accr))
        config.writer.add_scalar("eval/accr", eval_accr, step)
    if ave_teacher_accr is not None:
        config.writer.add_scalar("eval/teacher_accr", ave_teacher_accr, step)
    if now_lrs is not None:
        for k, v in now_lrs.items():
            config.writer.add_scalar("train/" + str(k) + "-lr", float(v), step)


if __name__ == '__main__':
    gen_split_data("test")
    # context = "豆渣样或凝乳样白带，伴外阴奇痒难忍，即使是存在于大众背景也禁不住用手揉擦外阴以减轻痒感。常见于念珠菌感染，可用3%苏打水冲洗阴#idiom429947#后放米可定论腾片于阴道貌岸然深处，患病治疗持续期间暂停夫妻生活，以免互相传染，亦可用克霉灵片口服，每次二片，每日二次。夫妇同时服药三天为一疗程。此外还有许多杀念珠菌的药，但价格较贵昂，不宜推广。可找大夫诊治选举较说得通的药物。"
    # options = ["百尺竿头",
    #            "随波逐流",
    #            "方兴未艾",
    #            "身体力行",
    #            "一日千里",
    #            "三十而立",
    #            "逆水行舟",
    #            "日新月异",
    #            "百花齐放",
    #            "沧海一粟"]
    # tag = "#idiom429947#"
    # label = 6
    # convert_sentence_to_features(context, options, tag, label)
