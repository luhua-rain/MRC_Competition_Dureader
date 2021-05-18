from functions import gen_split_data
from chinese_eda import augment
from parameters import *

# 将数据转化为直接输入xlnet的格式
def convert_sentence_to_features(context, options, tag, label=None):  # tag = "#idiom429947#" , label = 6
    # options = ["百尺竿头","随波逐流","方兴未艾", position = tokens.index("<mask>")

    parts = re.split(tag, context)
    assert len(parts) == 2
    before_part = config.tokenizer.tokenize(parts[0]) if len(parts[0]) > 0 else []
    after_part = config.tokenizer.tokenize(parts[1]) if len(parts[1]) > 0 else []

    half_length = int(config.max_seq_length / 2)
    if len(before_part) < half_length:  # cut at tail
        st = 0
        ed = min(len(before_part) + 1 + len(after_part), config.max_seq_length - 2)
    elif len(after_part) < half_length:  # cut at head
        ed = len(before_part) + 1 + len(after_part)
        st = max(0, ed - (config.max_seq_length - 2))
    else:  # cut at both sides
        st = len(before_part) + 3 - half_length
        ed = len(before_part) + 1 + half_length

    option_ids = [config.idiom2index[each] for each in options]
    tokens = before_part + ["<mask>"] + after_part
    tokens = ["<cls>"] + tokens[st:ed] + ["<sep>"]
    position = tokens.index("<mask>")
    input_ids = config.tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)
    attention_mask += [0] * (config.max_seq_length - len(input_ids))
    input_ids += [config.tokenizer.convert_tokens_to_ids(config.tokenizer.pad_token)] * (
            config.max_seq_length - len(input_ids))

    res = [input_ids, attention_mask, position, option_ids, int(tag[6: -1])]
    if label is not None:
        res.append(label)
    new_res = []
    for item in res:
        new_res.append(t.tensor(item).long())
    return new_res


# dataset类
class IdiomData(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if self.mode == "train" or self.mode == "test":
            if os.path.isfile(config.data_root + "split_%s_data.json" % (mode)):
                with open(config.data_root + "split_%s_data.json" % (mode), "r") as f:
                    self.split_data = json.load(f)
            else:
                self.split_data = gen_split_data(mode)
        elif self.mode == "wrong":
            with open(config.data_root + "split_train_data.json", "r") as f:
                self.all_data = json.load(f)
            self.split_data = []
            for data in self.all_data:
                tag = int(data["tag"][6: -1])
                if tag in config.wrong_case:
                    self.split_data.append(data)
        random.shuffle(self.split_data)
        config.logger.info("读取数据结束")
        if self.mode == "train":
            train_size = int((1 - config.eval_ratio) * self.__len__())
            self.eval_size = self.__len__() - train_size

    def __getitem__(self, idx):
        context = self.split_data[idx]["context"]
        options = self.split_data[idx]["options"]
        tag = self.split_data[idx]["tag"]
        label = None
        if self.mode in ["train", "wrong"]:
            label = self.split_data[idx]["label"]
        if (self.mode == "train" and idx >= self.eval_size) or self.mode == "wrong":
            context = augment(context)
        return convert_sentence_to_features(
            context=context, options=options, tag=tag, label=label)

    def __len__(self):
        return len(self.split_data)


# 得到dataloader，分为train模式和test模式
def getdataLoader(mode="train"):
    idiomdata = IdiomData(mode)
    if mode == "train":
        train_size = int((1 - config.eval_ratio) * idiomdata.__len__())
        eval_size = idiomdata.__len__() - train_size
        eval_dataset = Subset(idiomdata, range(eval_size))
        train_dataset = Subset(idiomdata, range(eval_size, idiomdata.__len__()))
        train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False,
                                      num_workers=config.num_workers)
        eval_dataloader = DataLoader(eval_dataset, batch_size=config.train_batch_size, shuffle=False,
                                     num_workers=config.num_workers)
        return train_dataloader, eval_dataloader
    else:
        dataloader = DataLoader(idiomdata, batch_size=config.test_batch_size, shuffle=False,
                                num_workers=config.num_workers)
        return dataloader


if __name__ == '__main__':
    train_dataloader, eval_dataloader = getdataLoader(mode="train")
    for _ in tqdm(train_dataloader):
        pass
