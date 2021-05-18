from parameters import *
from model import XlnetCloze, load_model, save_model
from data import getdataLoader
from functions import show_watch_index


# 将变量加载到gpu上
def to_device(*args):
    ans = []
    for val in args:
        val = val.to(config.device)
        ans.append(val)
    return ans

# 进行warmup
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

# 分层学习率 warmup
def warmup_adajust(num_train_steps, global_step, optimizer):
    lr_ratio_this_step = warmup_linear(global_step / num_train_steps, config.warmup_proportion)
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'xlnet':
            param_group['lr'] = config.xlnet_learning_rate * lr_ratio_this_step
        elif param_group['name'] == 'other':
            param_group['lr'] = config.other_learning_rate * lr_ratio_this_step

# 初始化随机数种子
def seed_init():
    t.backends.cudnn.benchmark = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    t.manual_seed(config.seed)
    t.cuda.manual_seed(config.seed)
    if config.n_gpu > 0:
        t.cuda.manual_seed_all(config.seed)


# 监控LR
def show_lr(optimizer):
    lrs = {}
    for param_group in optimizer.param_groups:
        # config.logger.info("name:%s lr:%f" % (param_group['name'], param_group['lr']))
        lrs[param_group['name']] = param_group['lr']
    return lrs


# 设置分层学习率
def get_optimizer_group(model):
    param_optimizer = list(model.named_parameters())
    # hack to remove pooler, which is not used
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    other_part = ['my', 'idiom_embedding']
    no_main = no_decay + other_part
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not (any(nd in n for nd in no_main))],
         'weight_decay': 0.01, 'lr': config.xlnet_learning_rate, 'name': 'xlnet'},
        {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay) and
                                                      all(nd not in n for nd in other_part))],
         'weight_decay': 0.0, 'lr': config.xlnet_learning_rate, 'name': 'xlnet'},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in other_part)],
         'weight_dacy': 0.01, 'lr': config.other_learning_rate, 'name': 'other'},
    ]
    return optimizer_grouped_parameters


# 计算准确率
def cal_accr(logits, labels):
    ans = logits.max(dim=1)[1]
    true_count = (ans == labels).sum().item()
    return float(true_count / logits.shape[0])


def get_watch_index():
    ave_loss = meter.AverageValueMeter()
    ave_loss.reset()
    ave_train_accr = meter.AverageValueMeter()
    ave_train_accr.reset()
    return ave_loss, ave_train_accr

# 训练
def train(start_epoch=-1):
    seed_init()
    train_dataloader, eval_dataloader = getdataLoader()
    xlnet_config = XLNetConfig.from_json_file(config.xlnet_config_root)
    model = XlnetCloze(xlnet_config)
    optimizer_grouped_parameters = get_optimizer_group(model)
    num_train_steps = int(
        train_dataloader.dataset.__len__() / config.train_batch_size * config.num_train_epochs)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config.xlnet_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True,
                                  factor=config.decay, min_lr=config.min_lr,
                                  patience=config.patience)
    load_model(start_epoch, model)
    model.to(config.device)
    if config.n_gpu > 1:
        model = nn.DataParallel(model)
    config.logger.info("***** Running training *****")
    config.logger.info("  Num split train examples = %d", train_dataloader.dataset.__len__())
    config.logger.info("  Batch size = %d", config.train_batch_size)
    config.logger.info("  Num steps total = %d", num_train_steps)
    model.train()
    ave_loss, ave_train_accr = get_watch_index()
    global_step = 0
    # out = True
    for epoch in trange(start_epoch + 1, config.num_train_epochs):
        model.zero_grad()
        for batch in tqdm(train_dataloader):
            now_lrs = show_lr(optimizer)
            input_ids, attention_mask, position, option_ids, tags, labels = batch
            input_ids, attention_mask, position, option_ids, tags, labels = to_device(
                input_ids, attention_mask, position, option_ids, tags, labels
            )
            loss, logits = model(input_ids, attention_mask, position, option_ids, tags, labels)

            ave_train_accr.add(cal_accr(logits, labels))
            if config.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            loss.backward()
            ave_loss.add(loss.item())
            if global_step < num_train_steps * config.warmup_proportion:
                warmup_adajust(num_train_steps, global_step, optimizer)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            # ipdb.set_trace()
            if (global_step + 1) % config.show_loss_step == 0:
                show_watch_index(global_step, ave_loss=ave_loss)
            show_watch_index(global_step, now_lrs=now_lrs)
            # if out:
            #     break
        eval_accr = eval(model, eval_dataloader)
        show_watch_index(epoch, ave_train_accr=ave_train_accr, eval_accr=eval_accr)
        scheduler.step(eval_accr)
        save_model(epoch, model)


# eval准确率
def eval(model, eval_dataloader):
    model.eval()
    count, true_count = 0, 0
    for batch in tqdm(eval_dataloader):
        input_ids, attention_mask, position, option_ids, tags, labels = batch
        input_ids, attention_mask, position, option_ids, tags, labels = to_device(
            input_ids, attention_mask, position, option_ids, tags, labels
        )
        with t.no_grad():
            batch_logits = model(input_ids, attention_mask, position, option_ids, tags)
            ans = batch_logits.max(dim=1)[1]
            true_count += (ans == labels).sum().item()
            count += labels.shape[0]
    model.train()
    accr = true_count / count
    return accr


if __name__ == '__main__':
    # print(config.show_members())
    train(start_epoch=-1)
    # xlnet_config = Config.from_json_file(config.xlnet_config_root)
    # model = BertClozev2(xlnet_config, num_choices=10)
    # model.to(config.device)
    # train_dataloader, eval_dataloader = getdataLoader()
    # eval(model, eval_dataloader)
