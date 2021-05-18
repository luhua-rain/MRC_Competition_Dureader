from parameters import *
from data import IdiomData
from distill import load_teachers, get_watch_index
from model import XlnetCloze, load_model, save_model
from data import getdataLoader
from train import to_device, eval, seed_init, get_optimizer_group, show_lr, warmup_adajust, cal_accr
from functions import show_watch_index, get_ansdict


# 由于普通蒸馏太慢，这里存下teacher的输出，直接加载概率，进行快速蒸馏
def gen_soft_target():
    idiomdata = IdiomData("train", is_aug=False)
    train_dataloader = DataLoader(idiomdata, batch_size=config.train_batch_size, shuffle=False,
                                  num_workers=config.num_workers)
    teacher_models = load_teachers()
    all_results = {}
    for batch in tqdm(train_dataloader):
        input_ids, attention_mask, position, option_ids, tags, labels = batch
        input_ids, attention_mask, position, option_ids, tags, labels = to_device(
            input_ids, attention_mask, position, option_ids, tags, labels
        )
        teachers_logits = []
        with t.no_grad():
            for teacher in teacher_models:
                # now_logits: [batch, 10]
                _, now_logits = teacher(input_ids, attention_mask, position, option_ids, tags, labels)
                teachers_logits.append(now_logits)
            teacher_probs = t.zeros(teachers_logits[0].shape[0], 10).to(config.device)
            for i in range(len(teachers_logits)):
                teachers_logits[i] = teachers_logits[i] / 20          # 为什么要除以20
                teacher_probs += F.softmax(teachers_logits[i], dim=1) # 把所有模型的预测相加（之后再平均） ： [batch, 10]
        for i, tag in enumerate(tags):
            prob = teacher_probs[i].detach().cpu().numpy()
            all_results["#idiom%06d#" % tag] = prob
    for k, v in all_results.items():
        all_results[k] = v / len(teacher_models)
    pickle.dump(all_results, open(config.soft_label_file, "wb"))
    config.logger.info("训练数据软标签生成完毕")

def check_soft_target():
    soft_labels = pickle.load(open(config.soft_label_file[:-4] + ".pkl", "rb"))
    ans_dict = get_ansdict()
    total, right = 0, 0
    for k, v in tqdm(soft_labels.items()):
        label = ans_dict[k]
        soft_label = v.argmax()
        if soft_label == label:
            right += 1
        total += 1
    config.logger.info("总数为{}，正确数目为{}，软标签正确率为{}".format(total, right, float(right / total)))

def get_teacher_probs(soft_labels, tags):
    teacher_probs = t.zeros(len(tags), 10)
    for i, tag in enumerate(tags):
        now_teacher_probs = soft_labels["#idiom%06d#" % tag]
        teacher_probs[i] = t.tensor(now_teacher_probs)
    return teacher_probs

def cross_entropy_loss_with_temperature_v2(student_logits, teacher_probs, temperature):
    student_logits = student_logits / temperature
    loss = t.mean(-t.sum(teacher_probs * F.log_softmax(student_logits, dim=1), dim=1))   # 点积越大越相似， 但不是交叉熵
    # ipdb.set_trace()
    return loss, teacher_probs

def do_quick_distillation(start_epoch=-1):
    seed_init()
    train_dataloader, eval_dataloader = getdataLoader()
    xlnet_config = XLNetConfig.from_json_file(config.xlnet_config_root)
    student = XlnetCloze(xlnet_config)
    soft_labels = pickle.load(open(config.soft_label_file, "rb"))
    optimizer_grouped_parameters = get_optimizer_group(student)
    num_train_steps = int(train_dataloader.dataset.__len__() / config.train_batch_size * config.num_train_epochs)
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.xlnet_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, factor=config.decay,
                                  min_lr=config.min_lr, patience=config.patience)
    load_model(start_epoch, student, optimizer)
    if config.n_gpu > 1:
        student = nn.DataParallel(student)
    student.to(config.device)
    student.train()
    ave_loss, ave_hard_loss, ave_soft_loss, ave_train_accr = get_watch_index()
    global_step = (start_epoch + 1) * num_train_steps

    for epoch in trange(start_epoch + 1, config.num_train_epochs):
        student.zero_grad()
        for batch in tqdm(train_dataloader):
            input_ids, attention_mask, position, option_ids, tags, labels = batch
            input_ids, attention_mask, position, option_ids, tags, labels = to_device(
                input_ids, attention_mask, position, option_ids, tags, labels
            )
            _, student_logits = student(input_ids, attention_mask, position, option_ids, tags, labels)
            teacher_probs = get_teacher_probs(soft_labels, tags).to(config.device)
            loss_hard = F.cross_entropy(student_logits, labels, reduction="mean")
            loss_soft, teacher_probs = cross_entropy_loss_with_temperature_v2(student_logits, teacher_probs,
                                                                              config.temperature)
            loss = config.alpha * loss_hard + (1.0 - config.alpha) * config.temperature * config.temperature * loss_soft

            loss.backward()
            ave_train_accr.add(cal_accr(student_logits, labels))
            ave_loss.add((config.alpha * loss_hard + (1.0 - config.alpha) * loss_soft).item())
            ave_soft_loss.add(loss_soft.item())
            ave_hard_loss.add(loss_hard.item())
            optimizer.step()
            optimizer.zero_grad()
            # ipdb.set_trace()
            show_watch_index(global_step, ave_teacher_accr=cal_accr(logits=teacher_probs, labels=labels))
            if (global_step + 1) % config.show_loss_step == 0:
                now_lrs = show_lr(optimizer)
                show_watch_index(global_step, ave_hard_loss=ave_hard_loss, now_lrs=now_lrs,
                                 ave_soft_loss=ave_soft_loss, ave_loss=ave_loss)
            if global_step <= num_train_steps * config.warmup_proportion:
                warmup_adajust(num_train_steps, global_step, optimizer)
            global_step += 1

        eval_accr = eval(student, eval_dataloader)
        show_watch_index(epoch, eval_accr=eval_accr, ave_train_accr=ave_train_accr)
        scheduler.step(eval_accr)
        save_model(epoch, student, optimizer)


def prob_combine():
    prob1 = pickle.load(open(config.data_root + "train_soft_label1.pkl", "rb"))
    prob2 = pickle.load(open(config.data_root + "train_soft_label2.pkl", "rb"))
    prob3 = pickle.load(open(config.data_root + "train_soft_label3.pkl", "rb"))
    result = {}
    for i, (k, v) in enumerate(tqdm(prob1.items())):
        now_prob1 = v
        now_prob2 = prob2[k]
        now_prob3 = prob3[k]
        ave = (now_prob1 + now_prob2 + now_prob3) / 3
        result[k] = ave
        if i % 50000 == 0:
            print(now_prob3 - ave)
    pickle.dump(result, open(config.data_root + "train_soft_label4.pkl", "wb"))


if __name__ == '__main__':
    import nvidia_smi

    #prob_combine()
    check_soft_target()
    # prob3 = pickle.load(open(config.data_root + "train_soft_label3.pkl", "rb"))
    # print(prob3)
    # check_soft_target()
    # do_quick_distillation(start_epoch=-1)
