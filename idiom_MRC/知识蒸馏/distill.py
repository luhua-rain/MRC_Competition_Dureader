from parameters import *
from model import XlnetCloze, load_model, save_model
from data import getdataLoader
from train import to_device, eval, seed_init, get_optimizer_group, show_lr, warmup_adajust, cal_accr
from functions import show_watch_index

# 加载teacher模型
def load_teacher_model(model_root, model):
    model_CKPT = t.load(model_root)
    state_dict = model_CKPT['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    config.logger.info("load {} successfully".format(model_root))

def load_teachers(root=config.teacher_root):
    teacher_names = os.listdir(root)
    teacher_models = []
    config.logger.info("load models {}".format(teacher_names))
    for t_name in tqdm(teacher_names):
        xlnet_config = XLNetConfig.from_json_file(config.xlnet_config_root)
        model = XlnetCloze(xlnet_config)
        load_teacher_model(config.teacher_root + t_name, model)
        model.freeze_all()
        model.eval()
        model.to(config.device)
        if config.n_gpu > 1:
            model = nn.DataParallel(model)
        teacher_models.append(model)
    config.logger.info("all teacher model load successfully")
    return teacher_models

# 计算带温度的BCE loss
def cross_entropy_loss_with_temperature(student_logits, teachers_logits, temperature):
    teacher_probs = t.zeros(student_logits.shape[0], 10).to(config.device)
    teachers_probs = []
    for i in range(len(teachers_logits)):
        teachers_logits[i] = teachers_logits[i] / temperature
        teachers_probs.append(F.softmax(teachers_logits[i], dim=1))
    for data in teachers_probs:
        teacher_probs += data
    teacher_probs = teacher_probs / len(teachers_probs)
    student_logits = student_logits / temperature
    loss = t.mean(-t.sum(teacher_probs * F.log_softmax(student_logits, dim=1), dim=1))
    # ipdb.set_trace()
    return loss, teacher_probs

def get_watch_index():
    ave_loss = meter.AverageValueMeter()
    ave_loss.reset()
    ave_hard_loss = meter.AverageValueMeter()
    ave_hard_loss.reset()
    ave_soft_loss = meter.AverageValueMeter()
    ave_soft_loss.reset()
    ave_train_accr = meter.AverageValueMeter()
    ave_train_accr.reset()
    return ave_loss, ave_hard_loss, ave_soft_loss, ave_train_accr

# 进行蒸馏
def do_distillation(start_epoch=-1):
    seed_init()
    train_dataloader, eval_dataloader = getdataLoader()
    xlnet_config = XLNetConfig.from_json_file(config.xlnet_config_root)
    student = XlnetCloze(xlnet_config)
    teacher_models = load_teachers()
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
            teachers_logits = []
            with t.no_grad():
                for teacher in teacher_models:
                    _, now_logits = teacher(input_ids, attention_mask, position, option_ids, tags, labels)
                    teachers_logits.append(now_logits)
            loss_hard = F.cross_entropy(student_logits, labels, reduction="mean")
            loss_soft, teacher_probs = cross_entropy_loss_with_temperature(student_logits, teachers_logits,
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


if __name__ == '__main__':
    # load_teachers()
    do_distillation(start_epoch=-1)
