from xlnet.modeling_xlnet import XLNetConfig, XLNetverify_cls, XLNet_rerank, XLNet_pure
from dataloader import Dureader
from evaluate import evaluate
from optimizer import BertAdam
from tqdm import tqdm
import random
import torch
import args
import numpy as np
from function import FGM, Freeze
from predict import pre

random.seed(args.seed)
torch.manual_seed(args.seed)

def train():
    f = open('log.txt', 'w')

    device = args.device

    config = XLNetConfig('/home/lh/xlnet_model_base/config.json')
    #config = XLNetConfig('/home/lh/xlnet_model/config.json')
    #model = XLNet_pure(config)
    model = XLNet_rerank(config)

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5, warmup=0.1,
                         t_total=args.num_train_optimization_steps)

    model = model.to(args.device)
    fgm = FGM(model)
    freeze = Freeze(model=model, batch_all=args.data_num // args.batch_size, layer_num=12)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    print("##### load dataloader #####")
    data = Dureader()
    min_acc = 0.0
    total_loss = []
    correct, total = 0, 0
    for epoch in range(args.epoch):

        for step, batch in enumerate(tqdm(data.get_batch())):

            input_ids, start, end, verify, gt_score, cls_label, ques_len = \
                batch['input_ids'], batch['start'], batch['end'], batch['verify_ids'], batch['gt_score'], batch['cls_label'], batch['ques_len']
            input_ids, start, end, verify, gt_score, cls_label, ques_len = \
                input_ids.to(device), start.to(device), end.to(device), verify.to(device), gt_score.to(device), cls_label.to(device), ques_len.to(device)
            tokentype_ids = batch['tokentype_ids'].to(device)
            context_cls = batch['context_cls'].to(device)

            loss, _, _, cls_logit = model(input_ids, verify_ids=verify, start_positions=start, end_positions=end,token_type_ids=tokentype_ids,
                                          gt_score=gt_score, cls_label=cls_label, ques_len=ques_len, ground_truth=batch['ground_truth'],
                                          context_cls=context_cls)

            loss = loss.mean() / args.gradient_accumulation_steps
            loss.backward()
            total_loss.append(loss.mean().item())

            fgm.attack()
            loss_adv, _, _, _ = model(input_ids, verify_ids=verify, start_positions=start, end_positions=end,token_type_ids=tokentype_ids,
                                          gt_score=gt_score, cls_label=cls_label, ques_len=ques_len, ground_truth=batch['ground_truth'])
            loss_adv = loss_adv.mean() / args.gradient_accumulation_steps
            loss_adv.backward()
            fgm.restore()
            #freeze.step()
            # _, predict = torch.max(cls_logit, dim=-1)
            # correct += (predict == cls_label).sum()
            total += input_ids.size(0)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % args.log_step == 4:
                print('train_acc:', float(correct) / total)
                correct, total = 0, 0
                print('loss: ', np.mean(total_loss))
                total_loss = []
                dev_acc = evaluate(model)
                message = "==> [eval] epoch {}, total_loss {}, train_acc {}".format(epoch, loss.item(), dev_acc)
                f.write(message+'\n')
                min_acc = dev_acc
                torch.save(model, "./checkpoints/"+str(round(dev_acc,4))+ ".pth")

                model.train()
                
            del input_ids, verify, loss, start, end, cls_logit

    print('loss: ', np.mean(total_loss))
    total_loss = []
    dev_acc = evaluate(model)
    message = "==> [eval] train_acc {}".format(dev_acc)
    f.write(message + '\n')
    torch.save(model, "./checkpoints/final_model.pth")
    dev_acc = evaluate(model, one_or_more='more')


if __name__ == "__main__":
    train()
    pre()

