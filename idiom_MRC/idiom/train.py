from bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from bert.modeling import BertForMaskedLM
from bert.optimizer import BertAdam
from dataloader import Dureader
from evaluate import evaluate
from tqdm import tqdm
import numpy as np
import random
import torch
import args
import os

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def train():
    
    if args.use_bert:
        model = BertForMaskedLM.from_pretrained("bert-base-chinese",
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1)))
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

    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

    print("##### load dataloader #####")
    data = Dureader()
    train_dataloader, dev_dataloader = data.train_iter, data.dev_iter
    print("##### load model ###########")

    min_acc = 0.0

    for epoch in range(args.epoch):

        for step, batch in enumerate(tqdm(train_dataloader)):
            input_ids, label = batch.input_ids, batch.label

            input_ids, label = input_ids.to(args.device), label.to(args.device)

            output, loss = model(input_ids, masked_lm_labels=label)  

            loss = loss.mean() / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            del input_ids, label, loss, output

            if step % args.log_step == 4:

                dev_acc = evaluate(model, dev_dataloader)
                if min_acc < dev_acc:
                    min_acc = dev_acc
                    torch.save(model, "./checkpoints/best_model.pth")
                print("dev_acc: ", dev_acc)
                model.train()

    dev_acc = evaluate(model, dev_dataloader)
    if min_acc < dev_acc:
        min_acc = dev_acc
        torch.save(model, "./checkpoints/best_model.pth")
    print("dev_acc: ", dev_acc)

if __name__ == "__main__":
    # train()
    print(len(args.fre_idiom))
