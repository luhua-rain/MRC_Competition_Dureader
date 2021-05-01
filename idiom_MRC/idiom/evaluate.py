from bert.modeling import BertForMaskedLM, BertConfig
from dataloader import Dureader
from tqdm import tqdm
import torch
import args

def evaluate(model, dataloader):

    correct, total = .0, .0
    model = model.to(args.device)
    model.eval()
    with torch.no_grad():

        for batch in tqdm(dataloader):
            input_ids, label, label_mask, label_relative, candidate = \
                 batch.input_ids, batch.label, batch.label_mask, \
                      batch.label_relative, batch.candidate

            input_ids, label, label_mask, label_relative, candidate = \
                input_ids.to(args.device), label.to(args.device), label_mask.to(args.device), label_relative.to(
                    args.device), candidate.to(args.device)

            out = model(input_ids)

            if args.use_No10:

                candidate = candidate.unsqueeze(1).repeat(1, out.size(1), 1)

                label_relative = label_relative.view(-1)
                label_relative_mask = label_relative >= 0
                active_labels = label_relative[label_relative_mask]

                active = label_mask.view(-1) == 1
                out = torch.gather(out, dim=-1, index=candidate)
                out = out.view(-1, 10)[active]
                _, active_logits = torch.max(out, dim=-1)

            else:

                active = label_mask.view(-1) == 1

                active_labels = label.view(-1)[active]
                _, predicted = torch.max(out.view(-1, args.idiom_num), dim=-1)
                active_logits = predicted.view(-1)[active]

            total += len(active_labels)
            correct += (active_logits == active_labels).sum()

            del active_logits, active_labels, out, candidate, label_relative

    print("total: ", total)
    print("correct: ", correct)
    return float(correct) / total

if __name__ == "__main__":


    model = idiomModel()
    model = torch.load(args.best_model_path, 'cpu')
    # model = torch.load(args.best_model_path, map_location="cuda:0")

    # model.load_state_dict(torch.load(args.best_model_path))
    data = Dureader()
    train, dev = data.train_iter, data.dev_iter
    acc = evaluate(model.cpu(), dev)
    print("acc: ", acc)




