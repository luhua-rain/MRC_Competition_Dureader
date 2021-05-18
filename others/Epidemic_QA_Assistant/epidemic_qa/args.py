import torch

context_data = './data/NCPPolicies_context_20200301.csv'
train_data = './data/NCPPolicies_train_20200301.csv'
test_data = './data/NCPPolicies_test.csv'

split_train_data = './data/train_data.json'
split_dev_data = './data/dev_data.json'

process_split_train_data = './data/pro_train_data.json'
process_split_dev_data = './data/pro_dev_data.json'

id_dev = './data/dev_id.json'
id_train = './data/train_id.json'

max_para_num = 30 # 使用f1召回的数量

# model and optimizer
epoch = 2
batch_size = 4
data_num = 12524 #{'noans': 6316, 'hasans': 6208}
gradient_accumulation_steps = 6
num_train_optimization_steps = int(data_num / gradient_accumulation_steps / batch_size) * epoch
log_step = int(data_num / batch_size / 3)

best_model_path = './checkpoints/best_model.pth'

# 数据处理
pad_id = 0
gt_score = 0.7
max_recall_para_len= 3000
max_c_len = 490
max_ans_len = 170
# other
device = torch.device("cuda", 0)
seed = 4

# 原始cls_logit没有影响，softmax的verify77.9, 不过：0.7828 mean:0.7838   cls_logit*5:0.7836
# soft_label + 0.2*verify: 0.7838，0.2*verify: 0.7743，soft_label：0.7709，pure：0.7794
# bert_recall_top15：0.7061