import torch

seed = 42
device = torch.device("cuda", 0)
test_lines = 187818  # 多少条训练数据，即：len(features)

search_input_file = "../data/extracted/trainset/search.train.json"
zhidao_input_file = "../data/extracted/trainset/zhidao.train.json"
dev_zhidao_input_file = "../data/extracted/devset/zhidao.dev.json"
dev_search_input_file = "../data/extracted/devset/search.dev.json"

max_seq_length = 512
max_query_length = 60

output_dir = "./model_dir"
predict_example_files='predict.data'

max_para_num=5  # 选择几篇文档进行预测
learning_rate = 5e-5
batch_size = 4
num_train_epochs = 4
gradient_accumulation_steps = 8   # 梯度累积
num_train_optimization_steps = int(test_lines / gradient_accumulation_steps / batch_size) * num_train_epochs
log_step = int(test_lines / batch_size / 4)  # 每个epoch验证几次，默认4次
