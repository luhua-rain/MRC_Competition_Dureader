export CUDA_VISIBLE_DEVICES=0,1,2,3                                     # 设置显卡
export pretrain_dir=../init_model/                                      # 预训练模型目录
export lm=roberta_wwm_ext                                               # 原始预训练模型名称
export cache_dir=cache                                                  # features 目录
export output_dir=du2021                                                # 训练结果目录
export task=384_bert                                                    # 训练任务


python main_bert.py \
  --model_type bert \
  --summary log/$task \
  --model_name_or_path $pretrain_dir/$lm/pytorch_model.bin \
  --config_name $pretrain_dir/$lm/config.json \
  --tokenizer_name $pretrain_dir/$lm/vocab.txt \
  --threads 12 \
  --warmup_ratio 0.1 \
  --logging_ratio 0.1 \
  --save_ratio 0.1 \
  --do_lower_case \
  --overwrite_cache \
  --data_dir ./datasets \
  --train_file 2021_train.json \
  --predict_file 2021_dev.json \
  --test_file 2021_test1.json \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --max_query_length 32 \
  --max_answer_length 64 \
  --n_best_size 10 \
  --output_dir $output_dir/$task \
  --overwrite_output_dir \
  --do_fgm \
  --gc \
  --do_train \
  --evaluate_during_training \
  # --do_test \
  # --version_2_with_negative \
