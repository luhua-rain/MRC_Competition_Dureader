export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7                                    # 设置显卡
export pretrain_dir=./du2021/384_bert/
export lm=checkpoint-320


python main_bert.py \
  --model_type bert \
  --summary log/$task \
  --model_name_or_path $pretrain_dir/$lm/ \
  --config_name $pretrain_dir/$lm/ \
  --tokenizer_name $pretrain_dir/$lm/ \
  --evaluate_during_training \
  --threads 24 \
  --logging_ratio 0.1 \
  --do_lower_case \
  --data_dir ../datasets \
  --test_file 2021_test1.json \
  --test_prob_file test_prob.pkl \
  --per_gpu_eval_batch_size 64 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --max_query_length 32 \
  --max_answer_length 384 \
  --n_best_size 10 \
  --overwrite_cache \
  --do_test \
  --output_dir .

