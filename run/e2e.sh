CUDA_VISIBLE_DEVICES=2 python -m exec.ELDScript --mode train --model_name bert_fifo --register_policy fifo --train_limit 2500 --dev_limit 1000
CUDA_VISIBLE_DEVICES=2 python -m exec.ELDScript --mode train --model_name bert_pre_cluster --register_policy pre_cluster --train_limit 2500 --dev_limit 1000
