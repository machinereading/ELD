CUDA_VISIBLE_DEVICES=3 python -m exec.ELDScript --mode train --model_name allattn_precluster_th0.3 --transformer separate --char_encoder selfattn --word_encoder selfattn --relation_encoder selfattn --type_encoder selfattn --register_policy pre_cluster