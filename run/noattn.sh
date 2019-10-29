CUDA_VISIBLE_DEVICES=1 python -m exec.ELDScript --mode train --model_name noattn_all
CUDA_VISIBLE_DEVICES=1 python -m exec.ELDScript --mode train --model_name noattn_nochar --char_encoder none
CUDA_VISIBLE_DEVICES=1 python -m exec.ELDScript --mode train --model_name noattn_noword --word_encoder none
CUDA_VISIBLE_DEVICES=1 python -m exec.ELDScript --mode train --model_name noattn_nowordctx --word_context_encoder none
CUDA_VISIBLE_DEVICES=1 python -m exec.ELDScript --mode train --model_name noattn_noentctx --entity_context_encoder none
CUDA_VISIBLE_DEVICES=1 python -m exec.ELDScript --mode train --model_name noattn_norel --relation_encoder none
CUDA_VISIBLE_DEVICES=1 python -m exec.ELDScript --mode train --model_name noattn_notype --type_encoder none
