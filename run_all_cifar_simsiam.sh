conda activate torch

for lambda in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 2.0 0.05 0.025 0.01
do
  CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
        --num_tasks 5 \
        --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simsiam/22bn8hmt/simsiam-imagenet32-22bn8hmt-ep=999.ckpt \
        --pretrained_method simsiam \
        --pl_lambda $lambda


  CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
        --num_tasks 10 \
        --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simsiam/22bn8hmt/simsiam-imagenet32-22bn8hmt-ep=999.ckpt \
        --pretrained_method simsiam \
        --pl_lambda $lambda


done