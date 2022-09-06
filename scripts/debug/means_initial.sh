conda activate torch


CUDA_VISIBLE_DEVICES=4,5 python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/mocov2plus/1kguyx5e/mocov2plus-imagenet32-1kguyx5e-ep=999.ckpt \
      --pretrained_method mocov2 \
      --cpn_initial means \
      --pl_lambda 0.5 \
      --perfix means-initial-

CUDA_VISIBLE_DEVICES=4,5 python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/mocov2plus/1kguyx5e/mocov2plus-imagenet32-1kguyx5e-ep=999.ckpt \
      --pretrained_method mocov2 \
      --cpn_initial means \
      --pl_lambda 0.5 \
      --perfix means-initial-

