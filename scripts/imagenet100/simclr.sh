conda activate torch
CUDA_VISIBLE_DEVICES=4,5 python main_continual_debug.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/ssl-pretrained-models/simclr_imagenet.ckpt \
      --pretrained_method simclr \
      --cpn_initial means \
      --pl_lambda 0.3 \
      --dataset imagenet100 \
      --project Incremental-CPN-Imagenet100

#CUDA_VISIBLE_DEVICES=4,5 python main_continual.py \
#      --num_tasks 10 \
#      --pretrained_model /share/wenzhuoliu/code/ssl-pretrained-models/simclr_imagenet.ckpt \
#      --pretrained_method simclr \
#      --cpn_initial means \
#      --pl_lambda 0.3 \
#      --dataset imagenet100 \
#      --project Incremental-CPN-Imagenet100
