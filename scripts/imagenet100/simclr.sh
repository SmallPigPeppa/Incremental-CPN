conda activate torch
for lambda in 0.01 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8
do
  CUDA_VISIBLE_DEVICES=4 python main_continual_debug.py \
        --num_tasks 5 \
        --pretrained_model /share/wenzhuoliu/code/ssl-pretrained-models/simclr_imagenet.ckpt \
        --pretrained_method simclr \
        --pl_lambda $lambda \
        --dataset imagenet100 \
        --project Incremental-CPN-Imagenet100

done
#CUDA_VISIBLE_DEVICES=4,5 python main_continual.py \
#      --num_tasks 10 \
#      --pretrained_model /share/wenzhuoliu/code/ssl-pretrained-models/simclr_imagenet.ckpt \
#      --pretrained_method simclr \
#      --cpn_initial means \
#      --pl_lambda 0.3 \
#      --dataset imagenet100 \
#      --project Incremental-CPN-Imagenet100
