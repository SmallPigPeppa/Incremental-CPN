conda activate torch
lambda=0.2
CUDA_VISIBLE_DEVICES=1 python main_continual_pretrain_feature.py \
      --num_tasks 5 \
      --pretrained_method byol \
      --pl_lambda $lambda \
      --dataset imagenet100 \
      --data_path /share/wenzhuoliu/code/debug/byol-deepmind/data_pretrained \
      --project Incremental-CPN-Imagenet100 \
      --perfix official-

#python main_continual.py \
#      --num_tasks 10 \
#      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
#      --pretrained_method byol
