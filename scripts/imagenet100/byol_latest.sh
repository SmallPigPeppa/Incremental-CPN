conda activate torch
lambda=0.2
pretrained_dir=/share/wenzhuoliu/code/ssl-pretrained-models/byol
pretrained_path="$(ls $pretrained_dir/*.ckpt)"
echo "pretrained_path: $pretrained_path"
CUDA_VISIBLE_DEVICES=0 python main_continual.py \
      --num_tasks 5 \
      --pretrained_model $pretrained_path \
      --pretrained_method byol \
      --pl_lambda $lambda \
      --dataset imagenet100 \
      --project Incremental-CPN-Imagenet100 \
      --perfix latest-

#python main_continual.py \
#      --num_tasks 10 \
#      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
#      --pretrained_method byol
