#conda activate torch

lambda=0.2
CUDA_VISIBLE_DEVICES=0 python main_continual_task_acc.py \
      --num_tasks 10 \
      --epochs 1 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol \
      --cpn_initial means \
      --pl_lambda $lambda \
      --dataset cifar100 \
      --project Incremental-CPN-cifar100-task-acc
