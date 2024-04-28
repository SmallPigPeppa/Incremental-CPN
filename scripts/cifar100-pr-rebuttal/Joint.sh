python main_continual_linear_joint.py \
      --num_tasks 5 \
      --pretrained_model /ppio_net0/pretrained/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol \
      --data_path /ppio_net0/torch_ds \
      --dataset cifar100 \
      --project PR-rebuttal

#python main_continual_linear_joint.py \
#      --num_tasks 10 \
#      --pretrained_model /ppio_net0/pretrained/byol-imagenet32-t3pmk238-ep=999.ckpt \
#      --pretrained_method byol \
#      --data_path /ppio_net0/torch_ds \
#      --dataset cifar100 \
#      --project PR-rebuttal

