#python pr_rebuttal_joint_linear.py \
#      --num_tasks 5 \
#      --pretrained_model /ppio_net0/pretrained/byol-imagenet32-t3pmk238-ep=999.ckpt \
#      --pretrained_method byol \
#      --data_path /ppio_net0/torch_ds \
#      --dataset cifar100 \
#      --epochs 30 \
#      --lr 0.1 \
#      --batch_size 256 \
#      --project PR-rebuttal-new

python pr_rebuttal_joint_linear_new.py \
      --num_tasks 5 \
      --pretrained_model /ppio_net0/pretrained/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol \
      --data_path /ppio_net0/torch_ds \
      --dataset cifar100 \
      --epochs 30 \
      --lr 0.1 \
      --batch_size 256 \
      --project PR-rebuttal-new-new


#python /ppio_net0/code/openapi.py stop afffe4aba431b52b
