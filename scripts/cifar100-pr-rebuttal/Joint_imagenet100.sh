#python pr_rebuttal_joint.py \
#      --num_tasks 5 \
#      --pretrained_model /ppio_net0/pretrained/byol-imagenet-3tx0at58-ep=999.ckpt \
#      --pretrained_method byol \
#      --data_path /ppio_net0/torch_ds \
#      --dataset imagenet100 \
#      --epochs 30 \
#      --lr 0.1 \
#      --batch_size 256 \
#      --project PR-rebuttal


python pr_rebuttal_joint_linear.py \
      --num_tasks 10 \
      --pretrained_model /ppio_net0/pretrained/byol-imagenet-3tx0at58-ep=999.ckpt \
      --pretrained_method byol \
      --data_path /ppio_net0/torch_ds \
      --dataset imagenet100 \
      --epochs 30 \
      --lr 0.1 \
      --batch_size 256 \
      --project PR-rebuttal-new