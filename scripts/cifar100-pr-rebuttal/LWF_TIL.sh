python pr_rebuttal_TIL.py \
      --num_tasks 5 \
      --pretrained_model /ppio_net0/code/PyCIL-latest/logs/lwf \
      --pretrained_method byol \
      --data_path /ppio_net0/torch_ds \
      --dataset cifar100 \
      --epochs 30 \
      --lr 0.1 \
      --batch_size 256 \
      --project jjj

