
lambda=0.2

python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/ssl-pretrained-models/byol-resnet50-imagenet-1000ep-90sd6ar-ep=999.ckpt \
      --pretrained_method byol \
      --pl_lambda $lambda \
      --dataset imagenet100 \
      --project Incremental-CPN-Imagenet100


python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/ssl-pretrained-models/byol-resnet50-imagenet-1000ep-90sd6ar-ep=999.ckpt \
      --pretrained_method byol \
      --pl_lambda $lambda \
      --dataset imagenet100 \
      --project Incremental-CPN-Imagenet100

