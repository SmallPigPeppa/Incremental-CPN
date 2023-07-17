conda activate torch
lambda=0.2
python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/ssl-pretrained-models/byol-resnet50-imagenet-100ep-25x5nqle-ep=99.ckpt \
      --pretrained_method byol \
      --pl_lambda $lambda \
      --dataset imagenet100 \
      --project Incremental-CPN-Imagenet100

#python main_continual.py \
#      --num_tasks 10 \
#      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
#      --pretrained_method byol
