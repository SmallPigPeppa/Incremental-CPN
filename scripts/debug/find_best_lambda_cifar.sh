conda activate torch

for lambda in 2.0 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05 0.025 0.01
do

CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/s5fh5bvf/barlow_twins-imagenet32-s5fh5bvf-ep=999.ckpt \
      --pretrained_method barlow_twins \
      --pl_lambda $lambda

CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/s5fh5bvf/barlow_twins-imagenet32-s5fh5bvf-ep=999.ckpt \
      --pretrained_method barlow_twins \
      --pl_lambda $lambda

CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol\
      --pl_lambda $lambda

CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol \
      --pl_lambda $lambda

CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/mocov2plus/1kguyx5e/mocov2plus-imagenet32-1kguyx5e-ep=999.ckpt \
      --pretrained_method mocov2 \
      --pl_lambda $lambda

CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/mocov2plus/1kguyx5e/mocov2plus-imagenet32-1kguyx5e-ep=999.ckpt \
      --pretrained_method mocov2 \
      --pl_lambda $lambda


CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simclr/2mv95572/simclr-imagenet32-2mv95572-ep=999.ckpt \
      --pretrained_method simclr \
      --pl_lambda $lambda

CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simclr/2mv95572/simclr-imagenet32-2mv95572-ep=999.ckpt \
      --pretrained_method simclr \
      --pl_lambda $lambda


CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simsiam/22bn8hmt/simsiam-imagenet32-22bn8hmt-ep=999.ckpt \
      --pretrained_method simsiam \
      --pl_lambda $lambda


CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simsiam/22bn8hmt/simsiam-imagenet32-22bn8hmt-ep=999.ckpt \
      --pretrained_method simsiam \
      --pl_lambda $lambda

CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/swav/yaaves5o/swav-imagenet32-yaaves5o-ep=999.ckpt \
      --pretrained_method swav \
      --pl_lambda $lambda



CUDA_VISIBLE_DEVICES=6,7 python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/swav/yaaves5o/swav-imagenet32-yaaves5o-ep=999.ckpt \
      --pretrained_method swav \
      --pl_lambda $lambda

done