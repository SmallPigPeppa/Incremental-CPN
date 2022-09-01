conda activate torch
CUDA_VISIBLE_DEVICES=6,7
pl_lambda
python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/s5fh5bvf/barlow_twins-imagenet32-s5fh5bvf-ep=999.ckpt \
      --pretrained_method barlow_twins

python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/s5fh5bvf/barlow_twins-imagenet32-s5fh5bvf-ep=999.ckpt \
      --pretrained_method barlow_twins

python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol

python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol

python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/mocov2plus/1kguyx5e/mocov2plus-imagenet32-1kguyx5e-ep=999.ckpt \
      --pretrained_method mocov2

python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/mocov2plus/1kguyx5e/mocov2plus-imagenet32-1kguyx5e-ep=999.ckpt \
      --pretrained_method mocov2


python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simclr/2mv95572/simclr-imagenet32-2mv95572-ep=999.ckpt \
      --pretrained_method simclr

python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simclr/2mv95572/simclr-imagenet32-2mv95572-ep=999.ckpt \
      --pretrained_method simclr


python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simsiam/22bn8hmt/simsiam-imagenet32-22bn8hmt-ep=999.ckpt \
      --pretrained_method simsiam


python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simsiam/22bn8hmt/simsiam-imagenet32-22bn8hmt-ep=999.ckpt \
      --pretrained_method simsiam

python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/swav/yaaves5o/swav-imagenet32-yaaves5o-ep=999.ckpt \
      --pretrained_method swav


python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/swav/yaaves5o/swav-imagenet32-yaaves5o-ep=999.ckpt \
      --pretrained_method swav