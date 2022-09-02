conda activate torch
python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simsiam/22bn8hmt/simsiam-imagenet32-22bn8hmt-ep=1000.ckpt \
      --pretrained_method simsiam


python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simsiam/22bn8hmt/simsiam-imagenet32-22bn8hmt-ep=1000.ckpt \
      --pretrained_method simsiam \
      --cpn_initial means




CUDA_VISIBLE_DEVICES=4,5 python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simsiam/22bn8hmt/simsiam-imagenet32-22bn8hmt-ep=1000.ckpt \
      --pretrained_method simsiam \
      --cpn_initial means \
      --perfix means-initial-