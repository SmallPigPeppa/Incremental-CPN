conda activate torch
python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simclr/2mv95572/simclr-imagenet32-2mv95572-ep=999.ckpt \
      --pretrained_method simclr

python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simclr/2mv95572/simclr-imagenet32-2mv95572-ep=999.ckpt \
      --pretrained_method simclr
