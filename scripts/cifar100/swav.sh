conda activate torch
python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/swav/yaaves5o/swav-imagenet32-yaaves5o-ep=999.ckpt \
      --pretrained_method swav


python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/swav/yaaves5o/swav-imagenet32-yaaves5o-ep=999.ckpt \
      --pretrained_method swav
