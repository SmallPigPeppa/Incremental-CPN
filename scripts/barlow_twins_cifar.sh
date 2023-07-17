conda activate torch
python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/s5fh5bvf/barlow_twins-imagenet32-s5fh5bvf-ep=999.ckpt \
      --pretrained_method barlow_twins


python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/s5fh5bvf/barlow_twins-imagenet32-s5fh5bvf-ep=999.ckpt \
      --pretrained_method barlow_twins
