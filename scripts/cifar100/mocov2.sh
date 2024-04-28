conda activate torch
python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/mocov2plus/1kguyx5e/mocov2plus-imagenet32-1kguyx5e-ep=999.ckpt \
      --pretrained_method mocov2

python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/mocov2plus/1kguyx5e/mocov2plus-imagenet32-1kguyx5e-ep=999.ckpt \
      --pretrained_method mocov2
