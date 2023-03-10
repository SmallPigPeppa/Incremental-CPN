#conda activate torch


python main_continual_pc.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol\
      --pl_lambda 0 \
      --project IPC-ablation-exp \
      --epochs 30 \
      --cpn_initial means \
      --perfix pc_

python main_continual_pc.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol\
      --pl_lambda 0 \
      --project IPC-ablation-exp \
      --epochs 30 \
      --cpn_initial means \
      --perfix pc_

python main_continual_pc.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol\
      --pl_lambda 0.2 \
      --project IPC-ablation-exp \
      --epochs 30 \
      --cpn_initial means \
      --perfix pc_w_pl_

python main_continual_pc.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol\
      --pl_lambda 0.2 \
      --project IPC-ablation-exp \
      --epochs 30 \
      --cpn_initial means \
      --perfix pc_w_pl_