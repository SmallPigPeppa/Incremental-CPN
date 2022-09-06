conda activate torch



for lambda in 0.5 0.6 0.7 0.8 0.9 1.0 2.0 3.0 4.0 5.0
do
  CUDA_VISIBLE_DEVICES=0,1 python main_continual.py \
        --num_tasks 5 \
        --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/simsiam/22bn8hmt/simsiam-imagenet32-22bn8hmt-ep=1000.ckpt \
        --pretrained_method simsiam \
        --pl_lambda $lambda

done

for lambda in 0.12 0.15 0.2
do
  CUDA_VISIBLE_DEVICES=0,1 python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/s5fh5bvf/barlow_twins-imagenet32-s5fh5bvf-ep=999.ckpt \
      --pretrained_method barlow_twins \
      --pl_lambda $lambda

done

