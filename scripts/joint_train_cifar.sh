data_path=/mnt/mmtech01/usr/liuwenzhuo/torch_ds
pretrained_dir=/mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/simclr/2mv95572
pretrained_path="$(ls $pretrained_dir/*.ckpt)"
echo "pretrained_path: $pretrained_path"
/share/wenzhuoliu/conda-envs/solo-learn/bin/python joint_train.py \
    --dataset cifar100 \
    --encoder resnet50 \
    --train_data_path ${data_path} \
    --val_data_path ${data_path} \
    --max_epochs 100 \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 1.0 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 10 \
    --crop_size 32 \
    --name joint-train-cifar \
    --pretrained_feature_extractor ${pretrained_path} \
    --project Incremental-CPN \
    --entity pigpeppa \
    --wandb \
    --save_checkpoint \
    --auto_resume

