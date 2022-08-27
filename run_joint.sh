for lambda in 0.2 0.1 0.05 0.025 0.01 0.005 0.0025 0.001 0.0005 0.00025
do
    data_path=/share/wenzhuoliu/torch_ds
    pretrained_dir=/share/wenzhuoliu/code/solo-learn/trained_models/swav/yaaves5o
    pretrained_path="$(ls $pretrained_dir/*.ckpt)"
    echo "pretrained_path: $pretrained_path"
    /share/wenzhuoliu/conda-envs/solo-learn/bin/python main_joint_train.py \
        --dataset cifar100 \
        --encoder resnet50 \
        --method cpn \
        --train_data_path ${data_path} \
        --val_data_path ${data_path} \
        --max_epochs 100 \
        --devices 0 \
        --accelerator gpu \
        --precision 16 \
        --optimizer sgd \
        --scheduler step \
        --lr 0.5 \
        --lr_decay_steps 60 80 \
        --weight_decay 0 \
        --batch_size 256 \
        --num_workers 10 \
        --crop_size 32 \
        --name joint-cifar-cpn-lamda:${lambda} \
        --pretrained_feature_extractor ${pretrained_path} \
        --project Incremental-CPN \
        --entity pigpeppa \
        --wandb \
        --save_checkpoint \
        --auto_resume \
        --pl_lambda ${lambda} \
        --mean 0.5071 0.4867 0.4408 \
        --std 0.2675 0.2565 0.2761
done