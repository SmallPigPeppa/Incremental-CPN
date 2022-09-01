import argparse


def parse_args_cpn() -> argparse.Namespace:
    """Parses feature extractor, dataset, pytorch lightning, linear eval specific and additional args.

    First adds an arg for the pretrained feature extractor, then adds dataset, pytorch lightning
    and linear eval specific args. If wandb is enabled, it adds checkpointer args. Finally, adds
    additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model", type=str,
                        default="/share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt",
                        required=True)
    parser.add_argument("--pretrained_method", type=str,
                        default="byol",
                        required=True)
    # incremental
    parser.add_argument("--num_tasks", type=int, default=5)
    # cpn
    parser.add_argument("--pl_lambda", type=float, default=0.2)
    parser.add_argument("--dim_feature", type=int, default=2048)

    parser.add_argument("--dataset", type=str, choices=["cifar100", "imagenet100"], default="cifar100")
    parser.add_argument("--data_path", type=str, default="/share/wenzhuoliu/torch_ds")

    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.3)
    # parse args
    args = parser.parse_args()

    return args
