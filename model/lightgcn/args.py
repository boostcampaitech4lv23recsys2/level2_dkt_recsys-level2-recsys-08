import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    
    num_layers = 1  # int
    
    parser.add_argument("--num_layers", default=1, type=int, help="num_layers")

    parser.add_argument("--user_wandb", default=True, type=bool, help="user_wandb")

    parser.add_argument("--use_cuda_if_available", default=True, type=bool, help="use_cuda_if_available")

    parser.add_argument("--basepath", default="/opt/ml/input/data/", type=str, help="basepath")

    parser.add_argument("--loader_verbose", default=True, type=bool, help="loader_verbose")

    parser.add_argument("--output_dir", default="./output/", type=str, help="output_dir")

    parser.add_argument("--pred_file", default="submission.csv", type=str, help="pred_file")

    parser.add_argument("--embedding_dim", default=64, type=int, help="embedding_dim")

    parser.add_argument("--alpha", default=None, type=list, help="alpha")

    parser.add_argument("--build_kwargs", default={}, type=dict, help="build_kwargs")
    
    parser.add_argument("--weight", default="./weight/best_model.pt", type=str, help="weight")

    parser.add_argument("--n_epoch", default=20, type=int, help="n_epoch")

    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning_rate")

    parser.add_argument("--weight_basepath", default="./weight", type=str, help="weight_basepath")

    args = parser.parse_args()

    return args
