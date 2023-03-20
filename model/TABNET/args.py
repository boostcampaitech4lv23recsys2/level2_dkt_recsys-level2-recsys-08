import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--SEED", default=42, type=int)

    parser.add_argument("--N_EPOCHS", default=5, type=int)

    parser.add_argument("--BATCH_SZ", default=1024, type=int)

    parser.add_argument("--PATIENCE", default=3, type=int)

    parser.add_argument("--VIRTUAL_BS", default=128, type=int)

    parser.add_argument("--LR", default=0.01, type=float)

    parser.add_argument("--ND", default=8, type=int)

    parser.add_argument("--NA", default=8, type=int)

    parser.add_argument("--N_STEPS", default=3, type=int)

    parser.add_argument("--GAMMA", default=1.3, type=float)

    parser.add_argument("--N_INDEPENDENT", default=1, type=int)

    parser.add_argument("--LAMBDA", default=0, type=float)

    parser.add_argument("--N_SHARED", default=3, type=int)

    parser.add_argument("--MOMENTUM", default=0.1, type=float)

    parser.add_argument("--CLIP", default=1.0, type=float)

    parser.add_argument("--DATA_PATH", default='../../data/', type=str)

    parser.add_argument("--CAT_EMB_DIM", default=1, type=int)

    args = parser.parse_args()

    return args
