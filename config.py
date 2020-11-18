import argparse

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--model", type=str, default="transformer")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--nclass", type=int, default=14)
    parser.add_argument("--ntokens", type=int, default=7551)
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--embsize", type=int, default=200)
    parser.add_argument("--hiddensize", type=int, default=200)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--outchannel", type=int, default=2)
    parser.add_argument("--resume", type=bool, default=False)

    return parser.parse_args()
