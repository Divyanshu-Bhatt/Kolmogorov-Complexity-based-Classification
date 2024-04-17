import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from compression.description import getCompressionDescription
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--num_points", type=int, default=-1)
    parser.add_argument("--compression_technique", type=str, default="btc")
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=3)
    args = parser.parse_args()

    if args.dataset == "MNIST":
        loader, _ = loadMNIST(
            args.batch_size,
            num_points=args.num_points,
            split=False,
        )

    elif args.dataset == "CIFAR10":
        loader, _ = loadCIFAR10(
            args.batch_size, num_points=args.num_points, split=False
        )

    getCompressionDescription(
        loader, args.compression_technique, args.block_size, args.rank
    )
