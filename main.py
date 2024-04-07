import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import *
from model.knn import getKNearestNeighbours


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--distance_metric", type=str, default="EUCLID")
    args = parser.parse_args()

    sub_classes = [0, 1, 2]

    if args.dataset == "MNIST":
        trainloader, testloader = loadMNIST(
            args.batch_size, num_points=args.num_points, classes=sub_classes
        )
        args.dataset = "MNIST" + "".join([str(i) for i in sub_classes])

    elif args.dataset == "CIFAR10":
        trainloader, testloader = loadCIFAR10(
            args.batch_size, num_points=args.num_points, classes=sub_classes
        )

    getKNearestNeighbours(args.k, trainloader, testloader, args, args.distance_metric)
