import time
import torch
import argparse
import numpy as np
import torchvision
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from compression.huffman import HuffmanCoded
from compression.arithmetic import ArithmeticCoded

from utils import sizeCalculator


def loadMNIST(batch_size, train_bool=False, num_points=1024):
    """
    Load the MNIST dataset

    Parameters
    ----------
    batch_size : int
        The batch size
    train_bool : bool, optional
        If True, load the train dataset, else load the test dataset
    num_points : int, optional
        The number of images to load

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        The dataloader for the MNIST dataset
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.numpy()),
            transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),
        ]
    )

    dataset = torchvision.datasets.MNIST(
        root="./data", train=train_bool, download=True, transform=transform
    )
    indices = np.random.choice(len(dataset), num_points, replace=False)
    dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="MNIST")
    args = parser.parse_args()

    batch_size = args.batch_size

    arithmetic_ratios = []
    huffman_ratios = []
    arithmetic_encoding_ratios = []
    huffman_encoding_ratios = []
    arithmetic_time = []
    huffman_time = []

    if args.dataset == "MNIST":
        dataloader = loadMNIST(batch_size)

    for i, (batch_images, _) in enumerate(tqdm(dataloader, desc="Processing images")):
        batch_images = batch_images.numpy()

        start = time.time()
        arithmetic = ArithmeticCoded(batch_images)
        arithmetic_time.append(time.time() - start)

        start = time.time()
        huffman = HuffmanCoded(batch_images)
        huffman_time.append(time.time() - start)

        arithmetic_encoding_ratios.append(
            sizeCalculator(arithmetic.encoding) / (np.prod(batch_images.shape) * 8)
        )
        huffman_encoding_ratios.append(
            sizeCalculator(huffman.encoding) / (np.prod(batch_images.shape) * 8)
        )
        arithmetic_ratios.append(arithmetic.compressionRatio())
        huffman_ratios.append(huffman.compressionRatio())

        pd.DataFrame(
            {
                "arithmetic_including_histogram": arithmetic_ratios,
                "huffman_including_tree": huffman_ratios,
                "arithmetic_encoding": arithmetic_encoding_ratios,
                "huffman_encoding": huffman_encoding_ratios,
                "arithmetic_time": arithmetic_time,
                "huffman_time": huffman_time,
            }
        ).to_csv(
            f"./results/compression_ratios_{args.dataset}{args.batch_size}.csv",
            index=False,
        )
