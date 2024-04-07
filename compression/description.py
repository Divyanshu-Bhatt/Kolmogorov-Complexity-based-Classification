import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from compression.huffman import HuffmanCoded
from compression.arithmetic import ArithmeticCoded
from utils import sizeCalculator


def getArithmeticDesc(batch_images):
    """
    Get compression description for arithmetic coding

    Parameters
    ----------
    batch_images : numpy.ndarray
        The image to compress

    Returns
    -------
    time : float
        The time taken to compress the image
    ratio : float
        The compression ratio
    encoding_ratio : float
        The encoding compression ratio
    """

    start = time.time()
    arithmetic = ArithmeticCoded(batch_images)
    time_taken = time.time() - start

    arithmetic_encoding = sizeCalculator(arithmetic.encoding) / (
        np.prod(batch_images.shape) * 8
    )
    arithmetic_ratio = arithmetic.compressionRatio()

    return time_taken, arithmetic_ratio, arithmetic_encoding


def getHuffmanDesc(batch_images):
    """
    Get compression description for Huffman coding

    Parameters
    ----------
    batch_images : numpy.ndarray
        The image to compress

    Returns
    -------
    time : float
        The time taken to compress the image
    ratio : float
        The compression ratio
    encoding_ratio : float
        The encoding compression ratio
    """

    start = time.time()
    huffman = HuffmanCoded(batch_images)
    time_taken = time.time() - start

    huffman_encoding = sizeCalculator(huffman.encoding) / (
        np.prod(batch_images.shape) * 8
    )
    huffman_ratio = huffman.compressionRatio()

    return time_taken, huffman_ratio, huffman_encoding


def getHuffmanArithmeticDesciription(dataloader, args):
    """
    Get compression description for Huffman and Arithmetic coding and store
    it in a csv file

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader for the dataset
    args : argparse.ArgumentParser
        The arguments passed to the script

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame containing the compression ratios and times
    """

    arithmetic_ratios = []
    huffman_ratios = []
    arithmetic_encoding_ratios = []
    huffman_encoding_ratios = []
    arithmetic_time = []
    huffman_time = []

    for i, (batch_images, _) in enumerate(tqdm(dataloader, desc="Processing images")):
        batch_images = batch_images.numpy()

        arithmetic_time_taken, arithmetic_ratio, arithmetic_encoding = (
            getArithmeticDesc(batch_images)
        )

        arithmetic_encoding_ratios.append(arithmetic_encoding)
        arithmetic_ratios.append(arithmetic_ratio)
        arithmetic_time.append(arithmetic_time_taken)

        huffman_time_taken, huffman_ratio, huffman_encoding = getHuffmanDesc(
            batch_images
        )

        huffman_encoding_ratios.append(huffman_encoding)
        huffman_ratios.append(huffman_ratio)
        huffman_time.append(huffman_time_taken)

        df = pd.DataFrame(
            {
                "arithmetic_including_histogram": arithmetic_ratios,
                "huffman_including_tree": huffman_ratios,
                "arithmetic_encoding": arithmetic_encoding_ratios,
                "huffman_encoding": huffman_encoding_ratios,
                "arithmetic_time": arithmetic_time,
                "huffman_time": huffman_time,
            }
        ).to_csv(
            f"./results/compression_ratios_{args.dataset}_{args.batch_size}.csv",
            index=False,
        )

    return df


def saveHuffmanCompressedImages(dataloader, path):
    """
    Save the compressed images using Huffman coding

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader for the dataset
    path : os.path
        The path to save the compressed images
    """

    for i, (batch_images, targets) in enumerate(
        tqdm(dataloader, desc="Saving Huffman Compressed Images")
    ):
        encodings = []
        original_images = []
        batch_images = batch_images.numpy()
        targets = targets.numpy()

        for j, image in enumerate(batch_images):
            encodings.append(HuffmanCoded(image).encoding[0])
            original_images.append(image)
            pd.DataFrame(
                {
                    "encoding": encodings,
                    "targets": targets[: j + 1],
                }
            ).to_csv(os.path.join(path, f"compressed_{i}.csv"), index=False)
            np.save(os.path.join(path, f"original_{i}.npy"), original_images)
