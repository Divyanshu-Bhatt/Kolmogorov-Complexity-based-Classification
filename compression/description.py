import os
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
from compression.svd import SVDCompressed
from compression.huffman import HuffmanCoded
from compression.btc import BlockTruncationCoding
from compression.arithmetic import ArithmeticCoded
from compression.jpeg import JPEGCompressed
from utils import sizeCalculator
from image_metrics import mseError, ssimIndexImages


def getArithmeticDesc(batch_images, cache):
    """
    Get compression description for arithmetic coding

    Parameters
    ----------
    batch_images : numpy.ndarray
        The image to compress
    cache : dict
        The cache for storing the metrics

    Returns
    -------
    cache : dict
        The cache for storing the metrics
    """

    arithmetic = ArithmeticCoded(batch_images)
    encoding_ratio, ratio = arithmetic.compressionRatio()
    cache["encoding_ratio"].append(encoding_ratio)
    cache["ratio"].append(ratio)

    return cache


def getHuffmanDesc(batch_images, cache):
    """
    Get compression description for Huffman coding

    Parameters
    ----------
    batch_images : numpy.ndarray
        The image to compress
    cache : dict
        The cache for storing the metrics

    Returns
    -------
    cache : dict
        The cache for storing the metrics
    """

    huffman = HuffmanCoded(batch_images)
    encoding_ratio, ratio = huffman.compressionRatio()
    cache["encoding_ratio"].append(encoding_ratio)
    cache["ratio"].append(ratio)

    return cache


def getGzipCompressionDesc(batch_images, cache):
    """
    Get compression description for gzip compression

    Parameters
    ----------
    batch_images : numpy.ndarray
        The image to compress
    cache : dict
        The cache for storing the metrics

    Returns
    -------
    cache : dict
        The cache for storing the metrics
    """

    compressed_image = gzip.compress(batch_images)
    encoding_size = sizeCalculator(compressed_image)
    encoding_ratio = encoding_size / (np.prod(batch_images.shape) * 8)
    cache["encoding_ratio"].append(encoding_ratio)

    return cache


def getBTCCompressionDesc(batch_images, block_size, cache):
    """
    Get compression description for BTC coding

    Parameters
    ----------
    batch_images : numpy.ndarray
        The image to compress
    block_size : int
        The block size

    Returns
    -------
    cache : dict
        The cache for storing the metrics
    """

    btc = BlockTruncationCoding(block_size, batch_images)
    encoding_ratio, ratio = btc.compressionRatio()
    decoded_images = btc.decode()

    mse = mseError(batch_images, decoded_images)
    ssim_index = []
    for i in range(batch_images.shape[0]):
        for j in range(batch_images.shape[1]):
            ssim_index.append(ssimIndexImages(batch_images[i, j], decoded_images[i, j]))
    ssim_index = np.mean(ssim_index)
    
    cache["encoding_ratio"].append(encoding_ratio)
    cache["ratio"].append(ratio)
    cache["mse"].append(mse)
    cache["ssim_index"].append(ssim_index)

    return cache


def getSVDDesc(batch_images, rank, cache):
    """
    Get compression description for SVD compression

    Parameters
    ----------
    batch_images : numpy.ndarray
        The image to compress
    rank : int
        The rank of the SVD
    cache : dict
        The cache for storing the metrics

    Returns
    -------
    cache : dict
        The cache for storing the metrics
    """

    svd = SVDCompressed(rank, batch_images)
    encoding_ratio = svd.compressionRatio()
    decoded_images = svd.decode()

    mse = mseError(batch_images, decoded_images)
    ssim_index = []
    for i in range(batch_images.shape[0]):
        for j in range(batch_images.shape[1]):
            ssim_index.append(ssimIndexImages(batch_images[i, j], decoded_images[i, j]))
    ssim_index = np.mean(ssim_index)

    cache["encoding_ratio"].append(encoding_ratio)
    cache["mse"].append(mse)
    cache["ssim_index"].append(ssim_index)

    return cache


def getJpegCompressionDesc(batch_images, quality, cache):
    """
    Get compression description for JPEG compression

    Parameters
    ----------
    batch_images : numpy.ndarray
        The image to compress
    cache : dict
        The cache for storing the metrics

    Returns
    -------
    cache : dict
        The cache for storing the metrics
    """

    jpeg = JPEGCompressed(quality, batch_images)
    encoding_ratio, ratio = jpeg.compressionRatio()
    decoded_images = jpeg.decode()

    mse = mseError(batch_images, decoded_images)
    ssim_index = []
    for i in range(batch_images.shape[0]):
        for j in range(batch_images.shape[1]):
            ssim_index.append(ssimIndexImages(batch_images[i, j], decoded_images[i, j]))

    ssim_index = np.mean(ssim_index)
    cache["encoding_ratio"].append(encoding_ratio)
    cache["ratio"].append(ratio)
    cache["mse"].append(mse)
    cache["ssim_index"].append(ssim_index)

    return cache


def getCompressionDescription(
    dataloader, compression_type, block_size=8, rank=10, quality=10, args=None
):
    """
    Get the compression description for the given dataloader

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader for the dataset
    compression_type : str
        The compression type
    block_size : int, optional
        The block size for BTC coding if compression_type is BTC
    rank : int, optional
        The rank for SVD compression if compression_type is SVD
    quality : int, optional
        The quality for JPEG compression if compression_type is JPEG
    args : argparse
        All the arguments for running this function
    """

    cache = {}
    path_name = f"./results/{compression_type}/{args.dataset}/"
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    if compression_type == "btc":
        compression = lambda x, y: getBTCCompressionDesc(x, block_size, y)
        cache = {"encoding_ratio": [], "ratio": [], "mse": [], "ssim_index": []}
        path_name += f"block_{block_size}"
    elif compression_type == "arithmetic":
        compression = getArithmeticDesc
        cache = {"encoding_ratio": [], "ratio": []}
    elif compression_type == "huffman":
        compression = getHuffmanDesc
        cache = {"encoding_ratio": [], "ratio": []}
    elif compression_type == "gzip":
        compression = getGzipCompressionDesc
        cache = {"encoding_ratio": []}
    elif compression_type == "svd":
        compression = lambda x, y: getSVDDesc(x, rank, y)
        cache = {"encoding_ratio": [], "mse": [], "ssim_index": []}
        path_name += f"rank_{rank}"
    elif compression_type == "jpeg":
        compression = lambda x, y: getJpegCompressionDesc(x, quality, y)
        cache = {"encoding_ratio": [], "ratio": [], "mse": [], "ssim_index": []}
        path_name += f"quality_{quality}"
    
    path_name += f"_batch{args.batch_size}.csv"

    for batch_images, _ in tqdm(dataloader):
        batch_images = batch_images.numpy()
        cache = compression(batch_images, cache)
        pd.DataFrame(cache).to_csv(path_name, index=False)
