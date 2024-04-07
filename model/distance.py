import numpy as np
import gzip
from compression.huffman import HuffmanCoded


def hammingDistance(compressed_image1, compressed_image2):
    """
    Calculate the Hamming Distance

    Parameters
    ----------
    compressed_image1 : str
        Code when image 1 is compressed
    compressed_image2 : str
        Code when image 2 is compressed

    Returns
    -------
    hamming_distance : int
        The Hamming Distance
    """

    maximum_length = max(len(compressed_image1), len(compressed_image2))
    minimum_length = min(len(compressed_image1), len(compressed_image2))
    max_length_diff = maximum_length - minimum_length

    hamming_distance = max_length_diff + sum(
        [
            bit1 != bit2
            for bit1, bit2 in zip(
                compressed_image1[: minimum_length + 1],
                compressed_image2[: minimum_length + 1],
            )
        ]
    )

    return hamming_distance


def normalisedCompressionDistance(
    compressed_together, compressed_image1, compressed_image2
):
    """
    Calculate the Normalised Compression Distance

    Parameters
    ----------
    compressed_together : str
        Code when both images are compressed together
    compressed_image1 : str
        Code when image 1 is compressed
    compressed_image2 : str
        Code when image 2 is compressed

    Returns
    -------
    ncd : float
        The Normalised Compression Distance
    """

    ncd = len(compressed_together) - min(len(compressed_image1), len(compressed_image2))
    ncd /= max(len(compressed_image1), len(compressed_image2))

    return ncd


def euclideanDistance(batch_images1, batch_images2):
    """
    Calculate the Euclidean Distance between two batches of images

    Parameters
    ----------
    batch_images1 : numpy.ndarray
        The first batch of images
    batch_images2 : numpy.ndarray
        The second batch of images

    Returns
    -------
    distance_matrix : numpy.ndarray
        The Euclidean Distance matrix
    """

    distance_matrix = np.sqrt(
        np.sum((batch_images1[:, None] - batch_images2) ** 2, axis=(2, 3, 4))
    )

    return distance_matrix


def gzipCompressionDistanceNCD(batch_images1, batch_images2):
    """
    Calculate the Gzip Compression Distance between two batches of images

    Parameters
    ----------
    batch_images1 : numpy.ndarray
        The first batch of images
    batch_images2 : numpy.ndarray
        The second batch of images

    Returns
    -------
    distance_matrix : numpy.ndarray
        The Gzip Compression Distance matrix
    """

    distance_matrix = []

    for image1 in batch_images1:
        compressed_image1 = gzip.compress(image1)
        for image2 in batch_images2:
            compressed_image2 = gzip.compress(image2)
            compressed_together = gzip.compress(np.concatenate([image1, image2]))
            ncd = normalisedCompressionDistance(
                compressed_together, compressed_image1, compressed_image2
            )
            distance_matrix.append(ncd)

    distance_matrix = np.array(distance_matrix).reshape(
        batch_images1.shape[0], batch_images2.shape[0]
    )
    return distance_matrix


def gzipCompressionDistanceHD(batch_images1, batch_images2):
    """
    Calculate the Gzip Compression Distance between two batches of images

    Parameters
    ----------
    batch_images1 : numpy.ndarray
        The first batch of images
    batch_images2 : numpy.ndarray
        The second batch of images

    Returns
    -------
    distance_matrix : numpy.ndarray
        The Gzip Compression Distance matrix
    """

    distance_matrix = []

    for image1 in batch_images1:
        compressed_image1 = gzip.compress(image1)
        for image2 in batch_images2:
            compressed_image2 = gzip.compress(image2)
            hamming_distance = hammingDistance(compressed_image1, compressed_image2)
            distance_matrix.append(hamming_distance)

    distance_matrix = np.array(distance_matrix).reshape(
        batch_images1.shape[0], batch_images2.shape[0]
    )
    return distance_matrix


def huffmanCompressionDistanceNCD(batch_images1, batch_images2):
    """
    Calculate the Huffman Compression Distance between two batches of images

    Parameters
    ----------
    batch_images1 : numpy.ndarray
        The first batch of images
    batch_images2 : numpy.ndarray
        The second batch of images

    Returns
    -------
    distance_matrix : numpy.ndarray
        The Huffman Compression Distance matrix
    """

    distance_matrix = []

    for image1 in batch_images1:
        compressed_image1 = HuffmanCoded(image1).encoding[0]
        for image2 in batch_images2:
            compressed_image2 = HuffmanCoded(image2).encoding[0]
            compressed_together = HuffmanCoded(
                np.concatenate([image1, image2])
            ).encoding[0]
            ncd = normalisedCompressionDistance(
                compressed_together, compressed_image1, compressed_image2
            )
            distance_matrix.append(ncd)

    distance_matrix = np.array(distance_matrix).reshape(
        batch_images1.shape[0], batch_images2.shape[0]
    )
    return distance_matrix


def huffmanCompressionDistanceHD(batch_images1, batch_images2):
    """
    Calculate the Huffman Compression Distance between two batches of images

    Parameters
    ----------
    batch_images1 : numpy.ndarray
        The first batch of images
    batch_images2 : numpy.ndarray
        The second batch of images

    Returns
    -------
    distance_matrix : numpy.ndarray
        The Huffman Compression Distance matrix
    """

    distance_matrix = []

    for image1 in batch_images1:
        compressed_image1 = HuffmanCoded(image1).encoding[0]
        for image2 in batch_images2:
            compressed_image2 = HuffmanCoded(image2).encoding[0]
            hamming_distance = hammingDistance(compressed_image1, compressed_image2)
            distance_matrix.append(hamming_distance)

    distance_matrix = np.array(distance_matrix).reshape(
        batch_images1.shape[0], batch_images2.shape[0]
    )
    return distance_matrix
