import numpy as np
from utils import sizeCalculator


class BlockTruncationCoding(object):
    """
    Implementation of Block Truncation Coding (BTC)
    """

    def __init__(self, block_size, images):
        """
        Constructor

        Parameters
        ----------
        block_size : int
            Size of the block to be used for BTC
        images : np.ndarray (N, C, H, W)
            Images to be encoded
        """

        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)
        elif len(images.shape) == 2:
            images = np.expand_dims(np.expand_dims(images, 0), 0)

        self.block_size = block_size
        self.num_images = images.shape[0]
        self.image_shape = images.shape[1:]
        self.block_means, self.block_std, self.quantized_images = self.encode(images)

    def encode(self, images):
        """
        Encode the given image using BTC algorithm

        Parameters
        ----------
        images : np.ndarray (N, C, H, W)
            Images to be encoded

        Returns
        -------
        np.ndarray
            Encoded image
        """

        assert (
            images.shape[2] % self.block_size == 0
            and images.shape[3] % self.block_size == 0
        ), "Image dimensions should be divisible by block size"

        reshaped_image = images.reshape(
            images.shape[0],
            images.shape[1],
            images.shape[2] // self.block_size,
            self.block_size,
            images.shape[3] // self.block_size,
            self.block_size,
        )  # (N, C, H//block_size, block_size, W//block_size, block_size)

        block_means = np.mean(
            reshaped_image, axis=(3, 5)
        )  # (N, C, H//block_size, W//block_size)
        block_vars = np.var(
            reshaped_image, axis=(3, 5)
        )  # (N, C, H//block_size, W//block_size)
        block_std = np.sqrt(block_vars)

        quantized_image = reshaped_image > np.expand_dims(block_means, axis=(3, 5))
        quantized_image = quantized_image.reshape(-1).astype(int)

        return block_means, block_std, quantized_image

    def decode(self):
        """
        Decode the BTC encoded image

        Returns
        -------
        np.ndarray
            Decoded image
        """

        self.quantized_images = self.quantized_images.reshape(
            self.num_images,
            self.image_shape[0],
            self.image_shape[1] // self.block_size,
            self.block_size,
            self.image_shape[2] // self.block_size,
            self.block_size,
        )

        q = np.sum(
            self.quantized_images, axis=(3, 5)
        )  # Counting number of 1s in each block
        p = self.block_size**2 - q

        bias_corrector = np.sqrt(q / p)
        decoded_image1 = self.quantized_images * np.expand_dims(
            self.block_std / (bias_corrector + 1e-8), axis=(3, 5)
        )
        decoded_image2 = (self.quantized_images - 1) * np.expand_dims(
            self.block_std * bias_corrector, axis=(3, 5)
        )

        decoded_image = (
            decoded_image1
            + decoded_image2
            + np.expand_dims(self.block_means, axis=(3, 5))
        )

        decoded_image = decoded_image.reshape(
            self.num_images,
            self.image_shape[0],
            self.image_shape[1],
            self.image_shape[2],
        )

        return decoded_image.astype(np.uint8)

    def __sizeof__(self):
        """
        Get the size of the BTC coded image

        Returns
        -------
        size_image : int
            The size of the BTC coded image
        size_total : int
            The total size of the BTC coded image
        """

        # TODO Maybe use np.vectorise here.
        size_image = np.prod(np.shape(self.quantized_images))
        mean_size = sizeCalculator(self.block_means)
        var_size = sizeCalculator(self.block_std)
        other_sizes = sizeCalculator(
            [self.block_size, self.num_images, self.image_shape]
        )

        size_total = size_image + mean_size + var_size + other_sizes
        return size_image, size_total

    def compressionRatio(self):
        """
        Compute the compression ratio

        Returns
        -------
        image_ratio : float
            The compression ratio for the image
        total_ratio : float
            The compression ratio
        """

        size_image, size_total = self.__sizeof__()
        image_ratio = size_image / (np.prod(self.image_shape) * 8 * self.num_images)
        total_ratio = size_total / (np.prod(self.image_shape) * 8 * self.num_images)

        return image_ratio, total_ratio
