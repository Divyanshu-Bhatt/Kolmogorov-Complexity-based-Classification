import numpy as np
from binary_fractions import Binary
import decimal
import time
from utils import sizeCalculator


class ArithmeticCoded(object):
    """
    Arithmetic coding class
    """

    def __init__(self, images):
        """
        Constructor

        Parameters
        ----------
        image : ndarray (N, C, H, W)
            The image to be compressed
        """

        decimal.getcontext().prec = 2500

        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)
        elif len(images.shape) == 2:
            images = np.expand_dims(np.expand_dims(images, 0), 0)

        self.num_images = images.shape[0]
        self.image_shape = images.shape[1:]
        self.cdf = self.__getCDF__(images)

        self.encoding = self.encodeImage(images)

    def image2histogram(self, images, normalised=True):
        """
        Compute the histogram of an image

        Parameters
        ----------
        images : ndarray (N, C, H, W)
            The image to compute the histogram of
        normalised : bool, optional
            If True, normalise the histogram

        Returns
        -------
        histogram : ndarray
            The histogram of the image
        """

        histogram = np.bincount(images.ravel(), minlength=256).astype(np.float64)
        if normalised:
            histogram /= np.prod(images.shape)

        return histogram

    def __getCDF__(self, images):
        """
        Compute the cumulative distribution function of the image

        Parameters
        ----------
        images : ndarray (N, C, H, W)
            The image to compute the CDF of

        Returns
        -------
        cdf : ndarray
            The CDF of the image
        """

        histogram = self.image2histogram(images, normalised=True)
        cdf = np.cumsum(histogram, dtype=np.float64)
        return cdf

    def __getEncoding__(self, lower_limit, upper_limit):
        """
        Get the value with the minimum number of bits between the lower and upper limits

        Parameters
        ----------
        lower_limit : decimal.Decimal
            The lower limit
        upper_limit : decimal.Decimal
            The upper limit

        Returns
        -------
        encoding : decimal.Decimal
            The encoding
        """

        lower_limit = str(lower_limit)[2:]
        upper_limit = str(upper_limit)[2:]

        encoding = ""
        itr = 0
        for lower, upper in zip(lower_limit, upper_limit):
            if lower == upper:
                encoding += lower
            else:
                break
            itr += 1

        encoding += str(int(lower_limit[itr]) + 1)
        encoding = decimal.Decimal("0." + encoding)
        return encoding

    def encodeImage(self, images):
        """
        Encode the image

        Parameters
        ----------
        images : ndarray (N, C, H, W)
            The image to be encoded

        Returns
        -------
        encoded_images : list of decimal.Decimal
            The encoded images
        """

        cdf = self.cdf.copy()
        cdf = [decimal.Decimal(i) for i in cdf]
        cdf = np.concatenate(([decimal.Decimal(0)], cdf), dtype=decimal.Decimal)

        encoded_images = []

        for image in images:
            transformed_cdf = cdf.copy()
            lower_limit = decimal.Decimal(0)
            upper_limit = decimal.Decimal(1)

            for pixel_value in image.flatten():
                lower_limit = transformed_cdf[pixel_value]
                upper_limit = transformed_cdf[pixel_value + 1]
                transformed_cdf = lower_limit + (upper_limit - lower_limit) * cdf

            self.upper_limit = upper_limit
            self.lower_limit = lower_limit
            encode = self.__getEncoding__(lower_limit, upper_limit)
            encoded_images.append(encode)

        return encoded_images

    def __decodePixel__(self, transformed_cdf, cdf, encoded_image):
        """
        Decode a pixel

        Parameters
        ----------
        transformed_cdf : ndarray
            The linearly transformed CDF of the image
        cdf : ndarray
            The CDF of the image

        encoded_image : float
            The encoded image

        Returns
        -------
        pixel : int
            The decoded pixel
        transformed_cdf : ndarray
            The transformed CDF
        """

        pixel_value = np.sum(transformed_cdf <= encoded_image) - 1
        lower_limit = transformed_cdf[pixel_value]
        upper_limit = transformed_cdf[pixel_value + 1]
        transformed_cdf = lower_limit + (upper_limit - lower_limit) * cdf

        return pixel_value, transformed_cdf

    def decodeImage(self, encoded_image, cdf):
        """
        Decode the image

        Parameters
        ----------
        encoded_image : decimal.Decimal
            The encoded image
        cdf : ndarray
            The CDF of the images

        Returns
        -------
        image : ndarray
            The decoded image
        """

        decoded_image = np.zeros(np.prod(self.image_shape), dtype=np.uint8)
        transformed_cdf = cdf.copy()

        for i in range(np.prod(self.image_shape)):
            decoded_image[i], transformed_cdf = self.__decodePixel__(
                transformed_cdf, cdf, encoded_image
            )

        return decoded_image.reshape(self.image_shape)

    def decodeAllImages(self):
        """
        Decode all the images

        Returns
        -------
        decoded_images : numpy.ndarray (N, C, H, W)
            The decoded images
        """

        decoded_images = []
        cdf = self.cdf.copy()
        cdf = [decimal.Decimal(i) for i in cdf]
        cdf = np.concatenate(([decimal.Decimal(0)], cdf), dtype=decimal.Decimal)

        for encoded_image in self.encoding:
            decoded_images.append(self.decodeImage(encoded_image, cdf))

        decoded_images = np.stack(decoded_images, axis=0)
        return decoded_images

    def __sizeof__(self):
        """
        Get the size of the Arithmetic coded image

        Returns
        -------
        size : int
            The size of the Arithmetic coded image
        """

        size = 0
        size += sizeCalculator(self.image_shape)
        size += sizeCalculator(self.cdf)
        size += sizeCalculator(self.encoding)
        return size

    def compressionRatio(self):
        """
        Get the compression ratio of the image

        Returns
        -------
        ratio : float
            The compression ratio of the image
        """

        return self.__sizeof__() / (np.prod(self.image_shape) * 8 * self.num_images)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    start = time.time()
    image = np.random.randint(0, 256, (28, 28))
    arithmetic = ArithmeticCoded(image)
    print(time.time() - start)
    decoded_image = arithmetic.decodeImage()
    print(np.allclose(image, decoded_image))
    print(arithmetic.compressionRatio())
    breakpoint()
    # breakpoint()
