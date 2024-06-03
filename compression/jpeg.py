import numpy as np
from PIL import Image
from io import BytesIO
from utils import sizeCalculator


class JPEGCompressed(object):
    """
    Implementation of compression using JPEG
    """

    def __init__(self, quality, images):
        """
        Constructor

        Parameters
        ----------
        images : np.ndarray (N, C, H, W)
            Images to be compressed
        quality : int
            Quality of the compression
        """

        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)
        elif len(images.shape) == 2:
            images = np.expand_dims(np.expand_dims(images, 0), 0)

        self.quality = quality
        self.num_images = images.shape[0]
        self.image_shape = images.shape[1:]
        self.encoding = self.encode(images)

    def encode(self, images):
        """
        Encode the given image using JPEG compression

        Parameters
        ----------
        images : np.ndarray (N, C, H, W)
            Images to be encoded

        Returns
        -------
        encoded_images : np.ndarray (N, C, H, W)
            Encoded images
        """

        encoded_images = []
        images_dummy = images.transpose(0, 2, 3, 1)
        for i in range(images.shape[0]):
            image = images_dummy[i].astype(np.uint8)
            if image.shape[2] == 1:
                image = image.reshape(image.shape[0], image.shape[1])
            image = Image.fromarray(image)

            encoding = BytesIO()
            image.save(encoding, format="JPEG", quality=self.quality)

            encoded_images.append(encoding)

        return encoded_images

    def decode(self):
        """
        Decode the given image using JPEG compression

        Returns
        -------
        decoded_images : np.ndarray (N, C, H, W)
            Decoded images
        """

        decoded_images = []
        for i in range(self.num_images):
            encoding = self.encoding[i]
            image = Image.open(encoding)
            image = np.array(image)
            if len(image.shape) == 2:
                image = np.expand_dims(image, 2)
            image = image.transpose(2, 0, 1)
            decoded_images.append(image)

        decoded_images = np.stack(decoded_images, axis=0)
        return decoded_images

    def __sizeof__(self):
        """
        Get the size of the object

        Returns
        -------
        size_image : int
            The size of the object
        size_total : int
            The size of the encoding
        """

        other_sizes = sizeCalculator([self.num_images, self.image_shape])
        size_encoding = sum(
            [len(encoding.getvalue()) * 8 for encoding in self.encoding]
        )

        return size_encoding, other_sizes + size_encoding

    def compressionRatio(self):
        """
        Get the compression ratio

        Returns
        -------
        image_ratio : float
            The compression ratio of the image
        total_ratio : float
            The total compression ratio
        """

        size_image, size_total = self.__sizeof__()
        image_ratio = size_image / (np.prod(self.image_shape) * 8 * self.num_images)
        total_ratio = size_total / (np.prod(self.image_shape) * 8 * self.num_images)

        return image_ratio, total_ratio
