import numpy as np
from utils import sizeCalculator


class Node(object):
    """
    Node class for Huffman tree
    """

    def __init__(self, value, prob, left=None, right=None):
        """
        Constructor

        Parameters
        ----------
        value : int
            Value of the node
        prob : float
            Probability of the node
        left : Node, optional
            Left child of the node
        right : Node, optional
            Right child of the node
        """

        self.value = value
        self.prob = prob
        self.left = left
        self.right = right

    def __sizeof__(self):
        """
        Get the size of the node

        Returns
        -------
        size : int
            The size of the node
        """

        size = 0
        if self.left is not None:
            size += self.left.__sizeof__()
        if self.right is not None:
            size += self.right.__sizeof__()

        return size + sizeCalculator(self.value) + sizeCalculator(self.prob)


class HuffmanCoded(object):
    """
    Huffman coding class
    """

    def __init__(self, images):
        """
        Constructor

        Parameters
        ----------
        images : numpy.ndarray (N, C, H, W)
            Images to be compressed
        """

        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)
        elif len(images.shape) == 2:
            images = np.expand_dims(np.expand_dims(images, 0), 0)

        self.num_images = images.shape[0]
        self.image_shape = images.shape[1:]
        self.huffman_tree = self.__build_tree__(images)

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

    def __build_tree__(self, images):
        """
        Build the Huffman tree

        Parameters
        ----------
        images : numpy.ndarray (N, C, H, W)
            Images to be compressed using the Huffman tree

        Returns
        -------
        root : Node
            The root of the Huffman tree
        """

        histogram = self.image2histogram(images, normalised=False)
        nodes = [Node(i, histogram[i]) for i in range(256)]

        while len(nodes) > 1:
            nodes.sort(key=lambda x: x.prob)
            left = nodes.pop(0)
            right = nodes.pop(0)
            parent = Node(None, left.prob + right.prob, left, right)
            nodes.append(parent)

        return nodes[0]

    def __buildDictionary__(self):
        """
        Build the dictionary for encoding the image

        Returns
        -------
        dictionary : dict
            The dictionary for encoding the image
        """

        dictionary = {}
        for i in range(256):
            dictionary[i] = ""

        def __buildDictionaryHelper__(node, code):
            if node.left is None and node.right is None:
                dictionary[node.value] = code
            else:
                __buildDictionaryHelper__(node.left, code + "0")
                __buildDictionaryHelper__(node.right, code + "1")

        __buildDictionaryHelper__(self.huffman_tree, "")

        return dictionary

    def encodeImage(self, images):
        """
        Encode the image using the Huffman tree

        Parameters
        ----------
        images : numpy.ndarray (N, C, H, W)
            The image to be encoded

        Returns
        -------
        encoded_images : list of str
            The encoded image
        """

        encoded_images = []
        dictionary = self.__buildDictionary__()

        for image in images:
            encode = ""
            for val in image.flatten():
                encode += dictionary[val]

            encoded_images.append(encode)

        return encoded_images

    def decodeImage(self, encoded_image):
        """
        Decode the image using the Huffman tree

        Parameters
        ----------
        encoded_image : str
            The encoded image

        Returns
        -------
        decoded : numpy.ndarray (C, H, W)
            The decoded image
        """

        decoded_image = np.zeros(np.prod(self.image_shape), dtype=np.uint8)
        iterator = 0
        node = self.huffman_tree
        for i in range(len(encoded_image)):
            if encoded_image[i] == "0":
                node = node.left
            else:
                node = node.right

            if node.left is None and node.right is None:
                decoded_image[iterator] = node.value
                node = self.huffman_tree
                iterator += 1

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
        for encoded_image in self.encoding:
            decoded_images.append(self.decodeImage(encoded_image))

        decoded_images = np.stack(decoded_images, axis=0)
        return decoded_images

    def __sizeof__(self):
        """
        Get the size of the Huffman coded image

        Returns
        -------
        size : int
            The size of the Huffman coded image
        """

        size = 0
        size += sizeCalculator(self.image_shape)
        size += self.huffman_tree.__sizeof__()
        size += sizeCalculator(self.encoding)

        return size

    def compressionRatio(self):
        """
        Compute the compression ratio

        Returns
        -------
        ratio : float
            The compression ratio
        """

        return self.__sizeof__() / (np.prod(self.image_shape) * 8 * self.num_images)


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     image_name = [
#         "4.2.03.tiff",
#         "5.1.10.tiff",
#         "5.1.11.tiff",
#         "5.2.09.tiff",
#         "5.3.02.tiff",
#         "7.1.01.tiff",
#         "7.1.02.tiff",
#         "7.1.08.tiff",
#     ]

#     for name in image_name:
#         image = plt.imread(name)
#         huffman = HuffmanCoded(image)
#         print(f"Compression ratio for {name}: {huffman.compressionRatio()}")

#     breakpoint()
