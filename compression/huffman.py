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

    def __init__(self, image):
        """
        Constructor

        Parameters
        ----------
        image : numpy.ndarray
            Images to be compressed
        """

        self.image_shape = image.shape
        self.huffman_tree = self.__build_tree__(image)
        self.encoding = self.encodeImage(image)

    def image2histogram(self, image, normalised=True):
        """
        Compute the histogram of an image

        Parameters
        ----------
        image : ndarray
            The image to compute the histogram of
        normalised : bool, optional
            If True, normalise the histogram

        Returns
        -------
        histogram : ndarray
            The histogram of the image
        """

        histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
        if normalised:
            histogram /= np.prod(image.shape)

        return histogram

    def __build_tree__(self, image):
        """
        Build the Huffman tree

        Returns
        -------
        root : Node
            The root of the Huffman tree
        """

        histogram = self.image2histogram(image, normalised=False)
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

    def encodeImage(self, image):
        """
        Encode the image using the Huffman tree

        Parameters
        ----------
        image : numpy.ndarray
            The image to be encoded

        Returns
        -------
        encoded : str
            The encoded image
        """

        encoded_image = ""
        dictionary = self.__buildDictionary__()

        for val in image.flatten():
            encoded_image += dictionary[val]

        return encoded_image

    def decodeImage(self):
        """
        Decode the image using the Huffman tree

        Returns
        -------
        decoded : numpy.ndarray
            The decoded image
        """

        decoded_image = np.zeros(np.prod(self.image_shape), dtype=np.uint8)
        iterator = 0
        node = self.huffman_tree
        for i in range(len(self.encoding)):
            if self.encoding[i] == "0":
                node = node.left
            else:
                node = node.right

            if node.left is None and node.right is None:
                decoded_image[iterator] = node.value
                node = self.huffman_tree
                iterator += 1

        return decoded_image.reshape(self.image_shape)

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

        return 1 - (self.__sizeof__() / (np.prod(self.image_shape) * 8))


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
