from heapq import heappush, heappop, heapify
import collections


class HuffmanCodingFast(object):
    """
    Implementation of Huffman Coding using Heap,

    Citation: https://rosettacode.org/wiki/Huffman_coding#Python
    """

    def __init__(self, image):
        """
        Constructor

        Parameters
        ----------
        image : np.ndarray
            A single image to be compressed
        """

        self.image = image.reshape(-1)
        symb2freq = collections.Counter(self.image)
        self.huff = self.encode(symb2freq)
        self.encoding = ["".join([self.huff[symb] for symb in self.image])]

    def encode(self, symb2freq):
        """
        Huffman encode the given dict mapping symbols to weights
        """

        heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
        heapify(heap)
        while len(heap) > 1:
            lo = heappop(heap)
            hi = heappop(heap)
            for pair in lo[1:]:
                pair[1] = "0" + pair[1]
            for pair in hi[1:]:
                pair[1] = "1" + pair[1]
            heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        return dict(heappop(heap)[1:])
