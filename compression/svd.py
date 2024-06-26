import numpy as np
from utils import sizeCalculator


class SVDCompressed(object):
    """
    Implementation of compression using Singular Value Decomposition (SVD)
    """

    def __init__(self, rank, images):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of the approximation
        images : np.ndarray (N, C, H, W)
            Images to be compressed
        """

        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)
        elif len(images.shape) == 2:
            images = np.expand_dims(np.expand_dims(images, 0), 0)

        self.rank = rank
        self.num_images = images.shape[0]
        self.image_shape = images.shape[1:]
        self.U, self.S, self.VT = self.encode(images)

    def encode(self, images):
        """
        Encode the given image using SVD compression

        Parameters
        ----------
        images : np.ndarray (N, C, H, W)
            Images to be encoded

        Returns
        -------
        np.ndarray
            Encoded image
        """

        U, S, VT = np.linalg.svd(images, full_matrices=False)

        U = (np.round(U[:, :, :, : self.rank], 1) * 1e5).astype(np.int32)
        S = (np.round(S[:, :, : self.rank], 1) * 1e2).astype(np.int32)
        VT = (np.round(VT[:, :, : self.rank, :], 1) * 1e5).astype(np.int32)

        return U, S, VT

    def decode(self):
        """
        Decode the given image using SVD compression

        Parameters
        ----------
        U : np.ndarray (N, C, H, rank)
            U matrix
        S : np.ndarray (N, C, rank)
            S matrix
        VT : np.ndarray (N, C, rank, W)
            VT matrix

        Returns
        -------
        np.ndarray
            Decoded image
        """

        U, S, VT = self.U, self.S, self.VT
        U = U.astype(np.float32)/1e5
        S = S.astype(np.float32)/1e2
        VT = VT.astype(np.float32)/1e5

        s = np.zeros((S.shape[0], S.shape[1], S.shape[2], S.shape[2]))
        s[:, :, np.arange(S.shape[2]), np.arange(S.shape[2])] = S

        compressed_image = np.einsum("ijkl,ijll,ijlm->ijkm", U, s, VT)

        return compressed_image

    def __sizeof__(self):
        """
        Get the size of the compressed image

        Returns
        -------
        size : int
            The size of the compressed image
        """

        vectorised_size = np.vectorize(sizeCalculator)
        U_size = np.sum(vectorised_size(self.U))
        S_size = np.sum(vectorised_size(self.S))
        VT_size = np.sum(vectorised_size(self.VT))

        return U_size + S_size + VT_size

    def compressionRatio(self):
        """
        Get the compression ratio

        Returns
        -------
        ratio : float
            The compression ratio
        """

        original_size = np.prod(self.image_shape) * 8 * self.num_images
        compressed_size = self.__sizeof__()

        return compressed_size / original_size
