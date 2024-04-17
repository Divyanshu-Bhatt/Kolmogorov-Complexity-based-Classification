import numpy as np


def mseError(original, decoded):
    """
    Compute the mean squared error between the original and decoded images

    Parameters
    ----------
    original : numpy.ndarray
        The original image
    decoded : numpy.ndarray
        The decoded image
    """

    return np.mean((original - decoded) ** 2)


def circularConvolution(image1, image2):
    """
    Computing the Circular Convolution of two images using summation

    Parameters
    ----------
    image1 : np.ndarray
        The first image
    image2 : np.ndarray
        The second image

    Returns
    -------
    np.ndarray
        The circular convolution of the two images
    """

    N1, M1 = image1.shape
    N2, M2 = image2.shape
    output_image_shape = (max(N1, N2), max(M1, M2))

    xx, yy = np.meshgrid(np.arange(N2), np.arange(M2))

    # Finding all the indices for the circular convolution
    xindices = (np.arange(output_image_shape[0]).reshape(-1, 1, 1) - xx) % N1
    yindices = (np.arange(output_image_shape[1]).reshape(-1, 1, 1) - yy) % M1

    # Repeating the indices to match the shape of the output image
    # Repeating both of them differently to get all the combinations
    xindices = np.tile(xindices, (M1, 1, 1, 1))
    yindices = np.tile(np.expand_dims(yindices, axis=1), (1, N1, 1, 1))

    output_image = np.sum(image1[xindices, yindices] * image2.T, axis=(2, 3))
    return output_image.T


def gaussianFilter(n, m=None, sigma=1):
    """
    Compute the gaussian filter of size n x m

    Parameters
    ----------
    n : int
        The size of the gaussian filter
    m : int, optional
        The size of the gaussian filter, if not provided, m = n
    sigma : float, optional
        The standard deviation of the gaussian filter

    Returns
    -------
    np.ndarray
        The gaussian filter of size n x m
    """

    if m is None:
        m = n
    x = np.arange(-n // 2 + 1, n // 2 + 1)
    y = np.arange(-m // 2 + 1, m // 2 + 1)
    x, y = np.meshgrid(x, y)
    filter = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return filter / np.sum(filter)


def ssimIndexImages(
    image1,
    image2,
    w=None,
    stabilising_constants=(
        (0.01 * 255) ** 2,
        (0.03 * 255) ** 2,
        ((0.03 * 255) ** 2) / 2,
    ),
):
    """
    Structural Similarity Index between two images

    Parameters
    ----------
    image1 : np.array
        First image
    image2 : np.array
        Second image
    w : np.array or None, optional
        Weights for the SSIM calculation, by default None then a Gaussian window is used
        of size 7x7 and standard deviation 1.5

    Returns
    -------
    float
        Structural Similarity Index between the two images
    np.array
        Structural Similarity Index between the two images for each channel
    """

    assert image1.shape == image2.shape, "Images must have the same shape"

    if w is None:
        w = gaussianFilter(7, 7, 1.5)

    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    C1, C2, C3 = stabilising_constants

    mu1 = circularConvolution(image1, w)
    mu2 = circularConvolution(image2, w)
    luminance = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)

    var1 = circularConvolution(image1**2, w) - mu1**2
    var2 = circularConvolution(image2**2, w) - mu2**2
    var1 = np.maximum(var1, 0)  # To avoid negative values due to numerical errors
    var2 = np.maximum(var2, 0)
    sigma1 = np.sqrt(var1)
    sigma2 = np.sqrt(var2)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)

    sigma12 = circularConvolution((image1 * image2), w) - mu1 * mu2
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    ssim_map = luminance * contrast * structure
    mean_ssim = np.mean(ssim_map)

    return mean_ssim
