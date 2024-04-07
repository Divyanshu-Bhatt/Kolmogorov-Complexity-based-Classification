import numpy as np
from binary_fractions import Binary
from tqdm import tqdm
import decimal
import torch
import torchvision
import torchvision.transforms as transforms


def sizeCalculator(value):
    """
    Calculate the size of the value in bits

    Parameters
    ----------
    value : float or tuple or list or int
        The value to calculate the size of
    """

    if isinstance(value, (int, float)):
        return len(str(Binary(value))[2:])
    elif isinstance(value, (tuple, list, np.ndarray)):
        size = 0
        for val in value:
            size += sizeCalculator(val)
        return size
    elif value is None:
        return 0
    elif isinstance(value, (decimal.Decimal)):
        if value == 0:
            return 0
        return sizeCalculator(int(str(value).split(".")[1]))
    elif isinstance(value, str):
        return len(value)


def loadMNISTDataset(train_bool=False):
    """
    Load the MNIST dataset

    Parameters
    ----------
    train_bool : bool, optional
        If True, load the train dataset, else load the test dataset

    Returns
    -------
    dataset : torch.utils.data.Dataset
        The MNIST dataset
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.numpy()),
            transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),
        ]
    )

    dataset = torchvision.datasets.MNIST(
        root="./data", train=train_bool, download=True, transform=transform
    )

    return dataset


def loadCIFAR10Dataset(train_bool=False):
    """
    Load the CIFAR10 dataset

    Parameters
    ----------
    train_bool : bool
        If True, load the train dataset, else load the test dataset

    Returns
    -------
    dataset : torch.utils.data.Dataset
        The CIFAR10 dataset
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.numpy()),
            transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=train_bool, download=True, transform=transform
    )

    return dataset


def loadMNIST(
    batch_size,
    train_bool=False,
    num_points=1024,
    shuffle=True,
    classes=None,
    train_split=0.8,
):
    """
    Load the MNIST dataset

    Parameters
    ----------
    batch_size : int
        The batch size
    train_bool : bool, optional
        If True, load the train dataset, else load the test dataset
    num_points : int, optional
        The number of images to load
    shuffle : bool, optional
        If True, shuffle the dataset
    classes : list, optional
        The classes to load
    train_split : float, optional
        The split of the train data

    Returns
    -------
    trainloader : torch.utils.data.DataLoader
        The dataloader for the train dataset
    testloader : torch.utils.data.DataLoader
        The dataloader for the test dataset
    """

    dataset = loadMNISTDataset(train_bool)

    if classes is not None:
        indices = []
        for c in classes:
            indices.append(np.where(np.array(dataset.targets) == c)[0])
        indices = np.concatenate(indices)
        dataset = torch.utils.data.Subset(dataset, indices)

    if num_points != -1:
        indices = np.random.choice(len(dataset), num_points, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle
    )

    return trainloader, testloader


def loadCIFAR10(batch_size, train_bool=False, num_points=1024, shuffle=True, split=0.8):
    """
    Load the CIFAR10 dataset

    Parameters
    ----------
    batch_size : int
        The batch size
    train_bool : bool, optional
        If True, load the train dataset, else load the test dataset
    num_points : int, optional
        The number of images to load
    shuffle : bool, optional
        If True, shuffle the dataset

    Returns
    -------
    trainloader : torch.utils.data.DataLoader
        The dataloader for the train dataset
    testloader : torch.utils.data.DataLoader
        The dataloader for the test dataset
    """

    dataset = loadCIFAR10Dataset(train_bool)

    if num_points != -1:
        indices = np.random.choice(len(dataset), num_points, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle
    )

    return trainloader, testloader


def getHistogramDataset(dataloader):
    """
    Get the histogram corresponding to the dataset

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader for the dataset

    Returns
    -------
    histogram : list of numpy.ndarray corresponding to each channel
        The histogram of the dataset
    """

    for batch_images, _ in dataloader:
        image_shape = batch_images.shape[1:]
        break

    histogram = [np.zeros(256) for _ in range(image_shape[0])]

    for i, (batch_images, _) in enumerate(
        tqdm(dataloader, desc="Genearting Histogram")
    ):
        batch_images = batch_images.numpy()
        for j in range(image_shape[0]):
            histogram[j] += np.bincount(
                batch_images[:, j].ravel(), minlength=256
            ).astype(np.float64)

    return histogram
