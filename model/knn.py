import numpy as np
import pandas as pd
from tqdm import tqdm
from model.distance import *


def getKNearestNeighboursBatched(
    k, batch_images1, batch_images2, distance_metric="EUCLID"
):
    """
    Get the indices of the k nearest neighbours for given batches of images

    Parameters
    ----------
    k : int or list
        The number of nearest neighbours to get
    batch_images1 : numpy.ndarray
        The first batch of images
    batch_images2 : numpy.ndarray
        The second batch of images
    distance_metric : str, optional
        The distance metric to use

    Returns
    -------
    distances : numpy.ndarray
        The distances of the k nearest neighbours
    """

    if distance_metric == "EUCLID":
        distance_metric = euclideanDistance
    elif distance_metric == "GZIP_NCD":
        distance_metric = gzipCompressionDistanceNCD
    elif distance_metric == "GZIP_HD":
        distance_metric = gzipCompressionDistanceHD
    elif distance_metric == "HUFF_NCD":
        distance_metric = huffmanCompressionDistanceNCD
    elif distance_metric == "HUFF_HD":
        distance_metric = huffmanCompressionDistanceHD

    distance_matrix = distance_metric(batch_images1, batch_images2)
    return distance_matrix


def getIndices(distances, k):
    """
    Get the indices of the k nearest neighbours

    Parameters
    ----------
    distances : numpy.ndarray
        The distances of the k nearest neighbours
    k : int or list
        The number of nearest neighbours to get

    Returns
    -------
    indices : numpy.ndarray or list of numpy.ndarray
        The indices of the k nearest neighbours
    """

    if isinstance(k, int):
        indices = np.argsort(distances, axis=1)[:, :k]
    elif isinstance(k, list):
        indices = []
        for k_val in k:
            indices.append(np.argsort(distances, axis=1)[:, :k_val])
    else:
        raise ValueError("k must be an int or a list")

    return indices


def getKNearestNeighbours(k, trainloader, testloader, args, distance_metric="EUCLID"):
    """
    Get the k nearest neighbours for the given data

    Parameters
    ----------
    k : int or list
        The number of nearest neighbours to get
    trainloader : torch.utils.data.DataLoader
        The training data
    testloader : torch.utils.data.DataLoader
        The test data
    args : argparse.Namespace
        The arguments passed to the script
    distance_metric : str, optional
        The distance metric to use

    Returns
    -------
    predictions : dict
        The predicted target values for different values of k and the actual target values
    """

    predictions = {"actual_targets": []}
    if isinstance(k, int):
        predictions[f"predicted_targets_{k}"] = []
    elif isinstance(k, list):
        for k_val in k:
            predictions[f"predicted_targets_{k_val}"] = []

    for i, (test_images, test_targets) in enumerate(
        tqdm(testloader, desc="Finding Nearest Neighbours")
    ):
        distances = []
        train_targets = []
        test_images = test_images.numpy()
        test_targets = test_targets.numpy()

        for j, (train_images, batched_train_targets) in enumerate(
            tqdm(
                trainloader,
                desc=f"Finding Nearest Neighbours Batch{i}/{len(testloader)}",
            )
        ):
            train_images = train_images.numpy()
            batched_train_targets = batched_train_targets.numpy()

            batched_distances = getKNearestNeighboursBatched(
                k, test_images, train_images, distance_metric
            )
            distances.append(batched_distances)
            train_targets.append(batched_train_targets)

        distances = np.concatenate(distances, axis=1)
        train_targets = np.concatenate(train_targets, axis=0)
        indices = getIndices(distances, k)

        if isinstance(k, int):
            labels = train_targets[indices]
            prediction_labels = np.array(
                [np.argmax(np.bincount(label)) for label in labels]
            )
            predictions[f"predicted_targets_{k}"] = np.concatenate(
                [predictions[f"predicted_targets_{k}"], prediction_labels], axis=0
            )
            predictions["actual_targets"] = np.concatenate(
                [predictions["actual_targets"], test_targets], axis=0
            )

        elif isinstance(k, list):
            for itr, k_val in enumerate(k):
                labels = train_targets[indices[itr]]
                prediction_labels = np.array(
                    [np.argmax(np.bincount(label)) for label in labels]
                )
                predictions[f"predicted_targets_{k_val}"] = np.concatenate(
                    [predictions[f"predicted_targets_{k_val}"], prediction_labels],
                    axis=0,
                )
            predictions["actual_targets"] = np.concatenate(
                [predictions["actual_targets"], test_targets], axis=0
            )

        pd.DataFrame(predictions).to_csv(
            f"./results/knn_{args.dataset}_{args.batch_size}_{distance_metric}.csv",
            index=False,
        )
