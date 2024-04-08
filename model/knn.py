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
    #     indices = np.argsort(distance_matrix, axis=1)[:, :k]
    # elif isinstance(k, list):
    #     indices = []
    #     for k_val in k:
    #         breakpoint()
    #         distance_matrix = distance_metric(batch_images1, batch_images2)
    #         indices.append(np.argsort(distance_matrix, axis=1)[:, : k_val + 1])
    # else:
    #     raise ValueError("k must be an int or a list")

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
    # return predictions, targets


# def getKNearestNeighbours(data, k, batch_size=None, distance_metric="euclidean"):
#     """
#     Get the k nearest neighbours of the data

#     Parameters
#     ----------
#     data : numpy.ndarray
#         The data to get the k nearest neighbours of
#     k : int
#         The number of nearest neighbours to get
#     batch_size : int, optional
#         The batch size to use
#     distance_metric : str, optional
#         The distance metric to use

#     Returns
#     -------
#     indices : numpy.ndarray
#         The indices of the k nearest neighbours
#     distances : numpy.ndarray
#         The distances of the k nearest neighbours
#     """

#     if batch_size is None:
#         batched_data = [data]
#     else:
#         batched_data = np.array_split(data, batch_size)

#     indices, distances = [], []

#     for batch in tqdm(batched_data, desc="Finding Nearest Neighbours Batch"):
#         if distance_metric == "euclidean":
#             batched_distances = np.linalg.norm(data - batch[:, np.newaxis], axis=2)

#         batched_indices = np.argsort(batched_distances, axis=1)[:, : k + 1]
#         batched_distances = np.sort(batched_distances, axis=1)[:, : k + 1]

#         # Remove the first element as it is the distance to itself
#         batched_indices = batched_indices[:, 1:]
#         batched_distances = batched_distances[:, 1:]

#         indices.append(batched_indices)
#         distances.append(batched_distances)

#     indices = np.concatenate(indices, axis=0)
#     distances = np.concatenate(distances, axis=0)

#     return indices, distances


# def getKNearestNeighboursCompressedVersion(
#     k, train_paths, test_path, distance_metric="euclidean"
# ):
#     """
#     Get the k nearest neighbours from batched files

#     Parameters
#     ----------
#     k : int
#         The number of nearest neighbours to get
#     train_paths : list of os.path
#         The paths to the training data
#     test_path : os.path
#         The path to the test data
#     coding_scheme : str, optional
#         The coding scheme to use

#     Returns
#     -------
#     target_values : numpy.ndarray
#         The target values of the k nearest neighbours
#     """

#     if coding_scheme == "arithmetic":
#         coding_scheme = ArithmeticCoded
#     elif coding_scheme == "huffman":
#         coding_scheme = HuffmanCoded
#     elif coding_scheme == "gzip":
#         coding_scheme = gzip.compress
#     else:
#         raise ValueError("Coding scheme not recognised")

#     test_compressed = pd.read_csv(test_path)
#     test_original = np.load(
#         test_path.replace("compressed", "original").replace("csv", "npy")
#     )

#     distance_matrix = []
#     targets = []

#     for train_path in tqdm(train_paths, desc="NN Batch"):

#         train_compressed = pd.read_csv(train_path)
#         train_original = np.load(
#             train_path.replace("compressed", "original").replace("csv", "npy")
#         )

#         batched_distances = []
#         for i in range(test_original.shape[0]):
#             for j in range(train_original.shape[0]):
#                 # compressed_together = coding_scheme(
#                 #     np.concatenate([test_original[i], train_original[j]])
#                 # ).encoding[0]
#                 # compressed_image1 = test_compressed["encoding"][i]
#                 # compressed_image2 = train_compressed["encoding"][j]

#                 compressed_together = gzip.compress(
#                     np.concatenate([test_original[i], train_original[j]])
#                 )
#                 compressed_image1 = gzip.compress(test_original[i])
#                 compressed_image2 = gzip.compress(train_original[j])

#                 ncd = normalisedCompressionDistance(
#                     compressed_together, compressed_image1, compressed_image2
#                 )
#                 batched_distances.append(ncd)

#         batched_distances = np.array(batched_distances).reshape(
#             test_original.shape[0], train_original.shape[0]
#         )

#         distance_matrix.append(batched_distances)
#         targets.append(train_compressed["targets"].values)

#     distance_matrix = np.concatenate(distance_matrix, axis=1)
#     batched_indices = np.argsort(distance_matrix, axis=1)[:, :k]
#     targets = np.concatenate(targets, axis=0)
#     targets = targets[batched_indices]

#     # Get the most common target value
#     target_values = np.array([np.argmax(np.bincount(target)) for target in targets])

#     return target_values
