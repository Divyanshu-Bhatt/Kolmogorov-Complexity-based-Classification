import argparse
from utils import *
from model.knn import getKNearestNeighbours
from compression.description import getCompressionDescription


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--k", nargs="*", type=int, default=[5])
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="EUCLID",
        choices=["EUCLID", "GZIP_NCD", "GZIP_HD", "HUFF_NCD", "HUFF_HD"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=["MNIST", "CIFAR10", "IMAGENETTE"],
    )

    # If compression Description then
    parser.add_argument(
        "--compression_technique",
        type=str,
        default=None,
        choices=[
            None,
            "arithmetic",
            "btc",
            "gzip",
            "huffman",
            "jpeg",
            "svd",
        ],
    )
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--quality", type=int, default=10)

    args = parser.parse_args()

    if args.compression_technique is not None:
        if args.dataset == "MNIST":
            loader, _ = loadMNIST(
                args.batch_size,
                num_points=args.num_points,
                split=False,
            )

        elif args.dataset == "CIFAR10":
            loader, _ = loadCIFAR10(
                args.batch_size, num_points=args.num_points, split=False
            )

        elif args.dataset == "IMAGENETTE":
            loader = loadImagenette(args.batch_size)

        getCompressionDescription(
            loader,
            args.compression_technique,
            args.block_size,
            args.rank,
            args.quality,
            args,
        )

    else:
        sub_classes = None
        if args.dataset == "MNIST":
            trainloader, testloader = loadMNIST(
                args.batch_size, num_points=args.num_points, classes=sub_classes
            )
            if sub_classes != None:
                args.dataset = "MNIST" + "".join([str(i) for i in sub_classes])

        elif args.dataset == "CIFAR10":
            trainloader, testloader = loadCIFAR10(
                args.batch_size, num_points=args.num_points, classes=sub_classes
            )

        if len(args.k) == 1:
            args.k = args.k[0]

        getKNearestNeighbours(
            args.k, trainloader, testloader, args, args.distance_metric
        )
