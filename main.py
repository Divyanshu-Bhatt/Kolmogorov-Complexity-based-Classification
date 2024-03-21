import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from compression.arithmetic import ArithmeticCoded
from compression.huffman import HuffmanCoded
from utils import sizeCalculator


if __name__ == "__main__":
    # image_names = [
    #     "4.2.03.tiff",
    #     "5.1.10.tiff",
    #     "5.1.11.tiff",
    #     "5.2.09.tiff",
    #     "5.3.02.tiff",
    #     "7.1.01.tiff",
    #     "7.1.02.tiff",
    #     "7.1.08.tiff",
    # ]
    arithmetic_ratios = []
    huffman_ratios = []
    arithmetic_encoding_ratios = []
    huffman_encoding_ratios = []
    arithmetic_time = []
    huffman_time = []

    ##
    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.numpy())]
    )

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    num_images = 10

    for i, data in enumerate(tqdm(trainloader, desc="Processing images")):
        if i == num_images:
            break
        image = data[0][0]
        image = image[0]
        image = image.numpy()

        # Convert image to 8-bit
        image = (image * 255).astype(np.uint8)

        start = time.time()
        arithmetic = ArithmeticCoded(image)
        arithmetic_time.append(time.time() - start)

        start = time.time()
        huffman = HuffmanCoded(image)
        huffman_time.append(time.time() - start)

        arithmetic_encoding_ratios.append(
            sizeCalculator(arithmetic.encoding) / (np.prod(image.shape) * 8)
        )
        huffman_encoding_ratios.append(
            sizeCalculator(huffman.encoding) / (np.prod(image.shape) * 8)
        )
        arithmetic_ratios.append(arithmetic.compressionRatio())
        huffman_ratios.append(huffman.compressionRatio())

    plt.bar(
        np.arange(num_images),
        arithmetic_ratios,
        width=0.35,
        label="Arithmetic",
    )
    plt.bar(
        np.arange(num_images) + 0.35,
        huffman_ratios,
        width=0.35,
        label="Huffman",
    )
    plt.ylabel("Compression Ratio")
    plt.legend()
    plt.savefig("compression_ratios.png")
    plt.show()

    plt.bar(
        np.arange(num_images),
        arithmetic_encoding_ratios,
        width=0.35,
        label="Arithmetic",
    )
    plt.bar(
        np.arange(num_images) + 0.35,
        huffman_encoding_ratios,
        width=0.35,
        label="Huffman",
    )
    plt.ylabel("Encoding Ratio")
    plt.legend()
    plt.savefig("encoding_ratios.png")
    plt.show()

    plt.bar(
        np.arange(num_images),
        arithmetic_time,
        width=0.35,
        label="Arithmetic",
    )
    plt.bar(
        np.arange(num_images) + 0.35,
        huffman_time,
        width=0.35,
        label="Huffman",
    )
    plt.ylabel("Time")
    plt.legend()
    plt.savefig("times.png")
    plt.show()
