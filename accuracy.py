import numpy as np
import pandas as pd

# file = "./results/clustering_MNIST012_16_5_GZIP_NCD.csv"
# file = "./results/clustering_MNIST012_16_5_EUCLID.csv"
# file = "./results/clustering_MNIST012_16_5_GZIP_HD.csv"
file = "./results/clustering_MNIST012_16_5_HUFF_NCD.csv"
# file = "./results/clustering_MNIST012_16_5_HUFF_HD.csv"
df = pd.read_csv(file)
predictions = df["predictions"].values
targets = df["targets"].values
accuracy = np.sum(predictions == targets) / len(predictions)
print("Accuracy: ", accuracy * 100)
