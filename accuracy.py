import numpy as np
import pandas as pd

# file = "./results/clustering_MNIST012_16_5_GZIP_NCD.csv"
# file = "./results/clustering_MNIST012_16_5_EUCLID.csv"
# file = "./results/clustering_MNIST012_16_5_GZIP_HD.csv"
# file = "./results/clustering_MNIST012_16_5_HUFF_NCD.csv"
# file = "./results/clustering_MNIST012_16_5_HUFF_HD.csv"
file = "./results/knn_MNIST_2_GZIP_NCD.csv"
df = pd.read_csv(file)

columns = [col for col in df.columns if "actual_targets" not in col]
targets = df["actual_targets"].values

for coloum in columns:
    predictions = df[coloum].values
    accuracy = np.sum(predictions == targets) / len(predictions)
    print(f"{coloum}| Accuracy: ", accuracy * 100)
