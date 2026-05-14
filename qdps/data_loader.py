"""
Data loader for QDPS experiments.
Loads pre-processed data from the SETS replication package.
"""
import numpy as np
import pickle
import os

DATA_MODEL_PAIRS = [
    ("mnist", "LeNet1"),
    ("mnist", "LeNet5"),
    ("Fashion_mnist", "LeNet4"),
    ("cifar10", "12Conv"),
    ("cifar10", "ResNet20"),
    ("SVHN", "LeNet5"),
    ("Fruit360", "ResNet50"),
    ("TinyImageNet", "ResNet101"),
]

# Mapping from (dataset, model) to the folder name in Fault_clusters
FOLDER_MAP = {
    ("mnist", "LeNet1"): "mnist_LeNet1",
    ("mnist", "LeNet5"): "mnist_LeNet5",
    ("Fashion_mnist", "LeNet4"): "Fashion_mnist_LeNet4",
    ("cifar10", "12Conv"): "cifar10_12conv",
    ("cifar10", "ResNet20"): "cifar10_ResNet20",
    ("SVHN", "LeNet5"): "SVHN_LeNet5",
    ("Fruit360", "ResNet50"): "Fruit360_ResNet50",
    ("TinyImageNet", "ResNet101"): "TinyImageNet_ResNet101",
}

# Mapping from (dataset, model) to the features filename in features/ dir
FEATURES_FILE_MAP = {
    ("mnist", "LeNet1"): "features_test_mnist_lenet1.npy",
    ("mnist", "LeNet5"): "features_test_mnist_lenet5.npy",
    ("Fashion_mnist", "LeNet4"): "features_test_fashion_lenet4.npy",
    ("cifar10", "12Conv"): "features_test_cifar10_12conv.npy",
    ("cifar10", "ResNet20"): "features_test_cifar10_resnet20.npy",
    ("SVHN", "LeNet5"): "features_test_svhn_lenet5.npy",
    ("Fruit360", "ResNet50"): "features_test_fruit360_resnet50.npy",
    ("TinyImageNet", "ResNet101"): "features_test_tinyimagenet_resnet101.npy",
}


def load_subject(data_name, model_name, base_path):
    """Load all data for a given subject (dataset + model pair).

    Returns dict with keys:
        - output_probability: (n_samples, n_classes) array
        - features: (n_samples, n_features) array or None if not available
        - cluster_labels: array of cluster labels for misclassified inputs
        - mis_index: array of misclassified input indices
        - total_faults: number of unique fault clusters
        - index: list of valid test indices (excluding noisy samples)
        - n_samples: total number of test inputs
        - n_classes: number of output classes
    """
    folder = FOLDER_MAP[(data_name, model_name)]
    folder_path = os.path.join(base_path, folder)

    # Load output probabilities
    output_probability = np.load(os.path.join(folder_path, "output_probability.npy"))

    # Load features - check primary location then features/ directory
    features_path = os.path.join(folder_path, "features_test.npy")
    if os.path.exists(features_path):
        features = np.load(features_path)
    else:
        features = None
        feat_filename = FEATURES_FILE_MAP.get((data_name, model_name))
        if feat_filename:
            alt_path = os.path.join(os.path.dirname(base_path), "..", "features", feat_filename)
            if os.path.exists(alt_path):
                features = np.load(alt_path)

    # Load cluster results and misclassified indices
    if data_name == "TinyImageNet":
        with open(os.path.join(folder_path, "cluster_results.pkl"), "rb") as f:
            cluster_labels = pickle.load(f)
        with open(os.path.join(folder_path, "mis_index_test.pkl"), "rb") as f:
            val_mis_data = pickle.load(f)
        mis_index = [item[2] for item in val_mis_data]
    else:
        cluster_labels = np.load(os.path.join(folder_path, "cluster_results.npy"))
        mis_index = np.load(os.path.join(folder_path, "mis_index_test.npy"))

    if data_name == "Fruit360":
        mis_index = mis_index[0]

    # Compute total faults (unique clusters, excluding noise label -1)
    total_faults = len(set(cluster_labels)) - 1

    # Compute valid indices (exclude noisy misclassified samples)
    noisy_index = []
    for i in range(len(mis_index)):
        if cluster_labels[i] == -1:
            noisy_index.append(mis_index[i])
    all_indices = set(range(len(output_probability)))
    valid_index = list(all_indices - set(noisy_index))

    return {
        "output_probability": output_probability,
        "features": features,
        "cluster_labels": cluster_labels,
        "mis_index": mis_index,
        "total_faults": total_faults,
        "index": valid_index,
        "n_samples": len(output_probability),
        "n_classes": output_probability.shape[1],
        "data_name": data_name,
        "model_name": model_name,
    }


def compute_fdr(selected_subset, mis_index, cluster_labels, budget, total_faults):
    """Compute Fault Detection Rate for a selected subset.

    Uses the same logic as the SETS replication package:
    - Each misclassified input in the subset that belongs to a cluster counts toward that cluster
    - Noisy samples (cluster=-1) each count as a unique fault
    - FDR = unique_faults_found / min(budget, total_faults)
    """
    cluster_lab = []
    nn = -1
    mis_list = list(mis_index)
    for idx in selected_subset:
        if idx in mis_list:
            pos = mis_list.index(idx)
            if cluster_labels[pos] > -1:
                cluster_lab.append(cluster_labels[pos])
            elif cluster_labels[pos] == -1:
                cluster_lab.append(nn)
                nn -= 1

    faults_found = len(set(cluster_lab))

    # Use the "1noisy" variant (all noisy samples count as one fault)
    cluster_1noisy = cluster_lab.copy()
    for i in range(len(cluster_1noisy)):
        if cluster_1noisy[i] <= -1:
            cluster_1noisy[i] = -1
    faults_1noisy = len(set(cluster_1noisy))

    fdr = faults_1noisy / min(budget, total_faults)
    return fdr, faults_1noisy
