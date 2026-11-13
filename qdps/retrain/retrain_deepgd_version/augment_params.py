"""Per-dataset transformation ranges for the generated test sets.

Copied from ATS (Gao et al., ICSE 2022), github.com/SSCT-Lab/ATS,
``gen_data/{Mnist,Fashion,Cifar,Svhn}Dau.py::get_dau_params``. DeepGD built its
generated test sets with these. Only the 5 transformations named in the DeepGD
paper are kept: rotation (RT), zoom (ZM), shear (SR), brightness (BR) and blur
(BL). Shift (SF) and contrast (CT) are dropped.

ATS has no ranges for Fruit-360 or TinyImageNet. Both use the CIFAR-10 ranges,
the only ATS dataset of natural colour photos of objects.
"""

# Order of the transformations in the generated files and reports.
OPS = ["RT", "ZM", "SR", "BR", "BL"]

OP_NAMES = {"RT": "rotation", "ZM": "zoom", "SR": "shear", "BR": "brightness", "BL": "blur"}

_MNIST = {
    "RT": (0, 40),                     # degrees, sign drawn at random
    "ZM": ((0.5, 2.0), (0.5, 2.0)),    # (zx range, zy range)
    "SR": (10, 40.0),                  # degrees, sign drawn at random
    "BR": 0.5,                         # max |delta| on the [0, 1] intensity scale
    "BL": "hard",                      # blur filter family
}

_FASHION = {
    "RT": (5, 20),
    "ZM": ((0.7, 1.5), (0.7, 1.5)),
    "SR": (10, 30),
    "BR": 0.5,
    "BL": "hard",
}

_CIFAR = {
    "RT": (5, 25),
    "ZM": ((0.7, 1.5), (0.7, 1.5)),
    "SR": (15, 30),
    "BR": 0.5,
    "BL": "easy",
}

_SVHN = {
    "RT": (5, 20),
    "ZM": ((0.8, 1.5), (0.8, 1.5)),
    "SR": (10, 30),
    "BR": 0.3,
    "BL": None,                        # ATS default blur family
}

DAU_PARAMS = {
    "mnist": _MNIST,
    "fashion_mnist": _FASHION,
    "cifar10": _CIFAR,
    "svhn": _SVHN,
    "fruit360": _CIFAR,                # no ATS ranges: CIFAR-10's
    "tinyimagenet": _CIFAR,            # no ATS ranges: CIFAR-10's
}

# Subjects evaluated on each dataset's generated set.
DATASET_SUBJECTS = {
    "mnist": ["mnist_LeNet1", "mnist_LeNet5"],
    "fashion_mnist": ["Fashion_mnist_LeNet4"],
    "cifar10": ["cifar10_12Conv", "cifar10_ResNet20"],
    "svhn": ["SVHN_LeNet5"],
    "fruit360": ["Fruit360_ResNet50"],
    "tinyimagenet": ["TinyImageNet_ResNet101"],
}
