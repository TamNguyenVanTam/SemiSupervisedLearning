import os
import numpy as np

from cifar_utils import load_cifar10_train_data, load_cifar10_test_data
from cifar_utils import load_cifar100

def sample_data_per_class(filenames,
                        labels,
                        num_sam_per_class):
    """
    Sample data from the given distribution

    Params:
        filenames: List of String or Array of String
        labels: List of Integer or Array of Intereger
        num_sam_per_class: Integer
    Returns:
        labeled_samples: List of String <filename label>
    """
    labeled_samples = []

    considered_classes = np.unique(labels)
    for c in considered_classes:
        c_filenames = filenames[labels == c]
        if len(c_filenames) > num_sam_per_class:
            idxs = np.random.permutation(len(c_filenames))[0:num_sam_per_class]
            c_filenames = c_filenames[idxs]
        labeled_samples += ["{} {}".format(filename, c) for filename in c_filenames]
    return labeled_samples

def sample_labeled_data(filenames,
                        labels,
                        num_sam_per_class,
                        num_sets,
                        save_dir):
    """
    Sampled Labeled Data for Semi-Supervised Learning Problem
    
    Params:
        filenames: List of String or Array of String
        labels: List of Integer or Array of Integer
        num_sam_per_class: Integer
        num_sets: Integer
        save_dir: String
    Returns:
        None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for s in range(1, num_sets+1):
        exp_name = "{}.txt".format(s)
        exp_filename = os.path.join(save_dir, exp_name)
        labeled_samples = sample_data_per_class(filenames,
                                                labels,
                                                num_sam_per_class)
        # Write to Log File
        with open(exp_filename, "w") as file:
            for labeled_sample in labeled_samples:
                file.write(labeled_sample)
                file.write("\n")

        file.close()

if __name__ == "__main__":
    np.random.seed(10)
    # Create Benchmask Datasets for CIFAR10
    CIFAR10_DATA_DIR = "../data/origin_data/cifar10/cifar-10-python/cifar-10-batches-py"
    SAMPLING_DIR = "../data/sampled_data/cifar10/"
    cifar10_exps = [10, 50, 100, 200, 400]
    train_x, train_filenames, train_labels = load_cifar10_train_data(CIFAR10_DATA_DIR)
    test_x, test_filenames, test_labels = load_cifar10_test_data(CIFAR10_DATA_DIR)
    
    for num in cifar10_exps:
        save_dir = os.path.join(SAMPLING_DIR, "EXP_{}".format(num))
        sample_labeled_data(filenames=train_filenames, labels=train_labels,
                            num_sam_per_class=num, num_sets=5,
                            save_dir=save_dir)

    # Create Benchmask Datasets for CIFAR100
    CIFAR100_DATA_DIR = "../data/origin_data/cifar100/cifar-100-python"
    SAMPLING_DIR = "../data/sampled_data/cifar100/"
    cifar10_exps = [10, 50, 100, 200, 400]
    train_x, train_filenames, train_labels = load_cifar100(CIFAR100_DATA_DIR, "train")
    test_x, test_filenames, test_labels = load_cifar100(CIFAR100_DATA_DIR, "test")
    for num in cifar10_exps:
        save_dir = os.path.join(SAMPLING_DIR, "EXP_{}".format(num))
        sample_labeled_data(filenames=train_filenames, labels=train_labels,
                            num_sam_per_class=num, num_sets=5,
                            save_dir=save_dir)
     