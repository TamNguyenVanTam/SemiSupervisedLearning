import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

def unpickle(filename):
	"""
	Load The CIFAR-10 Data
	"""
	with open(filename, "rb") as file:
		data = pickle.load(file, encoding="bytes")
	return data

def load_cifar10_train_data(data_dir):
	cifar10_data, cifar10_filenames, cifar10_labels = [], [], []

	for i in range(1, 6):
		cifar_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
		cifar10_data.append(cifar_data_dict[b"data"])
		cifar10_filenames += cifar_data_dict[b"filenames"]
		cifar10_labels += cifar_data_dict[b"labels"]
	cifar10_data = np.concatenate(cifar10_data, axis=0)
	cifar10_data = np.reshape(cifar10_data, (cifar10_data.shape[0], 3, 32, 32))
	cifar10_data = cifar10_data.transpose(0, 2, 3, 1).astype(np.uint8)
	
	cifar10_filenames = np.array(cifar10_filenames)
	cifar10_labels = np.array(cifar10_labels)

	return cifar10_data, cifar10_filenames, cifar10_labels

def load_cifar10_test_data(data_dir):
	
	cifar10_data_dict = unpickle(data_dir + "/test_batch")
	cifar10_data = cifar10_data_dict[b"data"]
	cifar10_filenames = cifar10_data_dict[b'filenames']
	cifar10_labels = cifar10_data_dict[b"labels"]

	cifar10_data = np.reshape(cifar10_data, (cifar10_data.shape[0], 3, 32, 32))
	cifar10_data = cifar10_data.transpose(0, 2, 3, 1).astype(np.uint8)

	cifar10_filenames = np.array(cifar10_filenames)
	cifar10_labels = np.array(cifar10_labels)

	return cifar10_data, cifar10_filenames, cifar10_labels

def load_cifar100(data_dir, kind="train"):
	if kind not in ["train", "test"]:
		raise Exception("Kind Must be Train or Test, but {}".format(kind))
	
	name = "/train"
	if kind=="test":
		name="/test"

	cifar100_data_dict = unpickle(data_dir + name)

	cifar100_data = cifar100_data_dict[b"data"]
	cifar100_filenames = cifar100_data_dict[b"filenames"]
	cifar100_labels = cifar100_data_dict[b"fine_labels"]

	cifar100_filenames = np.array(cifar100_filenames)
	cifar100_labels = np.array(cifar100_labels)

	return cifar100_data, cifar100_filenames,cifar100_labels

def file_parser(filename):
    with open(filename, "r") as file:
        data = file.read()
    # preprocess data
    data = data.strip("\n").split("\n")
    image_files = []
    for line in data:
        image_files.append(line.split(" ")[0])
    return image_files

def get_data(train_data,
            train_filenames,
            train_labels,
            exp_dir,
            exp_number):
    """
    We devide the training set into to partions
        (labeled data and unlabeled data)

    Params:
        train_data: Numpy Array
        train_filenames: Numpy Array
        train_labels: Numpy Array
        exp_dir: String
        exp_number: Integer
    Returns:
        labeled_data: List [X, Y]
        unlabeled_data: List [X]
    """
    exp_filename = os.path.join(exp_dir, "{}.txt".format(exp_number))

    labeled_filenames = file_parser(exp_filename)

    labeled_x, labeled_y, unlabeled_x = [], [], []

    for idx, image_file in enumerate(train_filenames):
        if str(image_file) in labeled_filenames:
            labeled_x.append(train_data[idx])
            labeled_y.append(train_labels[idx])
        else:
            unlabeled_x.append(train_data[idx])

    labeled_x = np.array(labeled_x)
    labeled_y = np.array(labeled_y)
    unlabeled_x = np.array(unlabeled_x)

    labeled_data = [labeled_x, labeled_y]
    unlabeled_data = [unlabeled_x]

    return labeled_data, unlabeled_data

if __name__ == "__main__":
	pass
	# CIFAR10_DATA_DIR = "../data/cifar10/cifar-10-python/cifar-10-batches-py"
	# train_x, _, _ = load_cifar10_train_data(DATA_DIR)
	# test_x, _, _ = load_cifar10_test_data(DATA_DIR)

	# CIFAR100_DATA_DIR = "../data/origin_data/cifar100/cifar-100-python"
	# load_cifar100(CIFAR100_DATA_DIR, "test")