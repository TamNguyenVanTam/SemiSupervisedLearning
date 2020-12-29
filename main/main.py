import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "../utils")
sys.path.insert(0, "../model")

from cifar_utils import load_cifar10_train_data
from cifar_utils import load_cifar10_test_data
from cifar_utils import load_cifar100
from cifar_utils import get_data

from cifar10_model import *
from center_loss_layer import CenterLoss

from data_generator import generator

def read_json_file(filename):
	with open(filename) as file:
		contents = json.load(file)
	if contents == None:
		print("Meeting wrong went read {}".format(filename))
		exit(0)
	return contents

import argparse
parser = argparse.ArgumentParser(description="arguments of traffic jam project")
parser.add_argument("--setting", dest="setting", default="configuration.json")
args = parser.parse_args()

if __name__ == "__main__":
	merged_config = read_json_file(args.setting)
	merged_config["input_shape"] = (32, 32, 3)
	"""
	MODEL DEFINITION
	"""
	images = keras.Input(shape=merged_config["input_shape"])
	labels = keras.Input(shape=(merged_config["num_classes"],))
	# extract features
	emb_features = extract_features(inputs=images, alpha=0.1, dropout_rate=0.5)
	# Classification Task
	logits = Dense(merged_config["num_classes"], activation="softmax", name="logit_out")(emb_features)
	center = CenterLoss(feature_dims=128,
						num_classes=merged_config["num_classes"],
						learning_rate=1e-4,
						name="center_out")([emb_features, labels])

	model = keras.Model(inputs=[images, labels], outputs=[logits, center])
	
	optimizer = tf.keras.optimizers.Adam(lr=0.001)
	model.compile(
		optimizer=optimizer,
		loss={
			"logit_out": tf.keras.losses.CategoricalCrossentropy(),
			"center_out": tf.keras.losses.MeanSquaredError()
			},
		metrics={
			"logit_out": tf.keras.metrics.CategoricalAccuracy(),
			"center_out": tf.keras.metrics.MeanSquaredError()
			},
		loss_weights={
			"logit_out": 1.0,
			"center_out": 0.05
		}
	)
	model.summary()
	"""
	DATA PREPARATION
	"""
	train_data, train_files, train_labels = load_cifar10_train_data(merged_config["data_dir"])
	x_test, _, y_test = load_cifar10_test_data(merged_config["data_dir"])

	labeled_data, unlabeled_data = get_data(train_data, train_files,
											train_labels, merged_config["exp_dir"],
											exp_number=merged_config["exp_number"])

	x_labeled_train, y_labeled_train = labeled_data[0], labeled_data[1]
	x_unlabeled_train = unlabeled_data[0]
	y_test = np.expand_dims(y_test, axis=1)
	y_labeled_train = np.expand_dims(y_labeled_train, axis=1)
	
	# Convert to One-Hot Data format
	encoder = OneHotEncoder().fit(y_labeled_train)
	y_labeled_train = encoder.transform(y_labeled_train).toarray()
	y_test = encoder.transform(y_test).toarray()

	x_labeled_train = x_labeled_train.astype(np.int8)
	x_test = x_test.astype(np.int8)

	# x_test = x_test[0:500]
	# y_test = y_test[0:500]
	
	tmp_label_train = np.zeros((x_labeled_train.shape[0], 1))
	tmp_label_test = np.zeros((x_test.shape[0], 1))
	ws_train = np.ones((x_labeled_train.shape[0], 1))

	print("The Number of Labeled Samples: {}, and The Number of Unlabeled Samples: {}".\
		format(x_labeled_train.shape[0], x_unlabeled_train.shape[0]))
	"""
	AUGMENTATION DATA
	"""
	train_data = generator(X=[x_labeled_train, y_labeled_train],
						Y=[y_labeled_train, tmp_label_train],
						aug_rate=merged_config["aug_rate"],
						batch_size=merged_config["batch_size"])

	test_data = generator(X=[x_test, y_test],
						Y=[y_test, tmp_label_test],
						aug_rate=0.0,
						shuffle=False,
						batch_size=merged_config["batch_size"])

	train_iters = int(np.ceil(x_labeled_train.shape[0]*1.0 / merged_config["batch_size"]))
	test_iters = int(np.ceil(x_test.shape[0]*1.0 / merged_config["batch_size"]))
	
	"""
	DEFINE CALLBACKS
	"""
	# earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_logit_out_categorical_accuracy',
	# 											patience=10,
	# 											mode='max')

	if not os.path.exists(merged_config["checkpoint_dir"]):
		os.makedirs(merged_config["checkpoint_dir"])

	checkpoint = os.path.join(merged_config["checkpoint_dir"],
					"model_exp{}.hdf5".format(merged_config["exp_number"]))

	savemodel = SaveModel(save_file=checkpoint,
						init_best=-np.Inf,
						monitor='val_logit_out_categorical_accuracy',
						mode="max")
	"""FIT DATA"""
	history = model.fit_generator(generator=train_data,
								epochs=merged_config["num_epochs"],
								steps_per_epoch=train_iters,
								verbose=2,
								callbacks=[savemodel],
    							validation_data=test_data,
    							validation_steps=test_iters)

