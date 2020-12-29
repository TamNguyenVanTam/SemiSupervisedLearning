import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer, LeakyReLU
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D, MaxPool2D
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import OneHotEncoder


def extract_features(inputs, alpha=0.1, dropout_rate=0.5):
	"""
	Build Deep Learning Architecture - 13 CNN layers
	
	The architecture is followed the paper 
		https://arxiv.org/abs/1904.04717

	Params:
		inputs: Keras Object
		dropout: Float
		num_classes: Integer
	Returns:
		outputs: Keras Object
		embedded_features: Keras Object
	"""
	# First Block
	x = WeightNormalization(Conv2D(filters=128, kernel_size=3, padding="same"))(inputs)
	x = LeakyReLU(alpha=alpha)(x)
	x = BatchNormalization()(x)
	x = WeightNormalization(Conv2D(filters=128, kernel_size=3, padding="same"))(x)
	x = LeakyReLU(alpha=alpha)(x)
	x = BatchNormalization()(x)
	x = WeightNormalization(Conv2D(filters=128, kernel_size=3, padding="same"))(x)
	x = LeakyReLU(alpha=alpha)(x)
	x = BatchNormalization()(x)
	x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)
	x = Dropout(dropout_rate)(x)

	# Second Block
	x = WeightNormalization(Conv2D(filters=256, kernel_size=3, padding="same"))(x)
	x = LeakyReLU(alpha=alpha)(x)
	x = BatchNormalization()(x)
	x = WeightNormalization(Conv2D(filters=256, kernel_size=3, padding="same"))(x)
	x = LeakyReLU(alpha=alpha)(x)
	x = BatchNormalization()(x)
	x = WeightNormalization(Conv2D(filters=256, kernel_size=3, padding="same"))(x)
	x = LeakyReLU(alpha=alpha)(x)
	x = BatchNormalization()(x)
	x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)
	x = Dropout(dropout_rate)(x)

	# Third Block
	x = WeightNormalization(Conv2D(filters=512, kernel_size=3, padding="valid"))(x)
	x = LeakyReLU(alpha=alpha)(x)
	x = BatchNormalization()(x)
	x = WeightNormalization(Conv2D(filters=256, kernel_size=1, padding="valid"))(x)
	x = LeakyReLU(alpha=alpha)(x)
	x = BatchNormalization()(x)
	x = WeightNormalization(Conv2D(filters=128, kernel_size=1, padding="valid"))(x)
	x = LeakyReLU(alpha=alpha)(x)
	x = BatchNormalization()(x)
	x = AveragePooling2D(pool_size=(6, 6), strides=(2, 2), padding="valid")(x)

	embedded_features = Flatten()(x)
	return embedded_features

class SaveModel(keras.callbacks.Callback):

	def __init__(self, save_file, init_best, monitor, mode, **kwargs):
		super(SaveModel, self).__init__(**kwargs)
		self._save_file = save_file
		self._best = init_best
		self._monitor = monitor
		self._mode = mode

	def on_epoch_end(self, epoch, logs=None):
		current = logs.get(self._monitor)
		updated = False
		if self._mode == "max":
			updated = True if current > self._best else False
		elif self._mode == "min":
			updated = True if current < self._best else False
		else:
			raise Exception("mode must be 'max' or 'min' but {}".format(mode))
			
		if updated:
			print("\nEpoch {} update from {:.2f} to {:.2f}\n".format(epoch+1, self._best*100, current*100))
			self._best = current
			self.model.save(self._save_file)
