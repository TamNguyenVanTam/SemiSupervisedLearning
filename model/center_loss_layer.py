import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class CenterLoss(keras.layers.Layer):
	
	def __init__(self,
				feature_dims,
				num_classes,
				learning_rate,
				**kwargs):
		"""
		Initialize Method

		Params:
			feature_dims: Integer
			num_classes: Integer
			learning_rate: Float
		Returns:
			None
		"""
		super(CenterLoss, self).__init__(**kwargs)

		self._feature_dims = feature_dims
		self._num_classes = num_classes
		self._learning_rate = learning_rate

	def build(self, input_shape):
		"""
		Declare Parameters for this layer
		"""
		self._center = self.add_weight(shape=(self._num_classes, self._feature_dims),
									initializer="random_normal", trainable=False)

	def call(self, inputs):
		"""
		Perform Center Loss Here
		"""
		[x, y] = inputs 
		# x: Embedded Features- batch_size x feature_dims, y: Labels- batch_size x num_classes
		dists = x - tf.matmul(y, self._center)
		grads = tf.matmul(-tf.transpose(y), dists)

		self._center.assign_sub(self._learning_rate * grads)

		loss = tf.norm(dists, ord=2, axis=1, keepdims=True)
		num_sams = tf.reduce_sum(y, axis=0, keepdims=True) + 1.0
		loss = loss / num_sams
		return loss

	def get_config(self):
		config = super(CenterLoss, self).get_config()
		return config
