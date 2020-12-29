"""
IN THIS FILE, DATA GENERATOR COMBINED WITH AUGMENTATION METHODS ARE IMPLEMENTED
"""
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def augment_image(image, rate, rotation_angle=10.0):
	"""
	We use three kinds of data transfomation:
		+ Rotation Image
		+ Horizontal Flipping
		+ Vertical Flipping
	Params:
		image: 2D Numpy Array or 3D Numpy Array
		rate: Float
		rotation_angle: Float
	Returns:
		augmented_image: 2D Numpy Array or 3D Numpy Array
	"""
	decision_flag = np.random.rand()
	if decision_flag >= rate:
		return image
	else:
		pil_image = Image.fromarray(image, mode="RGB")
		# Perform Data Augmentation
		if decision_flag <= rate / 3.0:
			# Perform Ratation
			augmented_image = pil_image.rotate(np.random.rand() * rotation_angle)

		elif (decision_flag > rate / 3.0) and (decision_flag <= 2 * rate / 3.0):
			# Perform Horizontal Flipping
			augmented_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
		else:
			# Perform Vertical Flipping
			augmented_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)

		return np.array(augmented_image)

def generator(X,
			Y,
			sample_weight=None,
			aug_rate=0.2,
			shuffle=True,
			batch_size=32):
	"""
	Perform Select and Apply some data augmentation methods on origin data

	Params:
		X: List - <Images, Labels>
		Y: List - <Labels, TMP-Labels>
		sample_weight: 1D Numpy Array
		aug_rate: Float
		shuffle: Boolean
		batch_size: Integer - Batchsize
	Returns:
		<batch_images, batch_labels, batch_weights>, <batch_labels, batch_tmp_labels> 
	"""
	images, labels = X[0], X[1]
	_, tmp_labels = Y[0], Y[1]

	if sample_weight is None:
		sample_weight = np.ones(images.shape[0])

	assert images.shape[0] == labels.shape[0],\
		"The number samples of inputs and that of labels must be same"
	assert images.shape[0] == sample_weight.shape[0],\
		"The number samples of inputs and that of weights must be same"

	num_sams = images.shape[0] 
	num_iters = int(np.ceil(num_sams * 1.0 / batch_size))
	idxs = np.arange(0, num_sams)
	cur_idx = 0

	while True:
		next_idx = min(cur_idx + batch_size, num_sams)
		batch_images, batch_labels, batch_tmp_labels, batch_weights = [], [], [], []
		for i in range(cur_idx, next_idx):
			batch_images.append(augment_image(images[idxs[i]], rate=aug_rate))
			batch_labels.append(labels[idxs[i]])
			batch_tmp_labels.append(tmp_labels[idxs[i]])
			batch_weights.append(sample_weight[idxs[i]])
		# perform for the end of this epoch
		if next_idx == num_sams:
			cur_idx = 0
			if shuffle:
				idxs = np.random.permutation(num_sams)
		else:
			cur_idx = next_idx
		batch_images = np.array(batch_images) / 255.0
		batch_labels = np.array(batch_labels)
		batch_tmp_labels = np.array(batch_tmp_labels)
		batch_weights = np.array(batch_weights)

		yield ([batch_images, batch_labels], [batch_labels, batch_tmp_labels], batch_weights)

if __name__ == "__main__":
	pass
	# (x, y), _ =  tf.keras.datasets.cifar10.load_data()
	# image = x[0]
	# images = [image]
	# for i in range(8):
	# 	aug_image = augment_image(image, rate=0.5)
	# 	images.append(aug_image)
	# plt.figure(figsize=(10, 10))
	# for i, img in enumerate(images):
	# 	ax = plt.subplot(3, 3, i+1)
	# 	plt.imshow(img)
	# 	plt.axis("off")
	# plt.savefig("image.jpg")
