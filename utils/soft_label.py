import os
import numpy as np

def assign_soft_label(samples, centers):
	"""
	Assign soft-label for unlabeled data

	Params:
		samples: 2D Numpy Array (NUMBER SAMPLES x NUMBER DIMENSIONS)
		centers: 2D Numpy Array (NUMBER CLASSES x NUMBER DIMENSIONS)
	Returns:
		soft_labels: 2D Numpy Array (NUMBER SAMPLES x NUMBER CLASSES)
	"""
	# Get Distance
	num_classes = centers.shape[0]
	dis2center = []
	
	for c in range(num_classes):
		center = centers[c]
		dis = (samples - center) ** 2
		dis = np.sum(dis, axis=-1, keepdims=True)
		dis = np.sqrt(dis)
		dis2center.append(dis)
	dis2center = np.concatenate(dis2center, axis=-1)
	# Convert to softlabel
	min_dis = np.min(dis2center, keepdims=True, axis=-1)
	dis2center -= min_dis

	exp_dis2center = np.exp(-dis2center)
	normalized_factor = (np.sum(exp_dis2center, axis=-1, keepdims=True) + 1e-8)
	soft_labels = exp_dis2center / normalized_factor
	
	return soft_labels

if __name__ == "__main__":
	samples = np.array([
					[1.0, 2.0, 3.0],
					[3.0, 4.0, 5.0],
					[7.0, 8.0, 9.0],
					[2.0, 3.0, 4.0],
					[10.0, 11.0, 12.0]
				])
	centers = np.array([
					[0.0, 0.0, 0.0],
					[10.0, 10.0, 10.0]
				])
	soft_labels = assign_soft_label(samples, centers)
	print(soft_labels)
	print(np.argmax(soft_labels, axis=-1))
