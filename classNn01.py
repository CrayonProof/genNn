import tensorflow as tf
import numpy as np
from sklearn.datasets import load_wine

data = load_wine()

#defines how data columns will be interperated by tensorflow
def construct_features (features):
	return [tf.feature_column.numeric_column(feature) for feature in features]

def my_input_fn(data, batch_size, num_epochs, shuffle):
	feature_dict = {}
	for i in range(len(data.feature_names)):
		feature_dict[data.feature_names[i]] = data.data[:,i]

		return tf.estimator.inputs.numpy_input_fn(
			x = feature_dict,
			y = data.target,
			batch_size = batch_size,
			num_epochs = num_epochs,
			shuffle = shuffle)

my_optimizer = tf.train.AdamOptimizer()

classifier = tf.estimator.DNNClassifier(feature_columns=construct_features(data.feature_names), 
	hidden_units = [10, 10], optimizer = my_optimizer, n_classes = 3)

classifier.train(input_fn = my_input_fn(data, 178, 500, True))

pred = classifier.predict(input_fn = my_input_fn(data, 178, 1, False))

print("Train accuracy is {} percent".format(pred['accuracy']))

print(pred)