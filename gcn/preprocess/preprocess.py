import json
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
import pickle

def binary_one_hot_encode(df, col):
	le = LabelEncoder()
	y = (le.fit_transform(df[:,col])).astype(int)
	y_one_hot = np.zeros((y.size, y.max()+1))
	for i in range(0, len(y)):
		y_one_hot[i][y[i]] = 1
	rejoined = np.concatenate([df[:,:col], y_one_hot, df[:,col+1:]], axis=1)
	return rejoined

def one_hot_encode(all_features):
	le = LabelEncoder()
	all_features_float = np.zeros_like(all_features)
	for col in range(all_features.shape[1]):
		try:
			all_features_float[:,col] = all_features[:,col].astype(float)
		except ValueError:
			transformed = le.fit_transform(all_features[:,col])
			all_features_float[:,col] = transformed.reshape(all_features.shape[0],1)
	return all_features_float

def getNodeFeatures(rawFile):
	"""
		Generate the N*F feature file and N*E output files
	"""
	with open(rawFile) as data_file:    
		json_object = json.load(data_file)

	feature_array = []

	counter = 0
	for node, node_list in json_object.iteritems():
		node_array = [node]
		if(counter == 0):
			key_array = [node]
		for element_key, element_val in node_list.iteritems():
			if type(element_val) is list or type(element_val) is dict:
				for subelement_key, subelement_val in element_val.iteritems():
					node_array += [subelement_val]
					if(counter == 0):
						key_array += [element_key + "." + subelement_key]
			else:
				node_array += [element_val]
				if(counter == 0): 
					key_array += [element_key]
		feature_array.append([node_array])
		counter += 1

	df = np.asmatrix(np.concatenate(feature_array), dtype='O')
	class_index = key_array.index('Class')
	df_classes = np.column_stack([df[:,0], df[:,class_index]])
	df_out_classes = binary_one_hot_encode(df_classes, 1)

	df_feat = np.delete(df, class_index, axis=1)
	df_out_feat = one_hot_encode(df_feat)

	return df_out_classes, df_out_feat

def main():
	labels, features = getNodeFeatures("neuronsinfo.json")
	pickle.dump(labels, open( "labels.p", "wb" ))
	pickle.dump(features, open( "features.p", "wb" ))


if __name__ == '__main__':
    main()




