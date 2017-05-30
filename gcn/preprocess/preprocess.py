import json
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder

def one_hot_encode(all_features):
	le = LabelEncoder()
	for col in range(all_features.shape[1]):
		try:
			all_features[:,col] = all_features[:,col].astype(float)
		except ValueError:
			print "Not a float"
			print all_features[0,col]
			all_features[:,col] = le.fit_transform(all_features[:,col])
	return all_features

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
	print df.shape
	class_index = key_array.index('Class')
	print class_index
	df_classes = np.column_stack([df[:,0], df[:,class_index]])
	print df_classes.shape
	df_feat = np.array(df[-class_index])

	df_out_classes = one_hot_encode(df_classes)
	df_out_feat = one_hot_encode(df_feat)

	return df_out_classes, df_out_feat

def main():
	getNodeFeatures("neuronsinfo.json")


if __name__ == '__main__':
    main()




