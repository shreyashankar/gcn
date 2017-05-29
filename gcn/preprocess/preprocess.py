import json
import argparse

def getNodeFeatures(rawFile):
	"""
		Generate the N*F feature file and N*E output files
	"""
	json_object = json.load(rawFile)

	num_features = 1
	for element in json_object[0]:
		if type(element) is list or type(element) is dict:
			num_features += len(element)
		else:
			num_features += 1

	feature_array = np.empty([len(json_object), num_features])

	i = 0
	for node, node_list in json_object:
		feature_array[i,0] = node
		k = 0
		for element in node_list:
			if type(element) is list or type(element) is dict:
				for subelement in element:
					feature_array[i, k] = subelement
					k+=1
			else:
				feature_array[i, k] = element
				k+=1
		i+=1

	return feature_array

def main():
	parser = argparse.ArgumentParser(description='Preprocess neural data')
	parser.add_argument('-n','--node_json', help='Node json file',required=True)
	args = parser.parse_args()
	getNodeFeatures(args.node_json)






