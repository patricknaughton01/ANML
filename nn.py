import argparse
import numpy as np
import json
from tqdm import tqdm


def main():
	parser = argparse.ArgumentParser(description="Do nearest neighbors classification")
	parser.add_argument("file", type=str, help="path to file")
	args = parser.parse_args()
	with open(args.file, "r") as f:
		lines = f.readlines()
		lines = [json.loads(line)[0] for line in lines]
	vecs = []
	print("Starting lines")
	for line in tqdm(lines):
		vecs.append([float(val) for val in line])
	training = np.array(vecs[:15*600])
	testing = np.array(vecs[15*600:])
	correct = 0
	print("Measuring accuracy")
	for i in tqdm(range(len(testing))):
		vec = testing[i:i+1, :]
		label = i // 5
		dist_vecs = training - vec
		dists = np.linalg.norm(dist_vecs, axis=1)
		true_label = np.argmin(dists) // 15
		if label == true_label:
			correct += 1
	print("Accuracy: ", correct / len(testing))


if __name__ == "__main__":
	main()
