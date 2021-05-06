import argparse
import numpy as np


def main():
	parser = argparse.ArgumentParser(description="Average values in a file")
	parser.add_argument("file", type=str, help="path to file")
	args = parser.parse_args()
	with open(args.file, "r") as f:
		lines = f.readlines()
		lines = np.array([float(line) for line in lines])
		print("Average: ", np.mean(lines))



if __name__ == "__main__":
	main()
