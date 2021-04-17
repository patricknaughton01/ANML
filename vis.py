import json
import matplotlib.pyplot as plt
import argparse
import seaborn as sns


def main():
	parser = argparse.ArgumentParser(
		description="Visualize average accuracy over a run")
	parser.add_argument("paths", type=str, nargs="+",
		help="path to results (metadata.json) file(s)")
	args = parser.parse_args()
	for path in args.paths:
		with open(path, "r") as f:
			res_json = json.load(f)
		num_runs = res_json["params"]["runs"]
		x = res_json["params"]["schedule"]
		final_results = res_json["results"]["Final Results"]
		averages = [0] * len(x)
		name = res_json["params"]["model"].split(".")[0]
		for elt in final_results:
			ind = x.index(elt[0])
			# Grab the accuracy
			averages[ind] += elt[1]["0"][0] / num_runs
		plt.plot(x, averages, label=name)
	plt.ylim([0, 1])
	plt.legend()
	plt.xlabel("Tasks")
	plt.ylabel("Accuracy")
	plt.grid()
	sns.despine()
	plt.show()


if __name__ == "__main__":
	main()
