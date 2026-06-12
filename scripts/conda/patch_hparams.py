import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--method", required=True)
parser.add_argument("--key", required=True)
parser.add_argument("--value", required=True, type=float)
args = parser.parse_args()

with open(args.input) as f:
    hparams = json.load(f)

hparams[args.dataset][args.method][args.key] = args.value

with open(args.output, "w") as f:
    json.dump(hparams, f, indent=2)
