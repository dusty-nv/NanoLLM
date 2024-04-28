#!/usr/bin/env python3
import argparse
import pprint

from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="+", type=str)
args = parser.parse_args()
print(args)

for model in args.model:
    print(f"\nloading {model}\n")

    with safe_open(model, framework="pt", device="cpu") as f:
        pprint.pprint(f.keys())
