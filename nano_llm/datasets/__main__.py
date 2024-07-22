#!/usr/bin/env python3
import logging

from nano_llm import DatasetTypes, load_dataset
from nano_llm.utils import ArgParser


parser = ArgParser(extras=['log'])

parser.add_argument("--dataset", type=str, default=None, required=True, help=f"path or name of the dataset to load")
parser.add_argument("--dataset-type", type=str, default=None, choices=list(DatasetTypes.keys()), help=f"type of the dataset to load")
parser.add_argument("--max-episodes", type=int, default=None, help="the maximum number of episodes from the dataset to process")
parser.add_argument("--max-steps", type=int, default=None, help="the maximum number of frames to process across all episodes")
parser.add_argument("--rescan", action='store_true', help="rescan the dataset files for changes or rebuild the index")

args = parser.parse_args()

dataset = load_dataset(**vars(args))

logging.debug(f"dumping {args.max_steps} steps of data from {dataset.config.name}")
dataset.dump(max_steps=args.max_steps)
 
logging.debug(f"{dataset.__class__.__name__} | done inspecting {dataset.config.name}")
    
