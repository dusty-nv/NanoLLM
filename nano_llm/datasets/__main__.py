#!/usr/bin/env python3
import sys
import logging

from nano_llm import DatasetTypes, load_dataset, convert_dataset
from nano_llm.utils import ArgParser


parser = ArgParser(extras=['log'])

parser.add_argument("--dataset", type=str, default=None, required=True, help=f"path or name of the dataset to load")
parser.add_argument("--dataset-type", type=str, default=None, choices=list(DatasetTypes.keys()), help=f"type of the dataset to load")

parser.add_argument("--max-episodes", type=int, default=None, help="the maximum number of episodes from the dataset to process")
parser.add_argument("--max-steps", type=int, default=None, help="the maximum number of frames to process across all episodes")
parser.add_argument("--rescan", action='store_true', help="rescan the dataset files for changes or rebuild the index")
parser.add_argument("--dump", type=str, default=None, help="print out and save the dataset to viewable files under this path")

parser.add_argument("--convert", type=str, default=None, choices=['rlds'], help="convert a dataset into other formats (like RLDS/TFDS)")
parser.add_argument("--output", type=str, default=None, help="path to export the converted dataset to (only when --convert is used)")
parser.add_argument("--width", type=int, default=None, help="change the image resolution when converting a dataset")
parser.add_argument("--height", type=int, default=None, help="change the image resolution when converting a dataset")
parser.add_argument("--workers", type=int, default=-1, help="the number of parallel workers to use for loading/exporting")

parser.add_argument("--remap-keys", action='append', nargs='+', default=None, help="colon-separated k:v pairs to change the names of cameras or observations, from 'old_name:new_name'.  To remove a key, use 'name:None'")
parser.add_argument("--sample-steps", type=int, default=None, help="the factor of which to subsample time steps by skipping every N")
parser.add_argument("--sample-actions", type=int, default=None, help="window size for agreggating actions across time steps")

args = parser.parse_args()

if args.dump:
    args.convert = 'dump'
    args.output = args.dump
    
if args.convert:
    convert_dataset(**vars(args), output_type=args.convert)
    sys.exit(0)
   
''' 
dataset = load_dataset(**vars(args))

if args.dump:
    logging.debug(f"dumping {args.dump} steps of data from {dataset.config.name}")
    dataset.dump(max_steps=args.dump)

logging.debug(f"{dataset.__class__.__name__} | done inspecting {dataset.config.name}")
'''   
