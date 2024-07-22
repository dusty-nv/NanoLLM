#!/usr/bin/env python3
import os
import json
import time
import glob
import logging
import subprocess

import torch
import numpy as np

from nano_llm.utils import AttributeDict


class DroidDataset:
    """
    DROID robot manipulator dataset from https://github.com/droid-dataset/droid
    This is for the raw version with HD video, scanning the Google Cloud bucket 
    for the desired number of episodes instead of downloading all 8.7TB at once.
    """
    BucketRLDS = "gs://gresearch/robotics/droid"
    Bucket100 = "gs://gresearch/robotics/droid_100"
    BucketRaw = "gs://gresearch/robotics/droid_raw/1.0.1"
        
    def __init__(self, bucket=BucketRaw, max_episodes=None, cache_dir="/data/datasets/droid", **kwargs):
        """
        Download the dataset from GCS up to the set number of episodes.
        """
        if not bucket:
            bucket = DroidDataset.BucketRaw
            
        self.bucket = bucket
        self.cache_dir = cache_dir
        self.episodes = {}
        
        os.makedirs(cache_dir, exist_ok=True)

        # recursively list all the metadata files
        metadata_list_path = os.path.join(cache_dir, 'metadata_list.txt')
        
        if not os.path.isfile(metadata_list_path):
            self.run(f"gsutil -m ls {bucket}/**.json | tee {metadata_list_path}")
        
        with open(metadata_list_path) as file:
            metadata_list = [x.strip() for x in file.readlines()[:-3]]  # the last 3 lines are other json files
            
        def find_metadata(name):
            for metadata in metadata_list:
                if name in metadata:
                    return metadata

        # load annotations (language instructions)
        with open(self.download("aggregated-annotations-030724.json")) as file:
            annotations = json.load(file)
            
        # build episode index
        for episode_key in annotations:
            if len(annotations[episode_key]) > 1:
                continue
                
            time_begin = time.perf_counter()
            
            metadata_remote_path = find_metadata(episode_key)
            metadata_local_path = self.download(metadata_remote_path)
            
            if not metadata_local_path:
                logging.warning(f"DroidDataset | could not find metadata for episode {episode_key}  (skipping)")
                continue
                
            with open(metadata_local_path) as file:
                episode = AttributeDict(
                    **json.load(file),
                    path = metadata_local_path,
                    instruction = list(annotations[episode_key].values())[0],
                )

            def download_files():
                for key in episode:
                    if '_path' in key and 'svo' not in key:
                        episode[key] = self.download(os.path.join(
                            os.path.dirname(metadata_remote_path), 
                            '/'.join(episode[key].split('/')[3:])
                        ))
                        
                        if not episode[key]:
                            return False
                            
                return True  
        
            if not download_files():
                continue
                
            self.episodes[episode_key] = episode
            
            if time.perf_counter() - time_begin > 1.0:
                logging.info(f"downloaded {len(self.episodes)} episodes")
            
            if max_episodes and len(self.episodes) >= max_episodes:
                break
            
        logging.success(f"downloaded DROID dataset ({len(self.episodes)} episodes)")
       
    def download(self, filename, use_cache=True):
        if not filename:
            return None
            
        filename = filename.replace(self.bucket, '').strip('/')
        local_path = os.path.join(self.cache_dir, filename).replace(':', '-')
        
        if use_cache and os.path.isfile(local_path):
            return local_path
            
        try:
            self.run(f"gsutil -m cp {'-n' if use_cache else ''} {os.path.join(self.bucket,filename)} {local_path}")
        except Exception as error:
            logging.error(f"error downloading {filename}")
            return None
            
        return local_path
          
    def run(self, cmd, executable='/bin/bash', shell=True, check=True, **kwargs):
        logging.debug(f"DroidDataset | {cmd}")
        return subprocess.run(cmd, executable=executable, shell=shell, check=check, **kwargs)


