#!/usr/bin/env python3
import os
import json
import time
import imageio
import logging

import numpy as np
import torch

from nano_llm.utils import convert_tensor
from pprint import pprint, pformat


class DumpDataset:
    """
    Utility for extracting and inspecting datasets into viewable formats.
    """
    @staticmethod
    def export(dataset=None, output=None, **kwargs):
        """
        Extract the dataset into a navigable folder structure on disk of episodes containing
        the trajectories, actions, and states in JSON and the cameras in videos or image files.
        """ 
        if isinstance(dataset, str):
            from nano_llm.datasets import load_dataset
            dataset = load_dataset(dataset, **kwargs)

        start = time.perf_counter()
        steps = 0
        
        ep_idx = 0
        ep_steps = []
        ep_images = {}

        for step_idx, step in enumerate(dataset):
            logging.info(f"Episode {ep_idx}, Step {len(ep_steps)}\n\n{pformat(DumpDataset.filter_dict(step), indent=2)}\n\n")
            ep_steps.append(DumpDataset.filter_dict(step, max_array_len=1024))
            
            for img_key, img in step.images.items():
                ep_images.setdefault(img_key, []).append(convert_tensor(img, return_tensors='np'))
                    
            if output:
                ep_path = os.path.join(output, str(ep_idx))

                if step.is_last:
                    DumpDataset.save_episode(ep_path, ep_steps, ep_images)
                    
            if step.is_last:
                logging.info(f"End of episode {ep_idx}, {len(ep_steps)} steps. {'Saved to ' + ep_path if output else ''}")
                ep_idx += 1
                ep_steps = []
                ep_images = {}
                
            steps += 1

        if output:
            DumpDataset.save_episode(ep_path, ep_steps, ep_images)

        elapsed = time.perf_counter() - start
        logging.info(f"Done dumping dataset {dataset.config.name}\n\n   * {ep_idx} episodes, {steps} steps in {elapsed:.2f} sec\n   * {elapsed/ep_idx:.2f} seconds per episode\n   * {elapsed/steps*1000:.1f} ms per step\n{'   * output: ' + output if output else ''}\n")
        
    @staticmethod
    def save_episode(path, steps, images):
        """
        Save non-interleaved trajectory data to json and video/image files.
        """
        if not steps or not images:
            return
            
        os.makedirs(path, exist_ok=True)        
        
        steps_path = os.path.join(path, 'steps.json')
        
        with open(steps_path, 'w') as f:
            json.dump(steps, f, indent=2)

        for img_key, img_list in images.items():
            video_path = os.path.join(path, f'{img_key}.mp4')
            imageio.mimsave(video_path, np.stack(img_list), fps=20)
            
            for i, image in enumerate(img_list):
                image_path = os.path.join(path, f'{img_key}_{i}.jpg')
                imageio.imwrite(image_path, image)
            

    @staticmethod
    def filter_dict(obj, copy=True, max_array_len=8):
        """
        Recursively format a dict for printing metadata, by removing large array and replacing it with its dimensions.
        """
        if copy:
            obj = obj.copy()
            
        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                if value.size > max_array_len:
                    obj[key] = f"shape={value.shape} dtype={value.dtype}"
                else:
                    obj[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                if value.numel() > max_array_len:
                    obj[key] = f"shape={list(value.shape)} dtype={value.dtype} device={value.device}"
                else:
                    obj[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                if len(value) > max_array_len:
                    obj[key] = f"len={len(value)}"
            elif isinstance(value, dict):
                obj[key] = DumpDataset.filter_dict(value, copy=copy, max_array_len=max_array_len)
                
        return obj
        
