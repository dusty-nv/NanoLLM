#!/usr/bin/env python3
# TFDS/RLDS dataset builder template (do not import)
import os
import cv2
import threading

import tensorflow_datasets as tfds
import numpy as np

from multiprocessing.pool import ThreadPool
from collections import deque


# these are set by RLDSDataset.export()
DATASET = "_DATASET"
DATASET_TYPE = "_DATASET_TYPE"
        
MAX_EPISODES = _MAX_EPISODES
MAX_STEPS = _MAX_STEPS

WIDTH = _WIDTH
HEIGHT = _HEIGHT                
OUTPUT = "_OUTPUT"  
   
REMAP_KEYS = "_REMAP_KEYS"
SAMPLE_STEPS = _SAMPLE_STEPS
SAMPLE_ACTIONS = _SAMPLE_ACTIONS
NUM_WORKERS = _NUM_WORKERS  # number of parallel loaders

DOF = _DOF

                        
'''
'wrist_image': tfds.features.Image(
    shape=(HEIGHT, WIDTH, 3),
    dtype=np.uint8,
    encoding_format='png',
    doc='Wrist camera RGB observation.',
),
'''
                               
class RLDSDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """
    DatasetBuilder for RLDS/TFDS format
    """
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(HEIGHT, WIDTH, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
 
                        'state': tfds.features.Tensor(
                            shape=(DOF,),
                            dtype=np.float32,
                            doc='Robot state',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(DOF,),
                        dtype=np.float32,
                        doc='Robot action',
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),
        }
        
    def _generate_examples(self):
        """Generator of examples for each split."""
        self.load_dataset()

        if NUM_WORKERS and NUM_WORKERS > 1:
            print(f"Creating pool of {NUM_WORKERS} worker threads")
            
            worker_pool = ThreadPool(NUM_WORKERS) # HDF5/TFDS aren't multi-process
            worker_jobs = deque()
        
            for episode_id, episode in enumerate(self.dataset.episodes):
                worker_jobs.append(worker_pool.apply_async(self.load_episode, args=[episode, episode_id]))
                
            while len(worker_jobs) > 0:
                #print(f"{len(worker_jobs)} episodes remaining to be loaded")
                yield worker_jobs.popleft().get()
        else:
            for episode_id, episode in enumerate(self.dataset.episodes):
                yield(self.load_episode(episode, episode_id))
                
    def load_episode(self, episode, id):
        steps = []
        
        for i, step in enumerate(episode):
            obs = {}
            
            if 'state' in step:
                obs['state'] = step.state
            else:
                obs['state'] = np.array([0] * DOF, dtype=np.float32)
    
            for img_key, image in step.images.items():
                obs[img_key] = self.resize_image(image)      

            steps.append({
                'observation': obs,
                'action': step.action.astype(np.float32),
                'is_first': step.is_first,
                'is_last': step.is_last,
                'language_instruction': step.instruction
            })

        if self.sample_actions:
            for n in range(len(steps)):
                for m in range(n+1, min(n+sample_actions, len(steps))):
                    steps[n]['action'][:-1] += steps[m]['action'][:-1]
         
        if self.sample_steps:
            steps = [steps[n] for n in range(0, len(steps), self.sample_steps)]
            steps[-1]['is_last'] = True
            
        #print(f"Thread {threading.get_ident()} loaded episode with {len(steps)} steps")
        
        return f"episode_{id}", {
                'steps': steps,
                'episode_metadata': {
                    'file_path': OUTPUT,
                }}
                
    def load_dataset(self):
        from nano_llm import load_dataset
        from nano_llm.utils import convert_tensor

        self.dataset = load_dataset(
            DATASET, 
            dataset_type=DATASET_TYPE, 
            max_episodes=MAX_EPISODES,
            max_steps=MAX_STEPS,
            remap_keys=REMAP_KEYS,
            width=WIDTH,
            height=HEIGHT,
        )

        step = next(iter(self.dataset))
        
        self.output = OUTPUT
        self.width = WIDTH
        self.height = HEIGHT
        self.convert_tensor = convert_tensor
        
        if not self.width or not self.height:
            size = step.images[0].shape
            self.width = size[-2]
            self.height = size[-3]
          
        self.dof = len(step.action)
        
        self.sample_steps = SAMPLE_STEPS
        self.sample_actions = SAMPLE_ACTIONS
                       
    def resize_image(self, image):
        image = self.convert_tensor(image, return_tensors='np')
        
        img_width = image.shape[-2]
        img_height = image.shape[-3]
        
        if self.width == img_width and self.height == img_height:
            return image

        if self.width < img_width and self.height < img_height:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        
        return cv2.resize(image, (self.height, self.width), interpolation=interpolation)
 
