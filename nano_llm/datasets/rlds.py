#!/usr/bin/env python3
import os
import array
import logging
import subprocess

import numpy as np
import multiprocessing as mp

from glob import glob
from pprint import pprint, pformat

from .tfds import TFDSDataset
from nano_llm.utils import AttributeDict, KeyMap


class RLDSDataset(TFDSDataset):
    """
    Load a TDFS dataset in RLDS format - https://github.com/google-research/rlds
    """
    def __init__(self, path, split='train', max_episodes=None, max_steps=None, cache_dir='/data/datasets', **kwargs):
        """
        If path is a URL to an http/https server or Google Cloud Storage Bucket (gs://)
        then the TDFS dataset will first be downloaded and saved to cache_dir.
        """
        import tensorflow as tf
        self.tf = tf
        
        if max_episodes:
            split = f"{split}[:{max_episodes}]"
        
        super().__init__(path, split=split, cache_dir=cache_dir, **kwargs)

        step_raw = next(iter(next(iter(self.dataset))['steps']))
        step_img = next(iter(self))
        keys_img = list(step_img.images.keys())
        
        layout = AttributeDict(
            cameras = keys_img,
            image_size = step_img.images[keys_img[0]].shape,
            step = list(step_raw.keys()),
            action = step_raw['action'],
            observation = AttributeDict()
        )
        
        if isinstance(layout.action, tf.Tensor):
            layout.action = tf.shape(layout.action).numpy().tolist()
         
        for key, value in step_raw['observation'].items():
            if isinstance(value, tf.Tensor):
                if value.dtype == tf.string:
                    value = str
                else:
                    value = value.numpy()
                    value = (value.shape, value.dtype)
            elif hasattr(value, '__len__'):
                value = (type(value), len(value))
            else:
                value = type(value)
                
            layout.observation[key] = value
            
        self.config.update(layout)
        self.max_steps = max_steps
        
        logging.success(f"RLDSDataset | loaded {self.config.name} - episode format:\n{pformat(layout, indent=2)}")
        
    '''
        self.num_steps = 0

        for episode in iter(self.dataset):
            self.num_steps += len(episode['steps'])
            
        if self.max_steps:
            self.num_steps = min(self.num_steps, self.max_steps)

    def __len__(self):
        """
        Returns the number of timesteps or frames in the dataset, taken over all episodes.
        """
        return self.num_steps
    '''
                
    def __iter__(self):
        """
        Returns an iterator over all steps (or up to max_steps if it was set) with the episodes running back-to-back.  
        `step.is_first` will be set on new episodes, and `set.is_last` will be set at the end of an episode.
        """
        steps = 0
        for episode in iter(self.dataset):
            episode = self.filter_episode(episode)
            if not episode:
                continue
            for step in episode['steps']:
                step = self.filter_step(step)
                if not step:
                    continue
                yield(step)
                steps += 1
                if self.max_steps and steps >= self.max_steps:
                    return
                
    @property
    def episodes(self):
        """
        Returns an iterator over all the episodes, nested over the steps in each episode::
        
            for episode in dataset.episodes():
                for step in episode:
                    ...
        """
        def generator(episode):
            for step in episode['steps']:
                step = self.filter_step(step)
                if step:
                    yield(step)
                
        for episode in iter(self.dataset):
            episode = self.filter_episode(episode)
            if episode:
                yield(generator(episode))
          
    def dump(self, max_steps=1):
        """
        Print out the specified number of steps for inspecting the dataset format.
        """
        for i, step in enumerate(self):
            pprint(step, indent=2)
            if max_steps and i > max_steps:
                break

    def filter_episode(self, episode):
        """
        Override this function to implement custom filtering or transformations on each episode.
        """
        return episode
        
    def filter_step(self, step):
        """
        Apply filtering and data transformations to each step (override this for custom processing)
        """
        data = AttributeDict(
            action=step.get('action'),
            images={},
            instruction=None,
            is_first=bool(step.get('is_first')),
            is_last=bool(step.get('is_last')),
        )
        
        observation = step['observation']
        image_keys = ['image', 'wrist_image', 'agentview_rgb']

        for image_key in image_keys:
            for observation_key in observation:
                if image_key in observation_key:
                    data.images[observation_key] = observation[observation_key].numpy()

        instruction = observation.get('natural_language_instruction', step.get('language_instruction'))
        
        if instruction is not None:
            data.instruction = instruction.numpy().decode('UTF-8')
         
        if 'state' in observation:
            data.state = observation['state'].numpy()
            
        for key, value in data.items():
            value = self.filter_key(key, value)
            
            if value is None:
                logging.warning(f"RLDSDataset | episode step has missing or empty key: {key}  (skipping)")           
                return None
            else:
                data[key] = value
                        
        return data

    def filter_key(self, key, value):
        """
        Apply filtering to each data entry in the step dict (return None to exclude)
        """
        if value is None:
            return None
        elif isinstance(value, (array.array, np.ndarray)):
            if len(value) == 0:
                return None
        elif isinstance(value, self.tf.Tensor):
            if len(value) == 0:
                return None
            return value.numpy()
        elif not key.startswith('is_') and not value:
            return None
            
        return value   

    @staticmethod
    def export(dataset=None, dataset_type=None, output=None, 
               width=None, height=None, max_episodes=None, max_steps=None, 
               remap_keys=None, sample_steps=None, sample_actions=None, workers=-1, **kwargs):
        """
        Convert the episodes from the dataset into TFDS/RLDS format and save it to the output.
        
        ``remap_keys`` should be a dict or string from the original dataset's image names to the
        RLDS names 'image' and 'wrist_image" - for example, from the command line::
        
            python3 -m nano_llm.datasets \
                --dataset demos.hdf5 \
                --dataset-type robomimic \
                --convert rlds \
                --remap_keys agentview:image eye_in_hand:wrist_image \
                --output rlds/demos
                
        Unneeded or conflicting keys can be removed by setting them to None, like ``key:None``
        """
        if workers is None or workers == 0:
            workers = 1
        elif workers < 0:
            workers += mp.cpu_count()
         
        if not width:
            width = 224
            
        if not height:
            height = 224
               
        exporter = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'rlds_export.py') 
        dataset_name = os.path.basename(output)
        
        env={
            '_DATASET_TYPE': dataset_type,
            '_DATASET': dataset,
            '_OUTPUT': output,
            '_WIDTH': width,
            '_HEIGHT': height,
            '_MAX_EPISODES': max_episodes,
            '_MAX_STEPS': max_steps,
            '_REMAP_KEYS': KeyMap(remap_keys, to='str'),
            '_SAMPLE_STEPS': sample_steps,
            '_SAMPLE_ACTIONS': sample_actions,
            '_NUM_WORKERS': workers,
            '_DOF': 7,  # TODO read size from dataset
            'RLDSDatasetBuilder': dataset_name,
        }

        script = f"mkdir -p {output} ; cd {output} ; cp {exporter} {dataset_name}.py ; "

        for var, value in env.items():
            script += f"sed -i 's|{var}|{value}|g' {dataset_name}.py ; "
            
        script += f"tfds build --data_dir {output} "
        
        if workers > 1:
            script += f"--num-processes {workers} "
        
        script += "; "
        
        logging.info(f"RLDSDataset | converting {dataset} from {dataset_type} to RLDS/TFDS\n\n{pformat(env, indent=2)}\n\n{script}")
        subprocess.run(script, executable='/bin/bash', shell=True, check=True)

