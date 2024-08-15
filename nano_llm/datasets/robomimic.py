#!/usr/bin/env python3
import os
import h5py
import json
import pprint
import logging
import torchvision

import numpy as np

from nano_llm.utils import AttributeDict, KeyMap, convert_tensor


class RobomimicDataset:
    """
    Episodic state/action datasets in Robomimic/HDF5 format:
    
      https://robomimic.github.io/docs/datasets/overview.html
      https://github.com/ARISE-Initiative/robomimic/blob/master/examples/notebooks/datasets.ipynb

    These also support datasets recorded from robosuite and MimicGen.
    You can download them from the command-line like this::
    
        python3 -m robomimic.scripts.download_datasets \
            --download_dir /data/datasets/robomimic \
            --dataset_types ph \
            --hdf5_types raw \
            --tasks lift 
            
    The datasets online are often very low-resolution.  
    To regenerate them at higher resolutions, run this::
    
        python3 -m robomimic.scripts.dataset_states_to_obs \
            --dataset /data/datasets/robomimic/lift/ph/demo_v141.hdf5 \
            --output_name /data/datasets/robomimic/lift/ph/highres.hdf5 \
            --camera_names frontview robot0_eye_in_hand \
            --camera_height 512 \
            --camera_width 512 \
            --exclude-next-obs \
            --done_mode 2 \
            --compress

    And then load the path to that instead (i.e. highres.hdf5)
    """
    def __init__(self, path, max_episodes=None, max_steps=None, width=None, height=None,
                 remap_keys={}, remap_gripper=True, cache_dir="/data/datasets/robomimic", **kwargs):
        """
        Load the Robomimic dataset from the HDF5 file.
        """
        self.path = path
        self.file = h5py.File(self.path, 'r', locking=False, libver='latest')
        self.data = self.file['data']
        
        self.config = AttributeDict(json.loads(self.data.attrs['env_args']))
        self.config.name = self.config.env_name

        self.width = width
        self.height = height
        
        self.max_episodes = max_episodes
        self.num_episodes = len(self.file['data'])
        
        self.max_steps = max_steps
        self.num_steps = self.data.attrs['total']
        self.new_steps = 0
        
        self.remap_keys = KeyMap(remap_keys)
        self.remap_gripper = remap_gripper
        self.action_space = self.compute_stats()
        
        logging.info(f"Robomimic | found {self.num_episodes} episodes, {self.num_steps} steps total in {self.path}\n\nDataset Config:\n\n{pprint.pformat(self.config, indent=2)}\n\nAction Space:\n\n{pprint.pformat(self.action_space, indent=2)}\n\nImage Size:  {self.config.env_kwargs['camera_widths']} x {self.config.env_kwargs['camera_heights']}\n")

    def __len__(self):
        """
        Returns the number of timesteps or frames in the dataset, over all the episodes.
        """
        return self.num_steps
        
    def __iter__(self):
        """
        Returns an iterator over all steps (or up to max_steps if it was set) with the episodes running back-to-back.  
        `step.is_first` will be set on new episodes, and `step.is_last` will be set at the end of an episode.
        """
        for episode in self.episodes:
            for step in episode:
                yield(step)
                
    @property
    def episodes(self):
        """
        Returns an iterator over all the episodes, nested over the steps in each episode::
        
            for episode in dataset.episodes():
                for step in episode:
                    ...
        """
        def generator(episode_key):
            episode = self.data[episode_key]
            actions = self.remap(actions=np.array(episode['actions']))
            samples = int(episode.attrs['num_samples'])
            images = {}
            
            for obs_key, obs in episode['obs'].items():
                if 'image' in obs_key:
                    obs_key = obs_key.replace('_image', '').replace('robot0_', '')
                    obs_key = self.remap_keys.get(obs_key, obs_key)
                    if obs_key:
                        images[obs_key] = self.resize_images(obs[()])
             
            if 'instructions' in episode:
                instructions = episode['instructions']
            else:
                instructions = self.get_instruction(self.config.env_name)  
                 
            for i in range(samples):
                self.new_steps += 1
                
                stop = bool(self.max_steps and self.new_steps >= self.max_steps)
                step = AttributeDict(
                    #state = episode['states'][i], # MuJoCo states
                    action = actions[i],
                    images = {key : image[i] for key, image in images.items()},
                    is_first = bool(i == 0),
                    is_last = bool(i == samples-1) or stop,
                )

                if instructions:
                    if isinstance(instructions, str):
                        step.instruction = instructions
                    else:
                        step.instruction = instructions[i].decode("utf-8")

                yield(step)

                if stop:
                    return
                
        for episode_idx, episode_key in enumerate(self.data):
            if self.max_episodes and episode_idx >= self.max_episodes:
                return
            if self.max_steps and self.new_steps >= self.max_steps:
                return
            yield(generator(episode_key))
               
    def get_instruction(self, task):
        task = task.lower()
        if 'stack_three' in task:
            return "stack the red block on top of the green block, and then the blue block on top of the red block."
        elif 'stack' in task:
            return "stack the red block on top of the green block"
    
    def compute_stats(self):
        logging.debug(f"Robomimic | calculating dataset statistics ({self.num_episodes} episodes, {self.num_steps} steps)")
        
        actions = []
        
        for episode_key, episode in self.data.items():
            actions.append(self.remap(actions=np.array(episode['actions'])))
            
        actions = np.concatenate(actions, axis=0)
      
        '''
        if 'action_space' not in self.data.attrs:
            self.data.attrs['action_space'] = json.dumps(self.compute_stats())
            self.file.flush()

        self.action_space = AttributeDict(
            json.loads(self.data.attrs['action_space'])
        )
        '''
        
        return AttributeDict(
            mean = actions.mean(0).tolist(),
            std = actions.std(0).tolist(),
            max = actions.max(0).tolist(),
            min = actions.min(0).tolist(),
            q01 = np.quantile(actions, 0.01, axis=0).tolist(),
            q99 = np.quantile(actions, 0.99, axis=0).tolist()
        )
    
    def resize_images(self, images):
        img_width = images.shape[-2]
        img_height = images.shape[-3]
        
        if self.width is None:
            self.width = img_width
            
        if self.height is None:
            self.height = img_height
            
        if self.width == img_width and self.height == img_height:
            return images

        return torchvision.transforms.functional.resize(
            convert_tensor(images, return_tensors='pt', device='cuda').permute(0,3,1,2),
            (self.height, self.width)  # default is bilinear
        ).permute(0,2,3,1)

    def remap(self, action=None, actions=None, gripper=None):
        if not self.remap_gripper:
            if action is not None:
                return action
            elif actions is not None:
                return actions
            elif gripper_states is not None:
                return gripper_states
                
        if gripper is not None:  # invert from [-1,1] to [0,1]
            return 1.0 - ((gripper + 1.0) * 0.5) #(gripper + 1.0) * 0.5 #
         
        if action is not None:
            action[-1] = self.remap(gripper=action[-1])
            return action
            
            '''
            return np.concatenate([
                action[:6], self.remap(gripper=action[-1])[None]
            ], axis=0) 
            '''
            
        if actions is not None:
            for s in range(actions.shape[0]):
                actions[s,-1] = self.remap(gripper=actions[s,-1])
            return actions   
            '''
            return np.concatenate([
                actions[:, :6],
                self.remap(gripper=actions[:,-1])[:, None]
            ], axis=1)
            '''
                          
    def dump(self, max_steps=1):
        #import imageio
        for i, step in enumerate(self):
            if max_steps and i >= max_steps:
                break 
            print(f"\nStep [{i}/{max_steps}]\n")
            for key, value in step.items():
                if 'image' in key:
                    #for n, img in enumerate(value):
                    #    imageio.imwrite(f"/data/temp/robomimic_{n}.jpg", value[n])
                    value = [x.shape for x in value]
                print(f"  '{key}':  {value}")

