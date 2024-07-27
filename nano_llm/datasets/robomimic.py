#!/usr/bin/env python3
import os
import json
import pprint
import logging
import numpy as np

from nano_llm.utils import AttributeDict


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
    def __init__(self, path, max_episodes=None, max_steps=None, remap_gripper=True, cache_dir="/data/datasets/robomimic", **kwargs):
        """
        Load the Robomimic dataset from the HDF5 file.
        """
        import h5py
        
        self.path = path
        self.file = h5py.File(self.path, 'r+')
        self.data = self.file['data']
        
        self.config = AttributeDict(json.loads(self.data.attrs['env_args']))
        self.config.name = self.config.env_name

        self.episodes = sorted(list(self.data.keys()))
        
        self.num_episodes = len(self.episodes)
        self.max_episodes = max_episodes
        
        self.num_steps = self.data.attrs['total']
        self.max_steps = max_steps

        self.remap_gripper = remap_gripper
        self.action_space = self.compute_stats()
        
        logging.info(f"Robomimic | found {self.num_episodes} episodes, {self.num_steps} steps total in {self.path}\n\nAction Space:\n{pprint.pformat(self.action_space, indent=2)}\n\nImage Size:  {self.config.env_kwargs['camera_widths']} x {self.config.env_kwargs['camera_heights']}\n")

        
    def __len__(self):
        """
        Returns the number of timesteps or frames in the dataset, over all the episodes.
        """
        return self.num_steps
        
    def __iter__(self):
        """
        Returns an iterator over all steps (or up to max_steps if it was set) with the episodes running back-to-back.  
        `step.is_first` will be set on new episodes, and `set.is_last` will be set at the end of an episode.
        """
        steps = 0
        
        for ep_idx, ep_key in enumerate(self.episodes):
            if self.max_episodes and ep_idx >= self.max_episodes:
                return
            
            episode = self.data[ep_key]
            episode_len = episode.attrs['num_samples']
            
            for i in range(episode_len):
                step = AttributeDict(
                    #state = episode['states'][i], # MuJoCo states
                    action = self.remap(action=episode['actions'][i]),
                    images = [],
                    instruction = self.get_instruction(self.config.env_name),
                    is_first = (i == 0),
                    is_last = (i == episode_len-1),
                )
                
                # remap gripper from [-1,1] to [0,1]
                #step.action[-1] = max(step.action[-1], 0) 
                
                for obs_key, obs in episode['obs'].items():
                    if 'image' in obs_key:
                        step.images.append(obs[i])

                yield(step)
                steps += 1
                
                if self.max_steps and steps >= self.max_steps:
                    return

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
            actions.append(self.remap(actions=episode['actions']))
            
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
    
    def remap(self, action=None, actions=None, gripper=None):
        if not self.remap_gripper:
            if action is not None:
                return action
            elif actions is not None:
                return actions
            elif gripper_states is not None:
                return gripper_states
                
        if gripper is not None:
            return 1.0 - ((gripper + 1.0) * 0.5)
         
        if action is not None:
            return np.concatenate([
                action[:6], self.remap(gripper=action[-1])[None]
            ], axis=0) 
                 
        if actions is not None:
            return np.concatenate([
                actions[:, :6],
                self.remap(gripper=actions[:,-1])[:, None]
            ], axis=1)
                          
    def dump(self, max_steps=1):
        print('Action space:\n', pprint.pformat(self.action_space))
        
        for i, step in enumerate(self):
            if max_steps and i >= max_steps:
                break 
            print(f"\nStep [{i}/{max_steps}]\n")
            for key, value in step.items():
                if 'image' in key:
                    value = [x.shape for x in value]
                print(f"  '{key}':  {value}")

